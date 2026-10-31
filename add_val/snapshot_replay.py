"""Frozen-selected-rule, CPU-only recommendation trace replay and DB conflict audit.

This is NOT an end-to-end service run. Selected rule titles come from
`eval_data/data/pipeline_eval_results.json`; this tool does not rerun RAG,
classification, menu paths, LLM fallback, impasse LLM or final answer generation.
The legacy algorithm is frozen to main @ 0ba0670. No OpenSearch writes or API calls.

Usage:
  python snapshot_replay.py --repo PATH --mode baseline --out before.json
  python snapshot_replay.py --repo PATH --mode corrected --out after.json
  python snapshot_replay.py --repo PATH --mode compare --out comparison.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path

PINNED_SOURCE_COMMIT = '0ba0670b9a84791ce2d459c3f8fa03d2080d41ef'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_items(db: Path):
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
        rows = conn.execute('SELECT cafe_name, keyword, sentiment FROM cafe_keywords').fetchall()
    sets = defaultdict(lambda: {'긍정': set(), '부정': set()})
    for cafe, keyword, sentiment in rows:
        if sentiment in ('긍정', '부정'):
            sets[cafe][sentiment].add(keyword)
    return dict(sets), len(rows)


def audit(items, rules):
    op = set().union(*(set(x.get('operator_keywords', [])) for x in rules))
    tb = set().union(*(set(x.get('tiebreak_keywords', [])) for x in rules))
    neg = set().union(*(set(x.get('negative_keywords', [])) for x in rules))
    matched_negative = [(name, kw) for name, sets in items.items()
                        for kw in sets['부정'] if kw in op | tb]
    negative_without_positive = [(name, kw) for name, kw in matched_negative
                                 if kw not in items[name]['긍정']]
    positive_wrong_negative = [(name, kw) for name, sets in items.items()
                               for kw in sets['긍정'] if kw in neg and kw not in sets['부정']]
    both = [(name, kw) for name, sets in items.items()
            for kw in sets['긍정'] & sets['부정']]
    return {
        'negative_rows_matching_positive_rule_vocab': len(matched_negative),
        'negative_only_pairs_matching_positive_rule_vocab': len(negative_without_positive),
        'positive_only_pairs_matching_negative_rule_vocab': len(positive_wrong_negative),
        'same_cafe_keyword_both_sentiments': len(both),
        'note': 'Unique cafe-keyword pairs from the committed SQLite database; not observed wrong recommendations.',
        'examples_negative_only': [dict(cafe=n, keyword=k) for n, k in sorted(negative_without_positive)[:12]],
    }


def trace(op, tb, neg, items, *, separated):
    # Frozen service/nodes/soar_recommend.py semantics at the pinned commit.
    if separated:
        candidates = {c for c, kws in items.items() if set(op) & kws['긍정']}
    else:
        candidates = {c for c, kws in items.items() if set(op) & (kws['긍정'] | kws['부정'])}
    steps = [{'step': 'operator', 'keyword': None, 'remaining': sorted(candidates)}]
    depth = 0
    for k in neg:
        depth += 1
        clean = {c for c in candidates if k not in (
            items[c]['부정'] if separated else items[c]['긍정'] | items[c]['부정'])}
        if 0 < len(clean) < len(candidates):
            candidates = clean
            steps.append({'step': f'S{depth}_neg_narrowed', 'keyword': k, 'remaining': sorted(candidates)})
        else:
            steps.append({'step': f'S{depth}_neg_no_effect', 'keyword': k, 'remaining': sorted(candidates)})
        if len(candidates) == 1:
            return steps
    for k in tb:
        depth += 1
        filtered = {c for c in candidates if k in (
            items[c]['긍정'] if separated else items[c]['긍정'] | items[c]['부정'])}
        if len(filtered) == 1:
            candidates = filtered
            steps.append({'step': f'S{depth}_resolved', 'keyword': k, 'remaining': sorted(candidates)})
            break
        elif len(filtered) > 1:
            candidates = filtered
            steps.append({'step': f'S{depth}_narrowed', 'keyword': k, 'remaining': sorted(candidates)})
        else:
            steps.append({'step': f'S{depth}_no_effect', 'keyword': k, 'remaining': sorted(candidates)})
    return steps


def rank(kw, items, *, separated):
    op, tb, neg = (set(kw[x]) for x in ('operator', 'tiebreak', 'negative'))
    scored = []
    for name, sets in items.items():
        positive = sets['긍정'] if separated else sets['긍정'] | sets['부정']
        negative = sets['부정'] if separated else sets['긍정'] | sets['부정']
        op_hit = len(op & positive)
        if not op_hit:
            continue
        tb_hit = len(tb & positive)
        neg_hit = len(neg & negative)
        scored.append((tb_hit - neg_hit, op_hit, tb_hit, neg_hit, name))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [s[4] for s in scored[:3]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--mode', choices=('baseline', 'corrected', 'compare'), default='compare')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = args.repo.resolve()
    db = root / 'data/cafe.db'
    eval_path = root / 'eval_data/data/pipeline_eval_results.json'
    rule_path = root / 'data/soar_rule_keywords.json'
    items, row_count = read_items(db)
    evaluation = json.loads(eval_path.read_text(encoding='utf-8'))
    rules = json.loads(rule_path.read_text(encoding='utf-8'))
    kw_map = {r['title']: {'operator': r.get('operator_keywords', []),
                           'tiebreak': r.get('tiebreak_keywords', []),
                           'negative': r.get('negative_keywords', [])}
              for r in rules}
    try:
        sha = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = 'unknown'
    doc = {
        'metadata': {
            'source_commit_at_analysis': PINNED_SOURCE_COMMIT,
            'local_checkout_commit': sha,
            'db_sha256': digest(db), 'evaluation_sha256': digest(eval_path),
            'rule_sha256': digest(rule_path),
            'mode': args.mode, 'db_keyword_rows': row_count, 'cafe_count': len(items),
            'caveat': 'Offline frozen-selected-rule replay, NOT production end-to-end inference. '
                      'No OpenSearch, live model, menu filter, impasse LLM, tier-1/2 fallback or answer generation.',
        },
        'existing_eval_summary_not_recomputed': evaluation.get('summary', {}),
        'existing_eval_none_count': evaluation.get('none_count'),
        'original_eval_results_copy': evaluation,  # Freeze all 50 prior top-3/results before rerun.
        'db_audit': audit(items, rules), 'questions': [],
    }
    changes = 0
    for q in evaluation['results']:
        title = q.get('selected_rule') or ''
        out = {'qid': q.get('qid'), 'question': q.get('question'), 'selected_rule': title}
        if title not in kw_map:
            out['status'] = 'no_valid_rule'; doc['questions'].append(out)
            continue
        kw = kw_map[title]
        for mode in (['baseline','corrected'] if args.mode == 'compare' else [args.mode]):
            seg = mode == 'corrected'
            t = trace(kw['operator'], kw['tiebreak'], kw['negative'], items, separated=seg)
            remaining = t[-1]['remaining']
            typ = 'recommendation' if len(remaining) == 1 else 'impasse' if remaining else 'no_output'
            out[mode] = {'type': typ, 'cafe': remaining[0] if len(remaining) == 1 else None,
                         'impasse_candidates': remaining[:2] if len(remaining)>1 else [],
                         'trace': t, 'offline_top3': rank(kw, items, separated=seg)}
        if 'baseline' in out and 'corrected' in out:
            out['changed_trace'] = out['baseline']['trace'] != out['corrected']['trace']
            out['changed_top3'] = out['baseline']['offline_top3'] != out['corrected']['offline_top3']
            changes += int(out['changed_trace'])
        doc['questions'].append(out)
    doc['summary'] = {'total_questions': len(doc['questions']),
                      'valid_selected_rule': sum('status' not in x for x in doc['questions']),
                      'changed_traces_in_pairwise_replay': changes if args.mode == 'compare' else None}
    dest = args.out.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite frozen snapshot: {dest}')
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(dest), **doc['summary'], 'db_audit': doc['db_audit']},
                     ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
