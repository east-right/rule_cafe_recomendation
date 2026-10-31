"""Capture actual LangGraph state/Trace for a fixed question set with writes disabled.

Requires the user's running OpenSearch/model server and OpenAI credentials.
DO NOT run against production without reviewing the consequences/costs; LLM API calls
may be made by the graph. The two currently-known rule-index writes are monkeypatched
away: rule_create._persist_rule, impasse_resolve._update_os_tiebreak.

Does NOT call final answer generation, so it captures model/graph decisions, not
final user-visible responses. Run once on unmodified main, once on the patched branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='For quick testing; 0 means all 50')
    a = p.parse_args()
    root = a.repo.resolve()
    question_path = root / 'eval_data/data/questions_eval.json'
    questions = json.loads(question_path.read_text(encoding='utf-8'))
    if a.limit:
        questions = questions[:a.limit]
    sys.path.insert(0, str(root))
    # OpenSearch & model server read operations remain enabled. Calls to these
    # two write functions are disabled for the entire graph invocation.
    from service.main import run  # noqa: E402
    from service.nodes import impasse_resolve, rule_create  # noqa: E402
    def no_write(*args, **kwargs):
        return None
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        diff = subprocess.check_output(['git', '-C', str(root), 'diff', '--',
                                        'service/nodes/soar_recommend.py',
                                        'eval_data/eval_pipeline.py'], text=True)
        code_modified = bool(diff)
    except Exception:
        commit = 'unknown'
        code_modified = None
    report = {
        'metadata': {
            'git_commit': commit,
            'tracked_source_modified': code_modified,
            'db_sha256': sha(root / 'data/cafe.db'),
            'questions_sha256': sha(question_path),
            'notes': 'Actual LangGraph invocation: classification, RAG, model, fallbacks. '
                     'Two known OpenSearch write methods disabled. Final answer generation not executed. '
                     'External LLM/model calls and Langfuse observations may still happen.',
        },
        'results': [],
    }
    with patch.object(rule_create, '_persist_rule', no_write), \
         patch.object(impasse_resolve, '_update_os_tiebreak', no_write):
        for i, q in enumerate(questions, 1):
            print(f'[{i}/{len(questions)}] {q.get("qid")}: {q.get("question")}')
            rec = {'qid': q.get('qid'), 'question': q.get('question')}
            try:
                state = run(q['question'])
                outcome = state.get('soar_result') or {}
                rec.update({
                    'question_type': state.get('question_type'),
                    'selected_rule': state.get('selected_rule'),
                    'rule_candidates': state.get('rule_candidates'),
                    'fallback_rules': state.get('fallback_rules'),
                    'rule_fallback_used': state.get('rule_fallback_used'),
                    'rule_created_but_not_persisted': state.get('rule_created'),
                    'extracted_menu': state.get('extracted_menu'),
                    'menu_cafe_names': state.get('menu_cafe_names'),
                    'impasse_iterations': state.get('impasse_iterations'),
                    'outcome': outcome,
                })
            except Exception as e:
                rec['error'] = f'{type(e).__name__}: {e}'
            report['results'].append(rec)
    report['summary'] = {
        'questions': len(report['results']),
        'errors': sum('error' in r for r in report['results']),
        'none_rules': sum(r.get('selected_rule') == 'none' for r in report['results']),
        'recommendations': sum(r.get('outcome', {}).get('type') == 'recommendation' for r in report['results']),
        'impasses': sum(r.get('outcome', {}).get('type') == 'impasse' for r in report['results']),
    }
    dest = a.out.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite: {dest}')
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'output': str(dest), **report['summary']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
