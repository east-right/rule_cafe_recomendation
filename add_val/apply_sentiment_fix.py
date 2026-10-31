"""Apply the two sentiment-matching fixes to an *unmodified* repository checkout.

Refuses nonmatching source and unstaged changes. Requires explicit --apply; otherwise dry run.
Only edits service/nodes/soar_recommend.py and eval_data/eval_pipeline.py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

FILES = (
    'service/nodes/soar_recommend.py',
    'eval_data/eval_pipeline.py',
)
EXPECTED_BLOBS = {
    'service/nodes/soar_recommend.py': '2c9113dcacf47f5cf424fb5b339e4e90ea63960b',
    'eval_data/eval_pipeline.py': '6e1e0ad83fbba43a25910dd7899c744fa7832fd3',
}


def git_blob_sha(text: str) -> str:
    raw = text.encode('utf-8')
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    n = source.count(old)
    if n != 1:
        raise ValueError(f'{label}: expected 1 target, found {n}; repository may have changed')
    return source.replace(old, new, 1)


def patch_service(source: str) -> str:
    start = source.index('def _load_cafe_items(')
    end = source.index('# ── 규칙 기반 추천 추론', start)
    replacement = '''def _load_cafe_items(menu_cafe_names: list[str] | None = None) -> dict[str, dict[str, set[str]]]:
    """Keep observed positive and negative keywords separate per cafe."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if menu_cafe_names:
            placeholders = ",".join("?" * len(menu_cafe_names))
            rows = conn.execute(
                f"SELECT cafe_name, keyword, sentiment FROM cafe_keywords WHERE cafe_name IN ({placeholders})",
                menu_cafe_names,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT cafe_name, keyword, sentiment FROM cafe_keywords"
            ).fetchall()
    finally:
        conn.close()

    result: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        sentiments = result.setdefault(row["cafe_name"], {"긍정": set(), "부정": set()})
        if row["sentiment"] in sentiments:
            sentiments[row["sentiment"]].add(row["keyword"])
    return result


'''
    source = source[:start] + replacement + source[end:]
    source = replace_once(source,
        '    cafe_items: dict[str, list[str]],\n',
        '    cafe_items: dict[str, dict[str, set[str]]],\n',
        'service trace annotation')
    source = replace_once(source,
        'candidates = {c for c, kws in cafe_items.items() if op_set & set(kws)}',
        'candidates = {c for c, kws in cafe_items.items() if op_set & kws["긍정"]}',
        'service positive operator')
    source = replace_once(source,
        'clean = {c for c in candidates if neg_kw not in set(cafe_items.get(c, []))}',
        'clean = {c for c in candidates if neg_kw not in cafe_items[c]["부정"]}',
        'service negative narrowing')
    source = replace_once(source,
        'filtered = {c for c in candidates if tb_kw in set(cafe_items.get(c, []))}',
        'filtered = {c for c in candidates if tb_kw in cafe_items[c]["긍정"]}',
        'service positive tiebreak')
    return source


def patch_eval(source: str) -> str:
    start = source.index('def load_cafe_items()')
    end = source.index('def load_questions()', start)
    replacement = '''def load_cafe_items() -> dict[str, dict[str, set[str]]]:
    """Match service: keep positive/negative keywords in separate sets."""
    conn = sqlite3.connect(str(CAFE_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT cafe_name, keyword, sentiment FROM cafe_keywords"
        ).fetchall()
    finally:
        conn.close()
    items: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        sentiments = items.setdefault(row["cafe_name"], {"긍정": set(), "부정": set()})
        if row["sentiment"] in sentiments:
            sentiments[row["sentiment"]].add(row["keyword"])
    return items


'''
    source = source[:start] + replacement + source[end:]
    source = replace_once(source,
        'def recommend(kw: dict, cafe_items: dict[str, list[str]]) -> tuple[list[str], list[dict]]:',
        'def recommend(kw: dict, cafe_items: dict[str, dict[str, set[str]]]) -> tuple[list[str], list[dict]]:',
        'eval recommender annotation')
    source = replace_once(source,
        '    for cafe, kws in cafe_items.items():\n        ks = set(kws)\n        op_hit = len(op & ks)',
        '    for cafe, keyword_sets in cafe_items.items():\n        positive = keyword_sets["긍정"]\n        negative = keyword_sets["부정"]\n        op_hit = len(op & positive)',
        'eval operator positive')
    source = replace_once(source,
        '        tb_hit = len(tb & ks)\n        neg_hit = len(neg & ks)',
        '        tb_hit = len(tb & positive)\n        neg_hit = len(neg & negative)',
        'eval tiebreak/negative')
    source = replace_once(source,
        '    """operator 후보 → negative 감점 → tiebreak 가점 순으로 랭킹, top-3 반환.',
        '    """Positive operator 후보 → observed-negative 감점 → positive tiebreak 가점, top-3.',
        'eval description')
    return source


def ensure_clean(root: Path) -> None:
    if not (root / '.git').exists():
        print('Note: .git not found; cannot check working-tree status')
        return
    proc = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--', *FILES],
                          capture_output=True, text=True, check=True)
    if proc.stdout.strip():
        raise RuntimeError('The target files have uncommitted changes; commit or stash first:\n' + proc.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Write the changes; otherwise dry-run')
    args = parser.parse_args()
    root = args.repo.resolve()
    ensure_clean(root)
    old = {p: (root / p).read_text(encoding='utf-8') for p in FILES}
    for path, contents in old.items():
        actual = git_blob_sha(contents)
        expected = EXPECTED_BLOBS[path]
        if actual != expected:
            raise RuntimeError(f'{path}: blob SHA mismatch (expected {expected}, found {actual}). '
                               'Do not auto-apply to an updated repository.')
    new = {FILES[0]: patch_service(old[FILES[0]]), FILES[1]: patch_eval(old[FILES[1]])}
    for p, s in new.items():
        ast.parse(s, filename=p)
        print(f'Validated {p}: {len(old[p])} -> {len(s)} characters')
    if not args.apply:
        print('DRY RUN ONLY. Add --apply to modify the two files.')
        return
    # Take copies first, then apply both. Roll back on write failure.
    backup_dir = root / 'artifacts' / 'baselines' / 'pre_sentiment_fix_sources'
    for p, s in old.items():
        dest = backup_dir / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite baseline: {dest}')
    for p, s in old.items():
        (backup_dir / p).write_text(s, encoding='utf-8')
    written = []
    try:
        for p, s in new.items():
            dest = root / p
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=dest.parent, delete=False) as f:
                f.write(s)
                temp_path = Path(f.name)
            os.replace(temp_path, dest)
            written.append(p)
    except Exception:
        for p in written:
            (root / p).write_text(old[p], encoding='utf-8')
        raise
    print('Applied sentiment matching changes to service and evaluation. No models or indexes touched.')
    print('Run tests and offline replay; DO NOT claim metric improvements until re-evaluated.')


if __name__ == '__main__':
    main()
