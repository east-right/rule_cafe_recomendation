"""Self-contained tests for patch transformations and offline snapshot runner."""
import ast
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
from apply_sentiment_fix import patch_eval, patch_service  # noqa: E402

SOURCE_SERVICE = '''import sqlite3
from pathlib import Path
DB_PATH = None

def _load_cafe_items(menu_cafe_names: list[str] | None = None) -> dict[str, list[str]]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    if menu_cafe_names:
        placeholders = ",".join("?" * len(menu_cafe_names))
        rows = conn.execute(
            f"SELECT cafe_name, keyword FROM cafe_keywords WHERE cafe_name IN ({placeholders})",
            menu_cafe_names,
        ).fetchall()
    else:
        rows = conn.execute("SELECT cafe_name, keyword FROM cafe_keywords").fetchall()
    conn.close()
    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row["cafe_name"], []).append(row["keyword"])
    return result

# ── 규칙 기반 추천 추론 (XAI) ───────
def _trace_tiebreak(
    operator_keywords: list[str],
    tiebreak_keywords: list[str],
    negative_keywords: list[str],
    cafe_items: dict[str, list[str]],
    menu_mode: bool = False,
) -> list[dict]:
    if menu_mode:
        candidates = set(cafe_items.keys())
    else:
        op_set = set(operator_keywords)
        candidates = {c for c, kws in cafe_items.items() if op_set & set(kws)}
    traces = [{"step": "operator", "keyword": None, "remaining": sorted(candidates)}]
    depth = 0
    for neg_kw in negative_keywords:
        depth += 1
        clean = {c for c in candidates if neg_kw not in set(cafe_items.get(c, []))}
        if 0 < len(clean) < len(candidates):
            candidates = clean
            traces.append({"step": f"S{depth}_neg_narrowed", "keyword": neg_kw, "remaining": sorted(candidates)})
        else:
            traces.append({"step": f"S{depth}_neg_no_effect", "keyword": neg_kw, "remaining": sorted(candidates)})
        if len(candidates) == 1:
            return traces
    for tb_kw in tiebreak_keywords:
        depth += 1
        filtered = {c for c in candidates if tb_kw in set(cafe_items.get(c, []))}
        if len(filtered) == 1:
            candidates = filtered
            traces.append({"step": f"S{depth}_resolved", "keyword": tb_kw, "remaining": sorted(candidates)})
            break
        elif len(filtered) > 1:
            candidates = filtered
            traces.append({"step": f"S{depth}_narrowed", "keyword": tb_kw, "remaining": sorted(candidates)})
        else:
            traces.append({"step": f"S{depth}_no_effect", "keyword": tb_kw, "remaining": sorted(candidates)})
    return traces
'''

SOURCE_EVAL = '''import sqlite3
CAFE_DB = None

def load_cafe_items() -> dict[str, list[str]]:
    conn = sqlite3.connect(str(CAFE_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT cafe_name, keyword FROM cafe_keywords").fetchall()
    conn.close()
    items: dict[str, list[str]] = {}
    for r in rows:
        items.setdefault(r["cafe_name"], []).append(r["keyword"])
    return items


def load_questions():
    return []

def recommend(kw: dict, cafe_items: dict[str, list[str]]) -> tuple[list[str], list[dict]]:
    """operator 후보 → negative 감점 → tiebreak 가점 순으로 랭킹, top-3 반환.
    """
    op = set(kw["operator"])
    tb = set(kw["tiebreak"])
    neg = set(kw["negative"])
    scored = []
    for cafe, kws in cafe_items.items():
        ks = set(kws)
        op_hit = len(op & ks)
        if op_hit == 0:
            continue
        tb_hit = len(tb & ks)
        neg_hit = len(neg & ks)
        score = tb_hit - neg_hit
        scored.append((score, op_hit, tb_hit, neg_hit, cafe))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    ranked = [s[4] for s in scored[:3]]
    detail = [
        {"cafe": s[4], "score": s[0], "op": s[1], "tb": s[2], "neg": s[3]}
        for s in scored[:3]
    ]
    return ranked, detail
'''


class PatcherAndReplayTests(unittest.TestCase):
    def test_transform_matches_original_excerpts(self):
        a = patch_service(SOURCE_SERVICE)
        b = patch_eval(SOURCE_EVAL)
        ast.parse(a)
        ast.parse(b)
        self.assertIn('if op_set & kws["긍정"]', a)
        self.assertIn('neg_kw not in cafe_items[c]["부정"]', a)
        self.assertIn('tb_kw in cafe_items[c]["긍정"]', a)
        self.assertIn('neg_hit = len(neg & negative)', b)

    def test_mocks_execute_actual_patched_functions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'service/nodes').mkdir(parents=True)
            (root / 'eval_data').mkdir()
            (root / 'service/nodes/soar_recommend.py').write_text(patch_service(SOURCE_SERVICE), encoding='utf8')
            (root / 'eval_data/eval_pipeline.py').write_text(patch_eval(SOURCE_EVAL), encoding='utf8')
            db_path = root / 'cafe.db'
            with sqlite3.connect(db_path) as db:
                db.execute('CREATE TABLE cafe_keywords(cafe_name TEXT, keyword TEXT, sentiment TEXT)')
                db.executemany('INSERT INTO cafe_keywords VALUES (?,?,?)', [
                    ('A', '주차장', '부정'), ('B', '주차장', '긍정'),
                    ('B', '소음', '긍정'), ('C', '주차장', '긍정'), ('C', '소음', '부정'),
                ])
            ns = {'sqlite3': sqlite3}
            source = (root / 'service/nodes/soar_recommend.py').read_text('utf8')
            exec(compile(source, 'mock_source', 'exec'), ns)
            ns['DB_PATH'] = db_path
            items = ns['_load_cafe_items']()
            trace = ns['_trace_tiebreak'](['주차장'], [], ['소음'], items)
            self.assertEqual(trace[0]['remaining'], ['B', 'C'])
            self.assertEqual(trace[-1]['remaining'], ['B'])

    def test_offline_snapshot_no_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'data').mkdir()
            (root / 'eval_data/data').mkdir(parents=True)
            with sqlite3.connect(root / 'data/cafe.db') as db:
                db.execute('CREATE TABLE cafe_keywords(cafe_name TEXT, keyword TEXT, sentiment TEXT)')
                db.executemany('INSERT INTO cafe_keywords VALUES(?,?,?)', [
                    ('A', '주차장', '부정'), ('B', '주차장', '긍정')])
            (root / 'data/soar_rule_keywords.json').write_text(json.dumps([
                {'title':'parking', 'operator_keywords':['주차장'], 'tiebreak_keywords':[], 'negative_keywords':[]}]), encoding='utf8')
            (root / 'eval_data/data/pipeline_eval_results.json').write_text(json.dumps({
                'summary': {'hit1': 0.6}, 'none_count': 1,
                'results': [{'qid':'q000', 'question':'주차장 카페', 'selected_rule':'parking'},
                            {'qid':'q001', 'question':'none', 'selected_rule':'none(raw=none)'}]
            }), encoding='utf8')
            out = root / 'out.json'
            proc = subprocess.run([sys.executable, str(PACK / 'snapshot_replay.py'),
                                   '--repo', str(root), '--mode', 'compare', '--out', str(out)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(out.read_text('utf8'))
            self.assertEqual(data['summary']['changed_traces_in_pairwise_replay'], 1)
            self.assertEqual(data['questions'][0]['baseline']['trace'][0]['remaining'], ['A','B'])
            self.assertEqual(data['questions'][0]['corrected']['trace'][0]['remaining'], ['B'])
            self.assertEqual(data['db_audit']['negative_only_pairs_matching_positive_rule_vocab'], 1)


if __name__ == '__main__':
    unittest.main()
