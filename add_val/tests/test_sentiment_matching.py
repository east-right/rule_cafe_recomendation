"""Dependency-free regression tests for the ACTUAL source functions via AST extraction.

Run after applying patch:
  CAFE_REPO_ROOT=/path/to/repo python -m unittest discover -s tests -v
The AST extraction deliberately avoids imports of model/LLM serving dependencies.
"""
import ast
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(os.environ.get('CAFE_REPO_ROOT', '.')).resolve()


def extracted(path, *names):
    src = (ROOT / path).read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(path))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(body) != len(names):
        raise ValueError(f'{path}: cannot find {names}')
    ns = {'sqlite3': sqlite3, 'DB_PATH': None, 'CAFE_DB': None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 filename=str(path), mode='exec'), ns)
    return ns


class SentimentMatchingRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.temp.name) / 'cafe.db'
        with sqlite3.connect(cls.db) as db:
            db.execute('CREATE TABLE cafe_keywords(cafe_name TEXT, keyword TEXT, sentiment TEXT)')
            db.executemany('INSERT INTO cafe_keywords VALUES(?,?,?)', [
                ('P', '공부', '긍정'), ('P', '콘센트', '긍정'),
                ('N', '공부', '부정'),  # Must NOT qualify as positive operator.
                ('B', '공부', '긍정'), ('B', '콘센트', '부정'), ('B', '소음', '부정'),
                ('C', '공부', '긍정'), ('C', '콘센트', '긍정'), ('C', '소음', '긍정'),
                ('D', '주차장', '부정'), ('E', '주차장', '긍정'),
                ('M', '공부', '긍정'), ('M', '공부', '부정'),
            ])
        cls.service = extracted('service/nodes/soar_recommend.py', '_load_cafe_items', '_trace_tiebreak')
        cls.eval = extracted('eval_data/eval_pipeline.py', 'load_cafe_items', 'recommend')
        cls.service['DB_PATH'] = cls.db
        cls.eval['CAFE_DB'] = cls.db
        cls.eval['TOP_N'] = 3  # 모듈 상수: 격리 exec 네임스페이스에 주입 (recommend가 참조)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_loader_separates_sentiment(self):
        s = self.service['_load_cafe_items']()
        e = self.eval['load_cafe_items']()
        self.assertEqual(s, e)
        self.assertEqual(s['N']['긍정'], set())
        self.assertEqual(s['N']['부정'], {'공부'})
        self.assertEqual(s['M']['긍정'], {'공부'})
        self.assertEqual(s['M']['부정'], {'공부'})

    def test_negative_only_keyword_not_positive_operator(self):
        items = self.service['_load_cafe_items']()
        t = self.service['_trace_tiebreak'](['주차장'], [], [], items)
        self.assertEqual(t[0]['remaining'], ['E'])
        self.assertNotIn('D', t[0]['remaining'])

    def test_negative_narrowing_does_not_penalize_positive_only(self):
        items = self.service['_load_cafe_items']()
        t = self.service['_trace_tiebreak'](['공부'], [], ['소음'], items)
        self.assertEqual(t[0]['remaining'], ['B', 'C', 'M', 'P'])
        self.assertEqual(t[1]['remaining'], ['C', 'M', 'P'])
        self.assertNotIn('B', t[-1]['remaining'])
        self.assertIn('C', t[-1]['remaining'])  # C only has positive '소음'

    def test_tiebreak_positive_not_negative(self):
        items = self.service['_load_cafe_items']()
        t = self.service['_trace_tiebreak'](['공부'], ['콘센트'], [], items)
        self.assertEqual(t[-1]['remaining'], ['C', 'P'])
        self.assertNotIn('B', t[-1]['remaining'])

    def test_menu_restricted_candidates_still_separate_sentiment(self):
        items = self.service['_load_cafe_items'](['B', 'C'])
        t = self.service['_trace_tiebreak']([], [], ['소음'], items, menu_mode=True)
        self.assertEqual(t[-1]['remaining'], ['C'])

    def test_eval_operator_and_scoring_use_correct_sentiment(self):
        items = self.eval['load_cafe_items']()
        rank, detail = self.eval['recommend'](
            {'operator': ['공부'], 'tiebreak': ['콘센트'], 'negative': ['소음']}, items)
        self.assertNotIn('N', rank)
        self.assertTrue(all(x['cafe'] != 'N' for x in detail))
        b = next(x for x in detail if x['cafe'] == 'B') if any(x['cafe'] == 'B' for x in detail) else None
        if b:
            self.assertEqual(b['tb'], 0)
            self.assertEqual(b['neg'], 1)
        self.assertIn('C', rank)


if __name__ == '__main__':
    unittest.main()
