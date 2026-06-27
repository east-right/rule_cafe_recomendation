# update_index.py
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from tqdm import tqdm

load_dotenv(Path('.env'))

client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_HOST', 'localhost'), 'port': int(os.getenv('OPENSEARCH_PORT', 9200))}],
    http_auth=(os.getenv('OPENSEARCH_USER', 'admin'), os.getenv('OPENSEARCH_PASSWORD')),
    use_ssl=True, verify_certs=False, ssl_show_warn=False
)

with open('data/soar_rule_keywords.json', encoding='utf-8') as f:
    soar_rules = {r['title']: r for r in json.load(f)}

# 기존 인덱스에서 title 기준으로 id 찾아서 업데이트
result = client.search(index='cafe_rules', body={"query": {"match_all": {}}, "size": 500})
hits = result['hits']['hits']

updated = 0
skipped = 0
for hit in tqdm(hits):
    title = hit['_source']['title']
    if title in soar_rules:
        client.update(
            index='cafe_rules',
            id=hit['_id'],
            body={
                "doc": {
                    "operator_keywords": soar_rules[title]['operator_keywords'],
                    "tiebreak_keywords": soar_rules[title]['tiebreak_keywords'],
                    "negative_keywords": soar_rules[title].get('negative_keywords', []),
                }
            }
        )
        updated += 1
    else:
        skipped += 1

print(f"완료: {updated}개 업데이트, {skipped}개 스킵 (soar_rule_keywords에 없는 rule)")