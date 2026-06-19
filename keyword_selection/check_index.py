# check_index.py 수정
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from pathlib import Path

load_dotenv(Path('.env'))
client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_HOST', 'localhost'), 'port': int(os.getenv('OPENSEARCH_PORT', 9200))}],
    http_auth=(os.getenv('OPENSEARCH_USER', 'admin'), os.getenv('OPENSEARCH_PASSWORD')),
    use_ssl=True, verify_certs=False, ssl_show_warn=False
)
result = client.get(index='cafe_rules', id='0')
print(result['_source'].keys())
print(result['_source']['operator_keywords'])
print(result['_source']['tiebreak_keywords'])