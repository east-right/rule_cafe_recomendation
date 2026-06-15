# check_sample.py
import json, sys
sys.path.insert(0, 'soar/Make_Rule')
from generate_rule import generate_soar_rule

with open('data/soar_rule_keywords.json', encoding='utf-8') as f:
    rules = json.load(f)

rule = next(r for r in rules if r['title'] == '넓은공간')
content = generate_soar_rule(rule)

with open('soar/Make_Rule/soar_rules_sample/넓은공간.soar', 'w', encoding='utf-8') as f:
    f.write(content)

print("샘플 생성 완료")