import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('../data/ner_result_v2.json', encoding='utf-8') as f:
    data = json.load(f)

for place_id, entities in data.items():
    print(f'=== place_id: {place_id} | 총 {len(entities)}개 entity ===\n')
    for e in entities:
        cat   = e['category']
        typ   = e['type']
        senti = e['sentiment']
        ent   = e['entity']
        desc  = e.get('descriptor') or '-'
        src   = (e.get('source_review') or '')[:60]
        print(f'  [{cat}][{typ}][{senti}]  {ent}')
        print(f'    descriptor : {desc}')
        print(f'    source     : {src}')
        print()
