import sys, io, pandas as pd, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('../data/shinline_cafe_reviews_test.csv', encoding='utf-8-sig')
df = df[df['리뷰내용'].fillna('').str.strip() != '']

total_reviews = len(df)
total_cafes   = df['place_id'].nunique()
counts        = df.groupby('place_id').size()

capped        = counts.clip(upper=150)
capped_total  = int(capped.sum())
pass1_req     = math.ceil(capped_total / 5)
pass2_req     = int(pass1_req * 0.9)   # EVALUATIVE 없는 청크 ~10% 제외

# 토큰 추정
p1_input_per  = 2600   # system+fewshot+5리뷰
p1_output_per = 400
p2_input_per  = 1700   # system+fewshot+entity+원문
p2_output_per = 200

p1_in  = pass1_req * p1_input_per
p1_out = pass1_req * p1_output_per
p2_in  = pass2_req * p2_input_per
p2_out = pass2_req * p2_output_per

total_in  = p1_in  + p2_in
total_out = p1_out + p2_out

# gpt-4.1-mini Batch API 가격
price_in  = 0.20   # $ per 1M tokens
price_out = 0.80

cost_in  = total_in  / 1_000_000 * price_in
cost_out = total_out / 1_000_000 * price_out
total_cost = cost_in + cost_out

print(f"=== 데이터 규모 ===")
print(f"전체 리뷰 수     : {total_reviews:,}건")
print(f"카페 수          : {total_cafes}개")
print(f"150 초과 카페    : {(counts > 150).sum()}개")
print(f"cap 후 리뷰 수   : {capped_total:,}건")
print()
print(f"=== API 요청 수 ===")
print(f"Pass1 요청 수    : {pass1_req:,}건  ({capped_total:,} / 5)")
print(f"Pass2 요청 수    : {pass2_req:,}건  (Pass1 × 90%)")
print()
print(f"=== 토큰 추정 ===")
print(f"Pass1 입력 토큰  : {p1_in/1e6:.2f}M")
print(f"Pass1 출력 토큰  : {p1_out/1e6:.2f}M")
print(f"Pass2 입력 토큰  : {p2_in/1e6:.2f}M")
print(f"Pass2 출력 토큰  : {p2_out/1e6:.2f}M")
print(f"총 입력          : {total_in/1e6:.2f}M tokens")
print(f"총 출력          : {total_out/1e6:.2f}M tokens")
print()
print(f"=== 비용 (gpt-4.1-mini Batch API) ===")
print(f"입력 비용        : ${cost_in:.2f}")
print(f"출력 비용        : ${cost_out:.2f}")
print(f"총 예상 비용     : ${total_cost:.2f}  (~${total_cost*1.3:.2f} 오차 상한)")
