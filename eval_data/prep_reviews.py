"""리뷰 원본 중복 제거 → 정리본 저장.

- 완전중복(사업장명+리뷰내용+작성일) 제거: 같은 리뷰가 두 번 긁힌 크롤링 아티팩트만 제거.
  (같은 말이라도 날짜가 다르면 서로 다른 사람 리뷰로 보고 보존)
- NaN 리뷰내용 제거.
- 원본은 건드리지 않고 data/reviews_clean.csv 로 저장.
"""
from pathlib import Path

import pandas as pd

import batch_runner as br  # stdout utf-8 + ROOT

ROOT = br.ROOT
SRC = ROOT / "data" / "shinline_cafe_reviews_test.csv"
DST = ROOT / "data" / "reviews_clean.csv"

df = pd.read_csv(SRC)
n0 = len(df)

df = df.dropna(subset=["리뷰내용"])
n_nan = n0 - len(df)

before_per_cafe = df.groupby("사업장명").size()
dup_full = df.duplicated(subset=["사업장명", "리뷰내용", "작성일"]).sum()

clean = df.drop_duplicates(subset=["사업장명", "리뷰내용", "작성일"]).reset_index(drop=True)
clean.to_csv(DST, index=False, encoding="utf-8-sig")

after_per_cafe = clean.groupby("사업장명").size()
removed = (before_per_cafe - after_per_cafe).sort_values(ascending=False)

print(f"원본 행: {n0}")
print(f"NaN 리뷰 제거: {n_nan}")
print(f"완전중복 제거: {dup_full}")
print(f"정리본 행: {len(clean)}  (매장 {clean['사업장명'].nunique()}곳)")
print(f"\n저장: {DST}")
print("\n=== 중복 많이 제거된 매장 top10 ===")
for name, r in removed.head(10).items():
    if r > 0:
        print(f"  -{int(r):3d}  {name}  ({before_per_cafe[name]}→{after_per_cafe[name]})")
print(f"\n중복 있던 매장 수: {(removed > 0).sum()} / {len(after_per_cafe)}")
