# B4 — menu-complex 룰 none 시 Rule Fallback 우회 제거 (회귀 vs M0)

- **커밋**: `487a4d7` (`fix(B4): menu-complex 에서 룰 none 일 때 Rule Fallback 우회 제거`)
- **기준선**: `results/M0/baseline.json` (`48c67e1`)
- **재실행**: menu-complex **3문항(q052·q104·q139)** 을 B4 코드로 재실행(`rerun.json`), M0와 비교. **실측 OpenAI 15콜**. 비 menu-complex 47문항은 B4 라우팅 범위 밖 → 불변.

## 무엇을 고쳤나

`_route_rule_select` 의 menu-complex 특례(룰 none 이어도 soar 직행)를 제거. 이제 none 이면 유형과 무관하게 `rule_fallback` 으로 간다. menu-complex 는 `menu_extract` 와 병렬 경로이므로, 폴백이 **비메뉴 조건(operator/tiebreak/negative)** 을 채워 soar 합류 시점에 메뉴 후보와 함께 쓰인다. (미사용 `MENU_TYPES` 상수 제거)

### 병렬 합류 테스트
`question_valid → [menu_extract ∥ rule_select]`, 이제 none 이면 `rule_select → rule_fallback → soar`. 재실행에서 q052·q104가 **정상적으로 폴백을 거쳐 soar(menu_mode)에서 추천 생성** → 합류 교착/오류 없음 확인.

## 회귀 결과 (menu-complex 3문항)

| qid | 룰 M0 → B4 | 최종 매장 M0 → B4 | fallback M0→B4 | 변화 |
|---|---|---|---|---|
| q052 | none → **맛집** | **None(추천 실패) → 홍카페 HONG CAFE** | F → **T** | ✅ 추천 실패→성공 |
| q104 | none → **맛집** | 커피니쉬(COFFEE NISHE) → 미남커피 | F → **T** | 변경 (B3 impasse 효과 중첩) |
| q139 | 공부 (불변) | 카페라움(CAFE RAUM) (불변) | F → F | 불변 (룰 있어 B4 무관) |

**매장 변경 2/3.**

## 해석

- **핵심 q052**: menu-complex인데 룰이 none이라 비메뉴 조건 없이 메뉴만으로 돌다 추천 실패(None)였음 → B4는 폴백이 '맛집' 룰을 찾아 비메뉴 조건을 채워 **홍카페 추천 성공**.
- **q139**(룰이 none이 아니었음)는 라우팅이 안 바뀌어 **완전 동일** → B4 효과가 none-menu-complex에 국한됨을 확인(격리).
- q104는 none이면서 impasse도 발동했던 문항이라 B3+B4 효과가 겹침.
- 바뀐 매장의 적합성은 M4(예외경로 Hit@1)에서 qrels로 평가.

## 한계 메모

폴백이 `NONE_FIT`을 내면 `rule_create→(실패시)END`로 빠져 menu 경로가 소실될 수 있다(병렬 분기 중 한 쪽만 END). 이번 3문항에선 발생 안 함. 전체 파이프라인 거동은 **M4**에서 50문항으로 확인한다.
