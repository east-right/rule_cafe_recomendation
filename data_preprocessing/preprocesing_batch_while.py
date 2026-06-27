import os
import sys
import json
import re
import time
import pandas as pd
import emoji
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from prompt import (
    NER_PASS1_SYSTEM_PROMPT, PASS1_FEW_SHOT, PASS1_RESPONSE_FORMAT,
    NER_PASS2_SYSTEM_PROMPT, PASS2_FEW_SHOT, PASS2_RESPONSE_FORMAT,
)

load_dotenv()

# Windows CP949 터미널 한글/특수문자 깨짐 방지
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL             = "gpt-4.1-mini"
REVIEWS_PER_CHUNK = 5      # 한 호출당 리뷰 수
MAX_REVIEWS_CAFE  = 150    # 카페당 최대 리뷰 수

# TEST_MODE=True : 10건 배치 1회 → 50개 리뷰 결과 확인
# TEST_MODE=False: 50건 배치 반복 → 전체 처리
TEST_MODE         = False
BATCH_SIZE        = 10 if TEST_MODE else 50

SAVE_DIR          = "../data"
INPUT_FILE        = "../data/shinline_cafe_reviews_test.csv"
PASS1_OUTPUT      = os.path.join(SAVE_DIR, "ner_pass1_result.json")
PASS2_OUTPUT      = os.path.join(SAVE_DIR, "ner_pass2_result.json")
FINAL_OUTPUT      = os.path.join(SAVE_DIR, "ner_result_v2.json")
BATCH_INPUT_PASS1 = os.path.join(SAVE_DIR, "batch_input_pass1.jsonl")
BATCH_INPUT_PASS2 = os.path.join(SAVE_DIR, "batch_input_pass2.jsonl")
BATCH_ID_PASS1    = os.path.join(SAVE_DIR, "batch_id_pass1.txt")
BATCH_ID_PASS2    = os.path.join(SAVE_DIR, "batch_id_pass2.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 전처리 ────────────────────────────────────────────────────────────────────
emoji_pattern = re.compile(
    '[' '\U00010000-\U0010ffff' '☀-➿' ']',
    flags=re.UNICODE
)

def review_cleaner(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, replace=" ")
    text = emoji_pattern.sub(" ", text)
    text = re.sub(r'([^\w\s^])\1+', r'\1', text)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    before = len(df)
    df['리뷰내용'] = df['리뷰내용'].apply(review_cleaner)
    df = df[df['리뷰내용'].str.strip() != ""].reset_index(drop=True)
    print(f"데이터 로드: {before}건 → 전처리 후 {len(df)}건")
    return df

# ── 카페 단위 청크 생성 ───────────────────────────────────────────────────────
def create_cafe_chunks(df: pd.DataFrame) -> list[dict]:
    chunks = []
    for place_id, group in df.groupby('place_id'):
        cafe_name = group.iloc[0]['사업장명']
        reviews   = group['리뷰내용'].tolist()[:MAX_REVIEWS_CAFE]
        for chunk_idx, start in enumerate(range(0, len(reviews), REVIEWS_PER_CHUNK)):
            chunk_reviews = reviews[start : start + REVIEWS_PER_CHUNK]
            chunks.append({
                'custom_id': f"p1_{place_id}_{chunk_idx}",
                'place_id':  str(place_id),
                'cafe_name': cafe_name,
                'chunk_idx': chunk_idx,
                'reviews':   chunk_reviews,
            })
    print(f"총 청크: {len(chunks)}개 ({df['place_id'].nunique()}개 카페)")
    return chunks

# ── PASS 1 배치 입력 파일 생성 ────────────────────────────────────────────────
def create_pass1_batch_input(chunks: list[dict], done_ids: set) -> int:
    pending = [c for c in chunks if c['custom_id'] not in done_ids]
    print(f"Pass1 대기: {len(pending)}건 / 완료: {len(done_ids)}건")
    if not pending:
        return 0

    batch = pending[:BATCH_SIZE]
    with open(BATCH_INPUT_PASS1, 'w', encoding='utf-8') as f:
        for chunk in batch:
            review_text = '\n'.join(
                f"[리뷰 {j+1}] {r}" for j, r in enumerate(chunk['reviews'])
            )
            request = {
                'custom_id': chunk['custom_id'],
                'method':    'POST',
                'url':       '/v1/chat/completions',
                'body': {
                    'model':           MODEL,
                    'temperature':     0,
                    'max_tokens':      800,
                    'response_format': PASS1_RESPONSE_FORMAT,
                    'messages': [
                        {'role': 'system', 'content': NER_PASS1_SYSTEM_PROMPT},
                        *PASS1_FEW_SHOT,
                        {'role': 'user', 'content': review_text},
                    ],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    print(f"Pass1 배치 파일 생성: {len(batch)}건 → {BATCH_INPUT_PASS1}")
    return len(batch)

# ── PASS 2 배치 입력 파일 생성 ────────────────────────────────────────────────
def create_pass2_batch_input(pass1_results: dict, done_ids: set) -> int:
    pending = []
    for p1_cid, data in pass1_results.items():
        p2_cid  = p1_cid.replace('p1_', 'p2_')
        if p2_cid in done_ids:
            continue

        reviews = data.get('reviews', [])
        # 모든 카테고리에서 EVALUATIVE만 수집 (카테고리 정보 포함)
        evaluative = []
        for cat in ('MENU', 'FACILITY', 'ATMOSPHERE', 'TARGET'):
            for e in data.get('entities', {}).get(cat, []):
                if e.get('type') == 'EVALUATIVE':
                    evaluative.append({**e, 'category': cat})

        if not evaluative:
            continue

        pending.append({
            'custom_id':           p2_cid,
            'reviews':             reviews,
            'evaluative_entities': evaluative,
        })

    print(f"Pass2 대기: {len(pending)}건 / 완료: {len(done_ids)}건")
    if not pending:
        return 0

    batch = pending[:BATCH_SIZE]
    with open(BATCH_INPUT_PASS2, 'w', encoding='utf-8') as f:
        for item in batch:
            reviews    = item['reviews']
            evaluative = item['evaluative_entities']

            # 각 entity에 review_idx 매핑해서 원문 붙임
            lines = []
            for j, e in enumerate(evaluative):
                ridx    = e.get('review_idx', 1)
                src_rev = reviews[ridx - 1] if 0 < ridx <= len(reviews) else ""
                lines.append(
                    f"{j+1}. entity: \"{e['entity']}\", "
                    f"category: {e['category']}, sentiment: {e['sentiment']}\n"
                    f"   원문: \"{src_rev}\""
                )

            user_content = "EVALUATIVE 엔티티:\n" + '\n'.join(lines)
            request = {
                'custom_id': item['custom_id'],
                'method':    'POST',
                'url':       '/v1/chat/completions',
                'body': {
                    'model':           MODEL,
                    'temperature':     0,
                    'max_tokens':      1200,
                    'response_format': PASS2_RESPONSE_FORMAT,
                    'messages': [
                        {'role': 'system', 'content': NER_PASS2_SYSTEM_PROMPT},
                        *PASS2_FEW_SHOT,
                        {'role': 'user', 'content': user_content},
                    ],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    print(f"Pass2 배치 파일 생성: {len(batch)}건 → {BATCH_INPUT_PASS2}")
    return len(batch)

# ── 배치 제출 (quota 초과 재시도 포함) ──────────────────────────────────────
def submit_batch_with_retry(input_file: str, id_file: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            print(f"배치 업로드 중... (시도 {attempt+1}/{max_retries})")
            with open(input_file, 'rb') as f:
                uploaded = client.files.create(file=f, purpose='batch')

            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint='/v1/chat/completions',
                completion_window='24h',
            )
            with open(id_file, 'w', encoding='utf-8') as f:
                f.write(batch.id)

            print(f"제출 완료 | Batch ID: {batch.id}")
            return batch.id

        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ('quota', 'limit', '429', 'rate', 'exceeded', 'too many')):
                wait = 60 * (2 ** attempt)
                print(f"[!] 배치 한도 초과 - {wait}초 후 재시도 (사유: {e})")
                time.sleep(wait)
            else:
                print(f"[!] 배치 제출 실패 (사유: {e})")
                raise

    raise RuntimeError(f"배치 제출 {max_retries}회 모두 실패")

# ── 상태 확인 / 완료 대기 ─────────────────────────────────────────────────────
def check_status(id_file: str) -> str:
    with open(id_file, encoding='utf-8') as f:
        batch_id = f.read().strip()

    batch  = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"  상태: {batch.status} | {counts.completed}/{counts.total} 완료 | {counts.failed} 실패")

    if batch.status == 'failed' and batch.errors:
        for err in batch.errors.data:
            print(f"  [에러] {err.code}: {err.message}")

    return batch.status

def wait_for_completion(id_file: str, poll_interval: int = 60) -> bool:
    with open(id_file, encoding='utf-8') as f:
        batch_id = f.read().strip()

    start = time.time()
    pbar  = None

    # 배치 제출 직후 404 race condition 방지 — 최초 조회 전 10초 대기
    time.sleep(10)

    while True:
        try:
            batch  = client.batches.retrieve(batch_id)
        except Exception as e:
            if '404' in str(e) or 'not found' in str(e).lower():
                print(f"  [재시도] 배치 아직 등록 안됨, 30초 후 재조회...")
                time.sleep(30)
                continue
            raise
        counts = batch.request_counts

        if pbar is None and counts.total > 0:
            pbar = tqdm(
                total=counts.total,
                unit='req',
                ncols=80,
                bar_format='{desc}: {n_fmt}/{total_fmt} {bar} {percentage:3.0f}% | {postfix}',
                desc='배치 처리',
                file=sys.stdout,
            )

        if pbar is not None:
            elapsed = int(time.time() - start)
            pbar.n = counts.completed
            pbar.set_postfix_str(
                f"{elapsed // 60}분 {elapsed % 60}초 경과 | 실패 {counts.failed} | {batch.status}"
            )
            pbar.refresh()

        if batch.status == 'completed':
            if pbar:
                pbar.n = pbar.total
                pbar.refresh()
                pbar.close()
            print("완료!")
            return True

        if batch.status in ('failed', 'expired', 'cancelled'):
            if pbar:
                pbar.close()
            print(f"[!] 비정상 종료: {batch.status}")
            if batch.errors:
                for err in batch.errors.data:
                    print(f"  [에러] {err.code}: {err.message}")
            return False

        time.sleep(poll_interval)

# ── PASS 1 결과 수집 ─────────────────────────────────────────────────────────
def collect_pass1_results(chunks: list[dict], existing: dict) -> dict:
    with open(BATCH_ID_PASS1) as f:
        batch_id = f.read().strip()

    output_file_id = client.batches.retrieve(batch_id).output_file_id
    raw_text = client.files.content(output_file_id).text
    lines    = [json.loads(l) for l in raw_text.strip().split('\n') if l.strip()]

    chunk_map = {c['custom_id']: c for c in chunks}
    results   = dict(existing)
    ok, fail  = 0, 0

    for line in lines:
        cid = line['custom_id']
        if line.get('error'):
            print(f"  [{cid}] API 에러: {line['error']}")
            fail += 1
            continue
        try:
            content = line['response']['body']['choices'][0]['message']['content']
            parsed  = json.loads(content)
            chunk   = chunk_map.get(cid, {})
            results[cid] = {
                'place_id':  chunk.get('place_id', ''),
                'chunk_idx': chunk.get('chunk_idx', 0),
                'reviews':   chunk.get('reviews', []),
                'entities':  parsed,   # {MENU:[...], FACILITY:[...], ATMOSPHERE:[...], TARGET:[...]}
            }
            ok += 1
        except Exception as e:
            print(f"  [{cid}] 파싱 에러: {e}")
            fail += 1

    with open(PASS1_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Pass1 수집: 성공 {ok} / 실패 {fail} → {PASS1_OUTPUT}")
    return results

# ── PASS 2 결과 수집 ─────────────────────────────────────────────────────────
def collect_pass2_results(existing: dict) -> dict:
    with open(BATCH_ID_PASS2) as f:
        batch_id = f.read().strip()

    output_file_id = client.batches.retrieve(batch_id).output_file_id
    raw_text = client.files.content(output_file_id).text
    lines    = [json.loads(l) for l in raw_text.strip().split('\n') if l.strip()]

    results  = dict(existing)
    ok, fail = 0, 0

    for line in lines:
        cid = line['custom_id']
        if line.get('error'):
            print(f"  [{cid}] API 에러: {line['error']}")
            fail += 1
            continue
        try:
            content = line['response']['body']['choices'][0]['message']['content']
            parsed  = json.loads(content)
            # 확신인 것만 저장, 애매는 None 처리
            results[cid] = {
                str(d['idx']): d.get('descriptor') if d.get('confidence') == '확신' else None
                for d in parsed.get('descriptors', [])
            }
            ok += 1
        except Exception as e:
            print(f"  [{cid}] 파싱 에러: {e}")
            fail += 1

    with open(PASS2_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Pass2 수집: 성공 {ok} / 실패 {fail} → {PASS2_OUTPUT}")
    return results

# ── 최종 병합 ────────────────────────────────────────────────────────────────
def merge_and_save(pass1_results: dict, pass2_results: dict, total_cafes: int) -> None:
    final = {}

    for p1_cid, data in pass1_results.items():
        place_id = data.get('place_id', '')
        p2_cid   = p1_cid.replace('p1_', 'p2_')
        reviews  = data.get('reviews', [])
        desc_map = pass2_results.get(p2_cid, {})  # {"1": descriptor, "2": descriptor, ...}

        if place_id not in final:
            final[place_id] = []

        eval_idx = 0  # EVALUATIVE entity의 1-based 순서 (Pass2 idx 매핑용)
        for cat in ('MENU', 'FACILITY', 'ATMOSPHERE', 'TARGET'):
            for e in data.get('entities', {}).get(cat, []):
                ridx = e.get('review_idx')
                entry = {
                    'entity':        e['entity'],
                    'category':      cat,
                    'type':          e['type'],
                    'sentiment':     e['sentiment'],
                    'review_idx':    ridx,
                    'source_review': reviews[ridx - 1] if ridx and 0 < ridx <= len(reviews) else None,
                    'descriptor':    None,
                }
                if e['type'] == 'EVALUATIVE':
                    eval_idx += 1
                    entry['descriptor'] = desc_map.get(str(eval_idx))

                final[place_id].append(entry)

    with open(FINAL_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 누락 검증
    processed = len(final)
    print(f"\n최종 병합 완료 → {FINAL_OUTPUT}")
    print(f"처리 카페: {processed} / 전체: {total_cafes}")
    if processed < total_cafes:
        print(f"[!] 누락 {total_cafes - processed}개 카페 - 재실행 시 자동 재처리됩니다")

# ── 메인 파이프라인 ───────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame) -> None:
    total_cafes = df['place_id'].nunique()
    chunks      = create_cafe_chunks(df)

    # ── PASS 1 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PASS 1 - Entity 추출 (category / type / sentiment)")
    print("=" * 55)

    pass1_results: dict = {}
    if os.path.exists(PASS1_OUTPUT):
        with open(PASS1_OUTPUT, encoding='utf-8') as f:
            pass1_results = json.load(f)

    while True:
        count = create_pass1_batch_input(chunks, set(pass1_results.keys()))
        if count == 0:
            print("Pass1 전체 완료.")
            break

        submit_batch_with_retry(BATCH_INPUT_PASS1, BATCH_ID_PASS1)

        if not wait_for_completion(BATCH_ID_PASS1):
            print("Pass1 배치 실패. 파이프라인 중단.")
            return

        prev_done = len(pass1_results)
        pass1_results = collect_pass1_results(chunks, pass1_results)

        # 무한루프 방지: 이번 라운드에 새로 처리된 청크가 0개면
        # (반복적으로 API/파싱 에러가 나는 청크만 남은 것) 중단한다.
        if len(pass1_results) == prev_done:
            print(f"[!] Pass1 진전 없음 — 반복 실패 청크 {count}건을 건너뛰고 중단합니다.")
            break

        if TEST_MODE:
            print(f"\n[TEST_MODE] {BATCH_SIZE}건 완료. 결과 확인 후 TEST_MODE=False로 전환하세요.")
            break

        time.sleep(5)

    # ── PASS 2 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PASS 2 - Descriptor 추출 (EVALUATIVE entity만)")
    print("=" * 55)

    pass2_results: dict = {}
    if os.path.exists(PASS2_OUTPUT):
        with open(PASS2_OUTPUT, encoding='utf-8') as f:
            pass2_results = json.load(f)

    while True:
        count = create_pass2_batch_input(pass1_results, set(pass2_results.keys()))
        if count == 0:
            print("Pass2 전체 완료.")
            break

        submit_batch_with_retry(BATCH_INPUT_PASS2, BATCH_ID_PASS2)

        if not wait_for_completion(BATCH_ID_PASS2):
            print("Pass2 배치 실패. 파이프라인 중단.")
            return

        prev_done = len(pass2_results)
        pass2_results = collect_pass2_results(pass2_results)

        # 무한루프 방지: 이번 라운드에 새로 처리된 청크가 0개면
        # (반복적으로 API/파싱 에러가 나는 청크만 남은 것) 중단한다.
        if len(pass2_results) == prev_done:
            print(f"[!] Pass2 진전 없음 — 반복 실패 청크 {count}건을 건너뛰고 중단합니다.")
            break

        if TEST_MODE:
            print(f"\n[TEST_MODE] {BATCH_SIZE}건 완료. 결과 확인 후 TEST_MODE=False로 전환하세요.")
            break

        time.sleep(5)

    # ── 최종 병합 ────────────────────────────────────────────────────────────
    merge_and_save(pass1_results, pass2_results, total_cafes)


if __name__ == '__main__':
    df = load_data(INPUT_FILE)
    run_pipeline(df)
