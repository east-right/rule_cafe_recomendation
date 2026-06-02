import os
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN_WRITE"))  # 본인 토큰으로 교체

EMB_DIR = Path(__file__).resolve().parent
MODEL_DIR = EMB_DIR / "models" / "bge-m3-cafe"

api.create_repo(
    repo_id="east-right/bge-m3-cafe-finetuned", 
    repo_type="model", 
    exist_ok=True
)

api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id="east-right/bge-m3-cafe-finetuned",  # 본인 허깅페이스 아이디로 교체
    repo_type="model"
)

print("✅ 업로드 완료!")