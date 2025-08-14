# app.py
import os, logging, warnings, torch
from typing import Any, Dict, Optional

# ===== 환경변수/로그 억제 (import 전에) =====
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

warnings.filterwarnings("ignore", message=".*dtype=torch.float16.*cpu.*")
for name in ("diffusers", "transformers", "peft"):
    logging.getLogger(name).setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException, asynccontextmanager, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

import story_engine as SM  # 너의 모듈

# ===== FastAPI =====
app = FastAPI(title="Story Engine API", version="1.0.0")

# CORS (원하면 환경변수로 도메인 제한 가능: FRONTEND_ORIGINS="https://a.com,https://b.com")
origins_env = os.getenv("FRONTEND_ORIGINS", "*")
allow_origins = ["*"] if origins_env.strip() == "*" else [o.strip() for o in origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static 폴더 mount (story_engine.STATIC_ROOT 사용)
app.mount("/static", StaticFiles(directory=SM.STATIC_ROOT), name="static")

# ---------- Schemas ----------
class InitReq(BaseModel):
    protagonist_name: str
    protagonist_appearance: str
    protagonist_characteristic: str
    start_location: str
    era: str
    genre: str
    ENDING_POINT: int = Field(ge=1)

class InitRes(BaseModel):
    session_id: str
    page_index: int
    page: Dict[str, Any]
    image_url: Optional[str] = None

class ChooseReq(BaseModel):
    session_id: str
    choice: int = Field(ge=1, le=4)
    custom_text: Optional[str] = None

class PageRes(BaseModel):
    finished: bool
    page_index: int
    page: Dict[str, Any]
    image_url: Optional[str] = None

# ---------- Startup ----------
def _pick_gpus():
    n = torch.cuda.device_count()
    if n == 0: return None, None
    if n == 1: return 0, 0
    return 0, 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup 단계
    gt, gi = _pick_gpus()
    SM.init_models(gpu_text=gt, gpu_image=gi)
    print(f"[FastAPI] Startup complete. GPUs(text,image)=({gt},{gi})")
    
    yield  # 여기까지가 startup, 아래부터 shutdown

    # shutdown 단계
    print("[FastAPI] Server shutting down...")

app = FastAPI(lifespan=lifespan)

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpus": torch.cuda.device_count(),
    }

# JSON 바디 방식
@app.post("/init", response_model=InitRes)
def init_session(req: InitReq):
    try:
        out = SM.create_session(req.model_dump())
        return InitRes(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/choose", response_model=PageRes)
def choose(req: ChooseReq):
    try:
        out = SM.choose(req.session_id, req.choice, req.custom_text)
        return PageRes(**out)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/{session_id}")
def state(session_id: str):
    try:
        return SM.get_state(session_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))

# ---------- (옵션) Form 기반 엔드포인트 ----------
# 프론트에서 <form-data>로 보낼 때 사용 (예: fetch로 FormData 전송)
@app.post("/init_form", response_model=InitRes)
def init_session_form(
    protagonist_name: str = Form(...),
    protagonist_appearance: str = Form(...),
    protagonist_characteristic: str = Form(...),
    start_location: str = Form(...),
    era: str = Form(...),
    genre: str = Form(...),
    ENDING_POINT: int = Form(...),
):
    try:
        payload = {
            "protagonist_name": protagonist_name,
            "protagonist_appearance": protagonist_appearance,
            "protagonist_characteristic": protagonist_characteristic,
            "start_location": start_location,
            "era": era,
            "genre": genre,
            "ENDING_POINT": ENDING_POINT,
        }
        out = SM.create_session(payload)
        return InitRes(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/choose_form", response_model=PageRes)
def choose_form(
    session_id: str = Form(...),
    choice: int = Form(...),
    custom_text: Optional[str] = Form(None),
):
    try:
        out = SM.choose(session_id, choice, custom_text)
        return PageRes(**out)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Uvicorn 실행부 (포트포워딩 친화) ----------
if __name__ == "__main__":
    # 환경변수로 호스트/포트 조절 가능 (예: HOST=0.0.0.0 PORT=8080)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # PROXY 환경(nginx, render, railway 등)에서 X-Forwarded-* 헤더 신뢰
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        reload=bool(os.getenv("RELOAD", "0") == "1"),
        # workers는 GPU 초기화/메모리 문제 있을 수 있어 기본 1 권장
    )
