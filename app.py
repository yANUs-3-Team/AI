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



from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

import AI.story_engine as SM 

# ----------  ngrok 연동 ----------
import json, hmac, hashlib, requests
from threading import Thread

# ✅ 환경변수 이름으로 읽어야 함 (URL을 직접 넣으면 안 됨)
BE_INGEST_URL    = os.getenv("BE_INGEST_URL", "")         # 예: https://<be>.ngrok-free.app/be/ingest
BE_SHARED_SECRET = os.getenv("BE_SHARED_SECRET", "")       # 선택: AI→BE 서명용
AI_EXTERNAL_BASE = os.getenv("AI_EXTERNAL_BASE", "")       # 예: https://<ai>.ngrok-free.app (이미지 절대경로용)

def _abs_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if AI_EXTERNAL_BASE and u.startswith("/"):
        return AI_EXTERNAL_BASE.rstrip("/") + u
    return u

def _post_ingest(payload: Dict[str, Any]):
    if not BE_INGEST_URL:
        return
    headers = {"Content-Type": "application/json"}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if BE_SHARED_SECRET:
        sig = hmac.new(BE_SHARED_SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers["X-AI-Signature"] = sig
    try:
        requests.post(BE_INGEST_URL, data=body, headers=headers, timeout=5)
    except Exception as e:
        print(f"[BE ingest fail] {e}")

def push_to_be_async(session_id: str, finished: bool, page_index: int, page_dict: Dict[str, Any], image_url: Optional[str]):
    payload = {
        "session_id": session_id,
        "finished": bool(finished),
        "page_index": int(page_index),
        "page": page_dict or {},
        "image_url": _abs_url(image_url),
        "status": "ready",
    }
    Thread(target=_post_ingest, args=(payload,), daemon=True).start() 

# ---------- 조회용 캐시 생성 ----------

# FE-friendly 최신 페이지 응답 캐시
LATEST: Dict[str, Dict[str, Any]] = {}
# BE용 원본 최신 페이지 캐시
RAW_LAST: Dict[str, Dict[str, Any]] = {}
# BE가 특정 페이지를 GET할 수 있도록 인덱스별 원본 저장
RAW_PAGES: Dict[str, Dict[int, Dict[str, Any]]] = {}

# ---------- Startup ----------
def _pick_gpus():
    n = torch.cuda.device_count()
    if n == 0: return None, None
    if n == 1: return 0, 0
    return 0, 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    gt, gi = _pick_gpus()
    SM.init_models(gpu_text=gt, gpu_image=gi)
    print(f"[FastAPI] Startup complete. GPUs(text,image)=({gt},{gi})")    
    yield  # 여기까지 startup

    # shutdown 단계
    print("[FastAPI] Server shutting down...")

# ===== FastAPI =====
app = FastAPI(
    title="Story Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins_env = os.getenv("FRONTEND_ORIGINS", "*")
allow_origins = ["*"] if origins_env.strip() == "*" else [o.strip() for o in origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static
app.mount("/static", StaticFiles(directory=SM.STATIC_ROOT), name="static")

# ---------- Schemas ----------
class InitReq(BaseModel):
    name: str
    personality: str
    characteristics: str
    location: str
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

# ---------- Endpoints ----------
@app.get("/health")
def health():
    models_ready = bool(getattr(SM, "_INIT_DONE", False))
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpus": torch.cuda.device_count(),
        "models_ready": models_ready,
    }

# JSON 바디 방식

def _page_to_api(page_dict: Dict[str, Any], page_index: int, img_url: Optional[str]):
    raw = (page_dict or {}).get("choices") or {}
    def row(k, v):
        try:
            cid = int(str(k).split("-")[-1])   # "pageN-1" → 1
        except:
            cid = None
        return {"id": cid, "label": v, "requires_text": (cid == 4)}
    choices = [row(k, v) for k, v in sorted(raw.items())]
    img_url = _abs_url(img_url)  # ← FE가 AI ngrok에서 바로 보려면 주석 해제

    return {
        "page_index": page_index,
        "story": page_dict.get("story"),
        "image": {"url": img_url} if img_url else None,
        "choices": choices
    }


class SessionCreateIn(BaseModel):
    name: str
    personality: str
    characteristics: str
    location: str
    era: str
    genre: str
    ending_point: int = Field(ge=1)

@app.post("/sessions")
def create_session_api(req: SessionCreateIn):
    try:
        out = SM.create_session({
            "name": req.name,
            "personality": req.personality,
            "characteristics": req.characteristics,
            "location": req.location,
            "era": req.era,
            "genre": req.genre,
            "ENDING_POINT": req.ending_point,
        })
        page_api = _page_to_api(out["page"], out["page_index"], out.get("image_url"))
        resp = {"session_id": out["session_id"], "finished": False, "page": page_api}

        # FE-friendly 캐시
        LATEST[out["session_id"]] = resp

        # BE용 원본 캐시
        raw_payload = {
            "session_id": out["session_id"],
            "finished": False,
            "page_index": out["page_index"],
            "page": out["page"],
            "image_url": _abs_url(out.get("image_url")),
            "status": "ready",
        }
        RAW_LAST[out["session_id"]] = raw_payload
        RAW_PAGES.setdefault(out["session_id"], {})[out["page_index"]] = raw_payload

        # AI → BE ingest (비동기)
        push_to_be_async(out["session_id"], False, out["page_index"], out["page"], out.get("image_url"))

        return resp
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class ChooseIn(BaseModel):
    choice_id: int = Field(ge=1, le=4)
    text: Optional[str] = None

@app.post("/sessions/{session_id}/choose")
def choose_api(session_id: str, body: ChooseIn):
    try:
        if body.choice_id == 4 and not (body.text and body.text.strip()):
            raise HTTPException(status_code=400, detail="custom text is required for choice_id=4")

        out = SM.choose(session_id, body.choice_id, body.text)
        page_api = _page_to_api(out["page"], out["page_index"], out.get("image_url"))
        resp = {"finished": out["finished"], "page": page_api}

        # FE-friendly 캐시
        LATEST[session_id] = {"session_id": session_id, **resp}

        # BE용 원본 캐시
        raw_payload = {
            "session_id": session_id,
            "finished": out["finished"],
            "page_index": out["page_index"],
            "page": out["page"],
            "image_url": _abs_url(out.get("image_url")),
            "status": "ready",
        }
        RAW_LAST[session_id] = raw_payload
        RAW_PAGES.setdefault(session_id, {})[out["page_index"]] = raw_payload

        # AI → BE ingest (비동기)
        push_to_be_async(session_id, out["finished"], out["page_index"], out["page"], out.get("image_url"))

        return resp
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/sessions/{session_id}/current")
def get_current_page(session_id: str):
    cur = LATEST.get(session_id)
    if cur:
        return cur
    # 최소 상태라도 반환
    try:
        st = SM.get_state(session_id)
        return {"session_id": session_id, "finished": st["finished"], "page": None}
    except Exception:
        raise HTTPException(status_code=404, detail="session not found")

@app.get("/raw/sessions/{session_id}/current")    
def get_raw_current(session_id: str):
    cur = RAW_LAST.get(session_id)
    if not cur:
        raise HTTPException(status_code=404, detail="session not found")
    return cur

@app.get("/raw/sessions/{session_id}/pages/{page_index}")
def get_raw_page(session_id: str, page_index: int):
    pages = RAW_PAGES.get(session_id) or {}
    cur = pages.get(page_index)
    if not cur:
        raise HTTPException(status_code=404, detail="page not found")
    return cur

# ---------- (옵션) Form 기반 엔드포인트 ----------
# 프론트에서 <form-data>로 보낼 때 사용 
@app.post("/init_form", response_model=InitRes)
def init_session_form(
    name: str = Form(...),
    personality: str = Form(...),
    characteristics: str = Form(...),
    location: str = Form(...),
    era: str = Form(...),
    genre: str = Form(...),
    ENDING_POINT: int = Form(...),
):
    try:
        payload = {
            "name": name,
            "personality": personality,
            "characteristics": characteristics,
            "location": location,
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
    # 환경변수로 호스트/포트 조절 가능 
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
