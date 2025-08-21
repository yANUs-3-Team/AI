# app.py
# -------------------------------------------------
import os, logging, warnings, torch

# ===== 환경변수/로그 억제 (import 전에) =====
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

from typing import Any, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
import uvicorn
# 내부 스토리 엔진 의존성
import AI.story_engine as SM

warnings.filterwarnings("ignore", message=".*dtype=torch.float16.*cpu.*")
for name in ("diffusers", "transformers", "peft"):
    logging.getLogger(name).setLevel(logging.ERROR)

# ---------- GPU 선택 유틸 ----------
def _pick_gpus():
    n = torch.cuda.device_count()
    if n == 0:
        return None, None
    if n == 1:
        return 0, 0
    return 0, 1

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    gt, gi = _pick_gpus()
    SM.init_models(gpu_text=gt, gpu_image=gi)
    print(f"[FastAPI] Startup complete. GPUs(text,image)=({gt},{gi})")
    yield
    print("[FastAPI] Server shutting down…")

# ---------- App ----------
app = FastAPI(
    title="Story Engine API (minimal)",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------- Schemas ----------
class SessionCreateIn(BaseModel):
    name: str
    personality: str
    characteristics: str
    location: str
    era: str
    genre: str
    ending_point: int = Field(ge=1)

class SessionCreateOut(BaseModel):
    session_id: str
    page_index: int
    page: Dict[str, Any]
    image_url: Optional[str] = None

class ChooseIn(BaseModel):
    choice_id: str
    session_id: Optional[str] = None

class StepOut(BaseModel):
    finished: bool
    page_index: int
    page: Dict[str, Any]
    image_url: Optional[str] = None

class FlatPage(BaseModel):
    page_number: int
    story: str
    session_id: Optional[str] = None
    image: Optional[str] = None
    choices_1: Optional[str] = None
    choices_2: Optional[str] = None
    choices_3: Optional[str] = None
    choices_4: Optional[str] = None

# ---------- Helpers ----------

def _flatten_story_page(session_id: Optional[str], page_index: int, page: Dict[str, Any], image_url: Optional[str]) -> Dict[str, Any]:
    story = (page or {}).get("story", "") or ""
    choices = (page or {}).get("choices", {}) or {}

    def pick(i: int) -> Optional[str]:
        # page{N}-i / page{N}i / "i" 어떤 키로 와도 잡아주기
        for key in (f"page{page_index}-{i}", f"page{page_index}{i}", str(i)):
            if key in choices:
                return choices[key]
        return None

    return {
        "page_number": int(page_index),
        "story": story,
        "image": image_url,
        "session_id": session_id,
        "choices_1": pick(1),
        "choices_2": pick(2),
        "choices_3": pick(3),
        "choices_4": pick(4),
    }

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

@app.post("/sessions", response_model=FlatPage)
def create_session(req: SessionCreateIn):
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
        return _flatten_story_page(out.get("session_id"), out["page_index"], out["page"], out.get("image_url"), )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/choose", response_model=FlatPage)
def choose(session_id: str, body: ChooseIn):
    try:
        out = SM.choose(body.choice_id, body.session_id)
        return _flatten_story_page(session_id, out["page_index"], out["page"], out.get("image_url"))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/state")
def get_state(session_id: str):
    try:
        st = SM.get_state(session_id)
        # SM.get_state 결과를 그대로 반환(프론트 변환 없음)
        return st
    except Exception:
        raise HTTPException(status_code=404, detail="session not found")
    
app.mount("/static", StaticFiles(directory=SM.STATIC_ROOT), name="static")
# ---------- Entrypoint ----------
if __name__ == "__main__":
    print("[ENTRY] test_app __main__ reached")
    host = os.getenv("HOST", "0.0.0.0"); port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app, host=host, port=port, log_level="debug",
        proxy_headers=True, forwarded_allow_ips="*", reload=False, lifespan="on",
    )