# ---------- 유틸 ----------
import os, torch, traceback
from typing import Dict, Any, Optional
from threading import RLock

from AI.model_loader import get_image_pipe

# 전역(싱글톤)
IMAGE_DEVICE: Optional[str] = None
TOK = None
LLM = None
PIPE = None
IMAGE_LOCK = RLock()
SESSIONS: Dict[str, Dict[str, Any]] = {}

def strip_code_block(text: str) -> str:
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    count = 0
    for i in range(start, len(text)):
        if text[i] == "{": count += 1
        elif text[i] == "}":
            count -= 1
            if count == 0:
                return text[start : i + 1]
    return text

def clamp_prompt(p: str, max_words=70):
    return " ".join((p or "").split()[:max_words])

def sanitize_user_text(t: str, max_len=120):
    t = (t or "").replace("\n", " ").strip()
    for bad in ["```", "{", "}", "<<", ">>"]:
        t = t.replace(bad, "")
    return t[:max_len]

def ensure_choices(sd, branch_prefix):
    must = [f"{branch_prefix}-1", f"{branch_prefix}-2",
            f"{branch_prefix}-3", f"{branch_prefix}-4"]
    return (
        isinstance(sd, dict)
        and isinstance(sd.get("choices"), dict)
        and all(k in sd["choices"] for k in must)
    )
def _ensure_pipe_device(pipe, target: str) -> None:
    try:
        exec_dev = getattr(pipe, "_execution_device", None)
        if exec_dev is not None and str(exec_dev) == str(target):
            return
    except Exception:
        pass

    try:
        if getattr(pipe, "device_map", None) is not None:
            try:
                pipe._execution_device = torch.device(target)
                print(f"[Device] device_map detected → set _execution_device={target} (skip .to)")
            except Exception:
                pass
            return
    except Exception:
        pass

    try:
        pipe.to(target)
        try:
            pipe._execution_device = torch.device(target)
        except Exception:
            pass
        print(f"[Device] pipe.to({target})")
    except Exception as e:
        print(f"[Device] skip .to due to error: {e}")


# def ensure_image_pipe() -> Any:
#     #이미지 파이프라인이 없으면 지금 로드하고, 실행 디바이스도 보정.
#     global PIPE
#     if PIPE is None:
#         with IMAGE_LOCK:
#             if PIPE is None:  # 더블체크
#                 print("[Image] Lazy loading SDXL pipeline...")
#                 PIPE = get_image_pipe(device=IMAGE_DEVICE or "cpu")
#                 try:
#                     _ensure_pipe_device(PIPE, IMAGE_DEVICE or "cpu")
#                 except Exception:
#                     pass
#     return PIPE
def ensure_image_pipe(exec_device: Optional[str] = None):
    global PIPE
    target = exec_device or "cpu"

    # 락을 기다리되 5초 넘으면 스택 덤프 후 계속 시도
    if not IMAGE_LOCK.acquire(timeout=5):
        try:
            from test.test_story_engine import dump_all_threads
            dump_all_threads("LOCK-WAIT ensure_image_pipe")
        except Exception:
            print("[WARN] dump_all_threads not available")
        IMAGE_LOCK.acquire()

    try:
        if PIPE is None:
            print(f"[Image] Lazy loading SDXL pipeline on {target} ...")
            PIPE = get_image_pipe(device=target)    # ← 여기서 오래 걸릴 수도
        _ensure_pipe_device(PIPE, target)
        return PIPE
    finally:
        IMAGE_LOCK.release()

# def generate_and_save_image(pipe, prompt: str, filename: str, seed: Optional[int]=None,
#                             size: str="fast", offload_after=False):
#     # 프리셋
#     if size == "fast":
#         H, W, steps, guidance = 832, 832, 20, 4.0
#     elif size == "balanced":
#         H, W, steps, guidance = 1024, 1024, 28, 5.0
#     else:  # "quality"
#         H, W, steps, guidance = 1152, 1152, 32, 5.5
        
#     exec_device = IMAGE_DEVICE or "cpu"
#     _ensure_pipe_device(pipe, exec_device)

#     gen = None
#     if seed is not None:
#         gen = torch.Generator(device=exec_device).manual_seed(int(seed))

#     d = os.path.dirname(filename)

#     if d: os.makedirs(d, exist_ok=True)
#     with torch.inference_mode():
#         image = pipe(
#             clamp_prompt(prompt),
#             height=H, width=W,
#             num_inference_steps=steps,
#             guidance_scale=guidance,
#             generator=gen
#         ).images[0]
#     image.save(filename)

#     if offload_after:
#         try:
#             pipe.to("cpu"); torch.cuda.empty_cache()
#         except Exception: pass
#     return pipe
def generate_and_save_image(
    pipe,
    prompt: str,
    filename: str,
    seed: Optional[int] = None,
    size: str = "fast",
    offload_after: bool = False,
    exec_device: Optional[str] = None,
):
    # 프리셋
    if size == "fast":
        H, W, steps, guidance = 832, 832, 20, 4.0
    elif size == "balanced":
        H, W, steps, guidance = 1024, 1024, 28, 5.0
    else:  # "quality"
        H, W, steps, guidance = 1152, 1152, 32, 5.5

    target = exec_device or "cpu"
    _ensure_pipe_device(pipe, target)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=target).manual_seed(int(seed))

    d = os.path.dirname(filename)
    if d:
        os.makedirs(d, exist_ok=True)

    with torch.inference_mode():
        image = pipe(
            clamp_prompt(prompt),
            height=H,
            width=W,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        ).images[0]
    image.save(filename)

    if offload_after:
        try:
            pipe.to("cpu"); torch.cuda.empty_cache()
        except Exception:
            pass
    return pipe

# def make_img_async(pipe, prompt, path):
#     try:
#         with IMAGE_LOCK:
#             real_pipe = ensure_image_pipe()  # ← lazy 로드 보장
#             offload = (torch.cuda.device_count() == 1)  # 단일 GPU면 생성 후 내리자
#             generate_and_save_image(real_pipe, prompt, path, offload_after=offload)
#         print(f"[이미지 완료: {path}]")
#     except Exception as e:
#         print(f"[이미지 생성 실패: {e}]")

def make_img_sync(prompt: str, path: str, exec_device: Optional[str] = None) -> bool:
    try:
        print("[DBG] make_img_sync: entering")
        # 여기선 락을 안 잡고, ensure_image_pipe에 맡기는 게 안전
        real_pipe = ensure_image_pipe(exec_device=exec_device)
        print("[DBG] make_img_sync: got pipe")

        offload = (torch.cuda.device_count() == 1)
        print(f"[Image] generate (sync, dev={exec_device or 'cpu'}) → {path}")

        generate_and_save_image(
            real_pipe, prompt, path,
            offload_after=offload, exec_device=exec_device,
        )
        ok = os.path.exists(path) and os.path.getsize(path) > 0
        if not ok:
            print(f"[Image] file not found after save: {path}")
        print("[DBG] make_img_sync: done")
        return ok
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[이미지 생성 실패(sync): {e}]\n{tb}")
        try:
            err_path = os.path.splitext(path)[0] + ".err.txt"
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"PROMPT:\n{prompt}\n\nERROR:\n{tb}")
            print(f"[이미지 실패 로그 저장: {err_path}]")
        except Exception:
            pass
        return False

def _normalize_device(dev: Optional[str]) -> str:
    if dev is None:
        return "cpu"
    dev = str(dev)
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[Device] CUDA not available → fallback to CPU (requested={dev})")
            return "cpu"
        if dev == "cuda":
            dev = "cuda:0"
        try:
            idx = int(dev.split(":")[1])
        except Exception:
            idx = 0
            dev = f"cuda:{idx}"
        if idx >= torch.cuda.device_count():
            print(f"[Device] Invalid CUDA index {idx} → fallback to cuda:0")
            dev = "cuda:0"
    return dev

def _get_tensor_device(model: torch.nn.Module) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"
    
def normalize_device(dev: Optional[str]) -> str:
    return _normalize_device(dev)

def get_tensor_device(model: torch.nn.Module) -> str:
    return _get_tensor_device(model)
    
# ---------- 생성 로직 ----------
def build_recent_context(st, max_chars=800):
    # 최근 2~3개 story를 뒤에서부터 모아 truncation
    chunks = []
    for ch in reversed(st["chapters"][-3:]):
        s = ((ch.get("ai_story") or {}).get("story") or "").strip()
        if s:
            chunks.append(s)
    ctx = "\n".join(reversed(chunks))[:max_chars]
    return ctx