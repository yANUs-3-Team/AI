# model_loader.py

import os,torch
from functools import lru_cache
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

@lru_cache(maxsize=1)
#-------------------- v3---------------------

def get_text_model(ax_dir: str = "cache/AX4.0", device: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(ax_dir, use_fast=True)

    target_device = device or ("cuda:0" if DEVICE == "cuda" else "cpu")
    use_fp16 = (str(target_device).startswith("cuda"))
    dtype = torch.float16 if use_fp16 else torch.float32

    # 1) config만 먼저 불러오고
    config = AutoConfig.from_pretrained(ax_dir)

    # 2) 비어 있는(meta) 모듈을 만든 뒤
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # 3) 체크포인트를 지정 디바이스로 안전하게 로드/디스패치
    #    - 단일 GPU면 {"": 0} 로 한 방에
    #    - VRAM 아슬아슬하면 offload_folder로 CPU 오프로딩 가능
    device_map = {"": 0} if str(target_device).startswith("cuda") else {"": "cpu"}
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ax_dir,      # 폴더 경로 그대로
        device_map=device_map,
        dtype=dtype,
        offload_folder=None,    # 필요 시 "offload" 등의 폴더로 바꾸면 CPU 오프로딩
    )

    model.eval()

    # 안정성 체크
    for n, p in model.named_parameters():
        if getattr(p, "device", None) and p.device.type == "meta":
            raise RuntimeError(f"Meta tensor remains: {n}")

    # pad/eos 정리
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    return tok, model

def _safe_load_lora(pipe: StableDiffusionXLPipeline, path: str, adapter_name: str) -> None:
    if os.path.exists(path):
        pipe.load_lora_weights(path, adapter_name=adapter_name)
    else:
        print(f"[model_loader] LoRA not found: {path} (skip)")

@lru_cache(maxsize=1)

def get_image_pipe(base_model: str = "./stable-diffusion-xl-base-1.0", device: Optional[str] = None):
    target = device or ("cuda:0" if DEVICE == "cuda" else "cpu")
    use_fp16 = str(target).startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32
    is_local = os.path.isdir(base_model)

    load_kwargs = dict(
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=is_local,
    )

    if use_fp16:
        load_kwargs["device_map"] = "balanced"
        load_kwargs["low_cpu_mem_usage"] = True
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(base_model, variant="fp16", **load_kwargs)
        except Exception:
            pipe = StableDiffusionXLPipeline.from_pretrained(base_model, **load_kwargs)
    else:
        load_kwargs["low_cpu_mem_usage"] = False
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model, **load_kwargs)

    try:
        pipe._execution_device = torch.device(target)
    except Exception:
        pass

    _safe_load_lora(pipe, "loras/StorybookRedmondV2-KidsBook-KidsRedmAF.safetensors", "illu")
    _safe_load_lora(pipe, "loras/J_oil_pastels_XL.safetensors", "fantasy")
    try: pipe.set_adapters(["illu", "fantasy"], adapter_weights=[0.5, 0.5])
    except: pass
    try: pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
    except: pass
    try:
        if DEVICE == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except: pass

    pipe.set_progress_bar_config(disable=True)
    return pipe
try: torch.cuda.empty_cache()
except: pass

# 시드 있는 제너레이터 헬퍼
def make_generator(seed: int | None = None) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g