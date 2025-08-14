# model_loader.py
import os
import torch
from functools import lru_cache
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

@lru_cache(maxsize=1)
def get_text_model(ax_dir: str = "cache/AX", device: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(ax_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(ax_dir, torch_dtype=DTYPE)  # device_map 제거
    model.to(device or ("cuda:0" if DEVICE == "cuda" else "cpu"))
    model.eval()
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
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=DTYPE, use_safetensors=True
    )
    pipe = pipe.to(device or ("cuda:0" if DEVICE == "cuda" else "cpu"))  # ← 이 줄만 남기기

    _safe_load_lora(pipe, "loras/StorybookRedmondV2-KidsBook-KidsRedmAF.safetensors", "illu")
    _safe_load_lora(pipe, "loras/J_oil_pastels_XL.safetensors", "fantasy")
    try:
        pipe.set_adapters(["illu", "fantasy"], adapter_weights=[0.1, 0.8])
    except Exception:
        pass

    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
    except Exception:
        pass

    try:
        if DEVICE == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass

    pipe.set_progress_bar_config(disable=True)
    return pipe

# (선택) 시드 있는 제너레이터 헬퍼
def make_generator(seed: int | None = None) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g
