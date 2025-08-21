# story_engine.py

import os, json, torch
from threading import RLock
from typing import Dict, Any, Optional
import time
from AI.utils import (
    strip_code_block,
    extract_json_object,
    sanitize_user_text,
    ensure_choices,
    build_recent_context,
    make_img_sync,   
    normalize_device,     
    get_tensor_device       
)

from AI.model_loader import get_text_model, get_image_pipe

# 전역(싱글톤)
TOK = None
LLM = None
PIPE = None
_INIT_DONE = False
_INIT_LOCK = RLock()
IMAGE_LOCK = RLock()
SESSIONS: Dict[str, Dict[str, Any]] = {}
_IMAGE_LAZY = True  # --------------------------------------추가
_TEXT_FIRST = True  # --------------------------------------추가


STATIC_ROOT = os.path.abspath("static")
os.makedirs(STATIC_ROOT, exist_ok=True)

TEXT_DEVICE: Optional[str] = None
IMAGE_DEVICE: Optional[str] = None

# ---------- 프롬프트 ----------
def build_system_prompt(st: Dict[str, Any]) -> str:
    return (
        f"너는 {st['genre']} 장르의 동화를 쓰는 작가야. 이야기는 {st['era']} 시대의 {st['location']}에서 시작되며, "
        f"주인공은 {st['characteristics']} {st['personality']}인 '{st['name']}'이야. "
        f"각 분기마다 사용자 선택에 따라 4개의 선택지를 제공하고, " # 마지막 1개는 사용자 입력용 고정 문구로 구성해줘. 
        f"이야기는 총 {st['ENDING_POINT']}장인 이야기이고 {st['ENDING_POINT']}페이지에 잘 끝나도록 이야기 길이를 조절해줘. "
        f"image_prompt에는 주인공 {st['name']}의 {st['personality']}가 잘 묘사되어야 하고 반드시 image_prompt만 영어로 작성해야 해, "
        f"나머지 텍스트는 한국어로 작성하고, 배경인 {st['location']}, 시대 {st['era']}, 장르 {st['genre']}의 분위기도 잘 표현해야 해. "
        "story는 200글자로 이내로 작성해줘야하고, '마침표(.)'뒤에는 '\n'를 포함시켜서 줄바꿈 시켜줘"
        "형식은 다음 JSON 스키마를 따라야 해:\n"
        "{\n"
        '  "story": "...",\n'
        '  "image_prompt": "...",\n'
        '  "scene_tags": ["...", "..."],\n'
        '  "character_state": {"emotion": "...", "action": "...", "place": "..."},\n'
        '  "choices": {\n'
        '    "pageN1": "...",\n'
        '    "pageN2": "...",\n'
        '    "pageN3": "...",\n'
        '    "pageN4": "..."\n'
        # '    "pageN4": "(당신이 직접 선택지를 입력해 보세요!)"\n'
        "  }\n"
        "}\n"
        "항상 순수 JSON만 응답해."
    )

# ---------- 모델/파이프 로딩 ----------
def init_models(gpu_text: Optional[int]=0, gpu_image: Optional[int]=1):
    global TOK, LLM, PIPE, _INIT_DONE, TEXT_DEVICE, IMAGE_DEVICE
    if _INIT_DONE:
        print("[Init] Already initialized; skip.")
        return

    text_dev = normalize_device(f"cuda:{gpu_text}" if gpu_text is not None else "cpu")
    img_dev  = normalize_device(f"cuda:{gpu_image}" if gpu_image is not None else text_dev)

    with _INIT_LOCK:
        if _INIT_DONE:
            return
        t0 = time.time()

        # 1) 텍스트 먼저 로드
        TOK, LLM = get_text_model(device=text_dev)

        # 2) 실제 올라간 장치 기록 (이 시점에 LLM 존재)
        try:
            TEXT_DEVICE = get_tensor_device(LLM)
        except Exception:
            TEXT_DEVICE = text_dev

        # 3) 이미지 파이프라인: lazy면 지금은 로드하지 않음
        if _IMAGE_LAZY:
            PIPE = None
        else:
            PIPE = get_image_pipe(device=img_dev)

        # 4) 이미지 실행 대상 기록 (lazy라도 기록만)
        IMAGE_DEVICE = img_dev

        _INIT_DONE = True
        print(
            f"[StoryModel] Models ready. text_dev={TEXT_DEVICE}, "
            f"img_dev={'(lazy)' if PIPE is None else IMAGE_DEVICE} ({time.time()-t0:.1f}s)"
        )
# ------------------추가-------------------
def _run_generate(*, input_ids, attention_mask, max_new_tokens=280,
                  do_sample=True, temperature=0.9, top_p=0.95,
                  pad_token_id=None, use_cache=True):
    model = LLM
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=use_cache,
        )

def _generate_branch(st: Dict[str, Any], branch_prefix: str, selected_choice: str = "") -> Dict[str, Any]:

    system_prompt = st["system_prompt"]
    ENDING_POINT = st["ENDING_POINT"]
    remaining = int(ENDING_POINT - st["ending_count"])
    ending_hint = (
        f"\n(이제 엔딩까지 {remaining} 장 남았습니다. 이야기의 복선 회수와 정리의 단서를 조금씩 드러내세요.)"
        if remaining <= max(1, int(ENDING_POINT * 0.33)) else ""
    )

    if branch_prefix == "page0":
        user_prompt = (
            f"'{st['characteristics']}하고 {st['personality']}'인 "
            f"'{st['name']}'이 어떻게 이 모험을 시작하게 되었는지 중심으로 "
            f'"{branch_prefix}"(프롤로그)를 작성해줘. '
            "story는 200글자로 이내로 작성해줘야하고, '마침표(.)'뒤에는 '\n'를 포함시켜서 줄바꿈 시켜줘"
            "image_prompt만 반드시 영어로 작성해. "
            "응답은 반드시 순수 JSON 형식으로만 작성하고, 주석/설명/마크다운은 절대 금지. "
            "아래 형식을 정확히 지켜줘:\n"
            "{\n"
            '  "story": "<서술적 이야기(한국어)>",\n'
            '  "image_prompt": "<Describe this scene in ENGLISH for image generation>",\n'
            '  "scene_tags": ["...", "..."],\n'
            '  "character_state": {"emotion": "...", "action": "...", "place": "..."},\n'
            '  "choices": {\n'
            f'    "{branch_prefix}-1": "행동 선택지 1",\n'
            f'    "{branch_prefix}-2": "행동 선택지 2",\n'
            f'    "{branch_prefix}-3": "행동 선택지 3",\n'
            f'    "{branch_prefix}-4": "행동 선택지 4"\n'
            "  }\n"
            "}\n"
        )
    else:
        recent_ctx = build_recent_context(st)
        safe_choice = sanitize_user_text(selected_choice)
        user_prompt = (
            f"[최근 내용 요약]\n{recent_ctx or '(요약 없음)'}\n\n"
            f"[사용자 선택]\n«{safe_choice}»\n"
            f"[작성 지시]\n\"{branch_prefix}\" 다음 장면을 이어서 서술해줘.{ending_hint} "
            "story는 200글자로 이내로 작성해줘야하고, '마침표(.)'뒤에는 '\n'를 포함시켜서 줄바꿈 시켜줘"
            "image_prompt는 반드시 영어로 작성해. "
            "응답은 반드시 순수 JSON 형식으로만 작성하고, 주석/설명/마크다운은 절대 금지. "
            "아래 형식을 지켜줘:\n"
            "{\n"
            '  "story": "<서술적 이야기>",\n'
            '  "image_prompt": "<Describe this scene in ENGLISH for image generation>",\n'
            '  "scene_tags": ["...", "..."],\n'
            '  "character_state": {"emotion": "...", "action": "...", "place": "..."},\n'
            '  "choices": {\n'
            f'    "{branch_prefix}-1": "행동 선택지 1",\n'
            f'    "{branch_prefix}-2": "행동 선택지 2",\n'
            f'    "{branch_prefix}-3": "행동 선택지 3",\n'
            f'    "{branch_prefix}-4": "행동 선택지 4"\n'
            "  }\n"
            "}\n"
        )

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tok, model = TOK, LLM
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)

    try:
        embed_device = model.model.embed_tokens.weight.device
    except Exception:
        embed_device = next(model.parameters()).device
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}

    # === 모델 실행 ===
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )

    # === 응답 디코딩/파싱 ===
    prompt_len = inputs["input_ids"].shape[1]
    reply = tok.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    
    story_dict = None
    try:
        cleaned = extract_json_object(strip_code_block(reply))
        story_dict = json.loads(cleaned)
    except json.JSONDecodeError:
        story_dict = {"error": "json_decode_failed", "raw_reply": reply}

    # === 리페어 1회 시도 ===
    if not ensure_choices(story_dict, branch_prefix):
        repair_msg = (
            "이전 출력이 스키마를 어겼습니다. 오직 JSON만 다시 출력하세요. "
            f'반드시 "choices" 안에 {branch_prefix}-1 ~ {branch_prefix}-4 키를 정확히 포함하세요.'
        )
        msgs += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": repair_msg},
        ]
        prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        reply = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        cleaned = extract_json_object(strip_code_block(reply))
        try:
            story_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            story_dict = {"error": reply}

    # === 상태 기록 ===
    st["chapters"].append({
        "path": branch_prefix,
        "user_request": f"{branch_prefix} 분기",
        "ai_story": story_dict,
        "is_ending": False,
    })
    st["current_index"] = int(branch_prefix.replace("page", ""))
    st["last_page_path"] = branch_prefix

    return story_dict

def _generate_ending(st: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = st["system_prompt"]
    name = st["name"]
    summary = "\n".join(
        (ch.get("ai_story") or {}).get("story", "")
        for ch in st["chapters"] if isinstance(ch.get("ai_story"), dict)
    )
    user_prompt = (
        f"지금까지의 이야기 흐름 요약:\n\"{summary}\"\n\n"
        f"이제 '{name}'의 모험을 마무리하는 엔딩 장면을 작성해줘. "
        "감정적 여운이 남도록 서술적이며 명확한 결말로 완결짓고, 선택지는 포함하지 마. "
        "story는 200글자로 이내로 작성해줘야하고, '마침표(.)'뒤에는 '\n'를 포함시켜서 줄바꿈 시켜줘"
        "출력은 오직 JSON. 마크다운/설명 금지.\n"
        "{\n"
        '  "story": "<엔딩 내용(한국어)>",\n'
        '  "image_prompt": "<Describe this ending scene in ENGLISH>",\n'
        '  "scene_tags": ["...", "..."],\n'
        '  "character_state": {"emotion": "...", "action": "...", "place": "..."},\n'
        '  "choices": null\n'
        "}"
    )

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tok, model = TOK, LLM
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)
    try:
        embed_device = model.model.embed_tokens.weight.device
    except Exception:
        embed_device = next(model.parameters()).device
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _run_generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=600,
            do_sample=True, temperature=0.9, top_p=0.95,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    reply = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    cleaned = extract_json_object(strip_code_block(reply))
    try:
        story_dict = json.loads(cleaned)
    except json.JSONDecodeError:    
        repair_msg = (
            "앞선 출력이 형식을 어겼습니다. 오직 JSON만 다시 출력하세요. "
            '형식: {"story":"...","image_prompt":"...","scene_tags":["..."],'
            '"character_state":{"emotion":"...","action":"...","place":"..."},"choices":null}'
        )
        msgs += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": repair_msg},
        ]
        prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=320,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        reply = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        cleaned = extract_json_object(strip_code_block(reply))
        try:
            story_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            story_dict = {"error": reply}

    end_path = f"page{st['current_index']}.end"
    story_dict["is_ending"] = True

    # 전체 이야기 구성
    ending_story = story_dict.get("story", "")
    total_story_content = summary + "\n\n" + ending_story
    story_dict["total_story"] = total_story_content

    st["chapters"].append({
        "path": end_path,
        "user_request": "엔딩",
        "ai_story": story_dict,
        "is_ending": True,
    })
    st["finished"] = True
    st["last_page_path"] = end_path
    return story_dict

def generate_title(session_id: str) -> str:
    st = SESSIONS.get(session_id)
    if not st:
        raise ValueError("Invalid session_id")
    if not st.get("finished"):
        raise ValueError("Story is not finished yet")

    # 마지막 챕터의 ai_story에서 total_story를 가져옵니다.
    try:
        total_story = st["chapters"][-1]["ai_story"]["total_story"]
    except (KeyError, IndexError):
        return "제목을 생성할 수 없습니다: 전체 이야기를 찾지 못했습니다."

    # LLM에 제목 생성을 요청하는 프롬프트
    title_prompt = (
        f"다음은 '{st['genre']}' 장르의 '{st['era']} 시대에 살았던 '주인공이 '{st['name']}'동화입니다. "
        "이 이야기의 내용을 기반으로 어린 아이들이 좋아할 멋진 제목을 딱 하나만 추천해 주세요. "
        "오직 제목만 응답하고, 다른 설명이나 따옴표는 붙이지 마세요.\n\n"
        f"--- 이야기 내용 ---\n{total_story}"
    )

    msgs = [
        # 시스템 역할 없이 사용자의 직접적인 요청으로 구성
        {"role": "user", "content": title_prompt},
    ]

    tok, model = TOK, LLM
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)

    try:
        embed_device = model.model.embed_tokens.weight.device
    except Exception:
        embed_device = next(model.parameters()).device
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}

    # === 모델 실행 ===
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,  # 제목은 길지 않으므로 토큰 수를 줄입니다.
            do_sample=True,
            temperature=0.8,
            pad_token_id=tok.eos_token_id,
        )

    # === 응답 디코딩 및 정리 ===
    prompt_len = inputs["input_ids"].shape[1]
    title = tok.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    
    # 불필요한 따옴표나 "제목:" 같은 접두사 제거
    title = title.replace('"', '').replace("'", "").removeprefix("제목:").strip()

    return title



# ---------- 외부로 노출할 API ----------
def create_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    # payload: {
    #   name, personality, characteristics,
    #   location, era, genre, ENDING_POINT
    # }
   
    import uuid
    session_id = uuid.uuid4().hex
    st = {
        "session_id": session_id,
        "name": payload["name"],
        "personality": payload["personality"],
        "characteristics": payload["characteristics"],
        "location": payload["location"],
        "era": payload["era"],
        "genre": payload["genre"],
        "ENDING_POINT": int(payload["ENDING_POINT"]),
        "ending_count": 0,
        "current_index": 0,
        "chapters": [],
        "finished": False,
        "last_page_path": None,
    }
    st["system_prompt"] = build_system_prompt(st)
    SESSIONS[session_id] = st

    # page0 생성
    page = _generate_branch(st, "page0", selected_choice="")
    img_dir = os.path.join(STATIC_ROOT, session_id)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "page0.png")
    img_url = None
    prompt = (page or {}).get("image_prompt", "")

    if isinstance(page, dict) and "error" not in page and prompt:
        if make_img_sync(prompt, img_path, exec_device=IMAGE_DEVICE):          
            img_url = f"/static/{session_id}/page0.png"  # 응답에 포함

    return {
        "session_id": session_id,
        "page_index": 0,
        "page": page,
        "image_url": img_url,
    }

def choose(session_id: str, choice: int, custom_text: Optional[str] = None) -> Dict[str, Any]:
    st = SESSIONS.get(session_id)
    if not st:
        raise ValueError("Invalid session_id")
    if st.get("finished"):
        return {"finished": True, "page_index": st["current_index"], "page": st["chapters"][-1]["ai_story"], "image_url": None}

    # 선택 텍스트 결정
    prev_story = (st["chapters"][-1]["ai_story"] if st["chapters"] else {}) or {}
    prev_choices = prev_story.get("choices", {}) or {}
    if choice == 4:
        if not custom_text:
            raise ValueError("custom_text is required for choice 4")
        selected_text = sanitize_user_text(custom_text)
    else:
        prev_idx = st["current_index"]
        key_dash   = f"page{prev_idx}-{choice}"
        key_nodash = f"page{prev_idx}{choice}"
        selected_text = sanitize_user_text(
        prev_choices.get(key_dash) or prev_choices.get(key_nodash) or prev_choices.get(str(choice)) or "(선택지 없음)"
        )

    # 카운트/인덱스 진행
    st["ending_count"] += 1
    st["current_index"] += 1
    branch_prefix = f"page{st['current_index']}"

    # 엔딩 여부
    if st["ending_count"] >= st["ENDING_POINT"]:
        end_page = _generate_ending(st)
        prompt = (end_page or {}).get("image_prompt", "")
        img_url = None
        if isinstance(end_page, dict) and "error" not in end_page and prompt:
            img_dir = os.path.join(STATIC_ROOT, st["session_id"])
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{branch_prefix}.end.png")
            if make_img_sync(prompt, img_path, exec_device=IMAGE_DEVICE):
                img_url = f"/static/{st['session_id']}/{branch_prefix}.end.png"
        return {
            "finished": True,
            "page_index": st["current_index"],
            "page": end_page,
            "image_url": img_url
        }

    # (엔딩이 아닐 때)
    page = _generate_branch(st, branch_prefix, selected_choice=selected_text)

    prompt = (page or {}).get("image_prompt", "")
    img_url = None
    if isinstance(page, dict) and "error" not in page and prompt:
        img_dir = os.path.join(STATIC_ROOT, st["session_id"])
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"{branch_prefix}.png")
        if make_img_sync(prompt, img_path, exec_device=IMAGE_DEVICE):
            img_url = f"/static/{st['session_id']}/{branch_prefix}.png"

    return {
        "finished": False,
        "page_index": st["current_index"],
        "page": page,
        "image_url": img_url
    }

def get_state(session_id: str) -> Dict[str, Any]:
    st = SESSIONS.get(session_id)
    if not st:
        raise ValueError("Invalid session_id")
    return {
        "session_id": session_id,
        "current_index": st["current_index"],
        "ending_count": st["ending_count"],
        "ENDING_POINT": st["ENDING_POINT"],
        "finished": st["finished"],
        "last_page_path": st["last_page_path"],
        "chapters_len": len(st["chapters"]),
    }
