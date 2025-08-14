# # story_model.py
# import os, json, torch
# from threading import Thread, Lock
# from typing import Dict, Any, Optional

# from model_loader import get_text_model, get_image_pipe

# # 전역(싱글톤)
# TOK = None
# LLM = None
# PIPE = None
# IMAGE_LOCK = Lock()
# SESSIONS: Dict[str, Dict[str, Any]] = {}

# STATIC_ROOT = os.path.abspath("static")
# os.makedirs(STATIC_ROOT, exist_ok=True)


# # ---------- 유틸 ----------
# def strip_code_block(text: str) -> str:
#     if text.startswith("```json"):
#         text = text[len("```json"):].strip()
#     elif text.startswith("```"):
#         text = text[len("```"):].strip()
#     if text.endswith("```"):
#         text = text[:-3].strip()
#     return text

# def extract_json_object(text: str) -> str:
#     start = text.find("{")
#     if start == -1:
#         return text
#     count = 0
#     for i in range(start, len(text)):
#         if text[i] == "{": count += 1
#         elif text[i] == "}":
#             count -= 1
#             if count == 0:
#                 return text[start : i + 1]
#     return text

# def clamp_prompt(p: str, max_words=70):
#     return " ".join((p or "").split()[:max_words])

# def ensure_choices(sd, branch_prefix):
#     must = [f"{branch_prefix}-1", f"{branch_prefix}-2",
#             f"{branch_prefix}-3", f"{branch_prefix}-4"]
#     return (
#         isinstance(sd, dict)
#         and isinstance(sd.get("choices"), dict)
#         and all(k in sd["choices"] for k in must)
#     )

# def generate_and_save_image(pipe, prompt: str, filename: str, offload_after=False):
#     d = os.path.dirname(filename)
#     if d: os.makedirs(d, exist_ok=True)
#     with torch.inference_mode():
#         image = pipe(
#             clamp_prompt(prompt),
#             height=832, width=832,
#             num_inference_steps=22, guidance_scale=4.5
#         ).images[0]
#     image.save(filename)
#     if offload_after:
#         try:
#             pipe.to("cpu")
#             torch.cuda.empty_cache()
#         except Exception:
#             pass
#     return pipe

# def make_img_async(pipe, prompt, path):
#     try:
#         with IMAGE_LOCK:
#             generate_and_save_image(pipe, prompt, path, offload_after=False)
#         print(f"[이미지 완료: {path}]")
#     except Exception as e:
#         print(f"[이미지 생성 실패: {e}]")


# # ---------- 프롬프트 ----------
# def build_system_prompt(st: Dict[str, Any]) -> str:
#     return (
#         f"너는 {st['genre']} 장르의 동화를 쓰는 작가야. 이야기는 {st['era']} 시대의 {st['start_location']}에서 시작되며, "
#         f"주인공은 {st['protagonist_characteristic']} {st['protagonist_appearance']}인 '{st['protagonist_name']}'이야. "
#         f"각 분기마다 사용자 선택에 따라 3개의 선택지를 제공하고, 마지막 1개는 사용자 입력용 고정 문구로 구성해줘. "
#         f"이야기는 총 {st['ENDING_POINT']}장인 이야기이고 {st['ENDING_POINT']}페이지에 잘 끝나도록 이야기 길이를 조절해줘. "
#         f"image_prompt에는 주인공 {st['protagonist_name']}의 {st['protagonist_appearance']}가 잘 묘사되어야 하고 반드시 image_prompt만 영어로 작성해야 해, "
#         f"나머지 텍스트는 한국어로 작성하고, 배경인 {st['start_location']}, 시대 {st['era']}, 장르 {st['genre']}의 분위기도 잘 표현해야 해. "
#         "형식은 다음 JSON 스키마를 따라야 해:\n"
#         "{\n"
#         '  "story": "...",\n'
#         '  "image_prompt": "...",\n'
#         '  "scene_tags": ["...", "..."],\n'
#         '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
#         '  "choices": {\n'
#         '    "pageN1": "...",\n'
#         '    "pageN2": "...",\n'
#         '    "pageN3": "...",\n'
#         '    "pageN4": "(당신이 직접 선택지를 입력해 보세요!)"\n'
#         "  }\n"
#         "}\n"
#         "항상 순수 JSON만 응답해."
#     )

# # ---------- 모델/파이프 로딩 ----------
# def init_models(gpu_text: Optional[int]=0, gpu_image: Optional[int]=1):
#     global TOK, LLM, PIPE
#     TOK, LLM = get_text_model()
#     try:
#         if gpu_text is not None:
#             LLM.to(f"cuda:{gpu_text}")
#     except Exception:
#         pass

#     PIPE = get_image_pipe()
#     try:
#         if gpu_image is not None:
#             PIPE.to(f"cuda:{gpu_image}")
#         elif gpu_text is not None:
#             PIPE.to(f"cuda:{gpu_text}")
#     except Exception:
#         pass
#     print("[StoryModel] Models ready.")


# # ---------- 생성 로직 ----------
# def _generate_branch(st: Dict[str, Any], branch_prefix: str, selected_choice: str = "") -> Dict[str, Any]:
#     system_prompt = st["system_prompt"]
#     ENDING_POINT = st["ENDING_POINT"]
#     remaining = int(ENDING_POINT - st["ending_count"])
#     ending_hint = (
#         f"\n(이제 엔딩까지 {remaining} 장 남았으니 이야기 흐름을 서서히 마무리할 준비를 해줘.)"
#         if remaining <= ENDING_POINT * 0.33 else ""
#     )

#     if branch_prefix == "page0":
#         user_prompt = (
#             f"'{st['protagonist_characteristic']}하고 {st['protagonist_appearance']}'인 "
#             f"'{st['protagonist_name']}'이 어떻게 이 모험을 시작하게 되었는지 중심으로 "
#             f'"{branch_prefix}"(프롤로그)를 작성해줘. '
#             "image_prompt만 반드시 영어로 작성해. "
#             "응답은 반드시 순수 JSON 형식으로만 작성하고, 주석/설명/마크다운은 절대 금지. "
#             "아래 형식을 정확히 지켜줘:\n"
#             "{\n"
#             '  "story": "<서술적 이야기(한국어)>",\n'
#             '  "image_prompt": "<Describe this scene in ENGLISH for image generation>",\n'
#             '  "scene_tags": ["...", "..."],\n'
#             '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
#             '  "choices": {\n'
#             f'    "{branch_prefix}-1": "행동 선택지 1",\n'
#             f'    "{branch_prefix}-2": "행동 선택지 2",\n'
#             f'    "{branch_prefix}-3": "행동 선택지 3",\n'
#             f'    "{branch_prefix}-4": "(당신이 직접 선택지를 입력해 보세요!)"\n'
#             "  }\n"
#             "}\n"
#         )
#     else:
#         user_prompt = (
#             f'"{branch_prefix}" 다음 장면을 이어서 서술해줘.{ending_hint} '
#             f'사용자가 방금 선택한 행동은 "{selected_choice}"야. 그 결과를 반영해서 다음 이야기를 써줘. '
#             "image_prompt는 반드시 영어로 작성해. "
#             "응답은 반드시 순수 JSON 형식으로만 작성하고, 주석/설명/마크다운은 절대 금지. "
#             "아래 형식을 지켜줘:\n"
#             "{\n"
#             '  "story": "<서술적 이야기>",\n'
#             '  "image_prompt": "<Describe this scene in ENGLISH for image generation>",\n'
#             '  "scene_tags": ["...", "..."],\n'
#             '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
#             '  "choices": {\n'
#             f'    "{branch_prefix}-1": "행동 선택지 1",\n'
#             f'    "{branch_prefix}-2": "행동 선택지 2",\n'
#             f'    "{branch_prefix}-3": "행동 선택지 3",\n'
#             f'    "{branch_prefix}-4": "(당신이 직접 선택지를 입력해 보세요!)"\n'
#             "  }\n"
#             "}\n"
#         )

#     msgs = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]
#     tok, model = TOK, LLM
#     prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
#     inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)

#     try:
#         embed_device = model.model.embed_tokens.weight.device
#     except Exception:
#         embed_device = next(model.parameters()).device
#     inputs = {k: v.to(embed_device) for k, v in inputs.items()}

#     with torch.inference_mode():
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=400,
#             do_sample=True, temperature=0.9, top_p=0.95,
#             pad_token_id=tok.eos_token_id,
#             use_cache=True,
#         )

#     prompt_len = inputs["input_ids"].shape[1]
#     reply = tok.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
#     cleaned = extract_json_object(strip_code_block(reply))
#     try:
#         story_dict = json.loads(cleaned)
#     except json.JSONDecodeError:
#         story_dict = {"error": reply}

#     # 리페어 1회
#     if not ensure_choices(story_dict, branch_prefix):
#         repair_msg = (
#             "이전 출력이 스키마를 어겼습니다. 오직 JSON만 다시 출력하세요. "
#             f'반드시 "choices" 안에 {branch_prefix}-1 ~ {branch_prefix}-4 키를 정확히 포함하세요.'
#         )
#         msgs += [
#             {"role": "assistant", "content": reply},
#             {"role": "user", "content": repair_msg},
#         ]
#         prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
#         inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)
#         inputs = {k: v.to(embed_device) for k, v in inputs.items()}
#         with torch.inference_mode():
#             outputs = model.generate(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=220,
#                 do_sample=False,
#                 pad_token_id=tok.eos_token_id,
#                 use_cache=True,
#             )
#         reply = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#         cleaned = extract_json_object(strip_code_block(reply))
#         try:
#             story_dict = json.loads(cleaned)
#         except json.JSONDecodeError:
#             story_dict = {"error": reply}

#     # 기록
#     st["chapters"].append({
#         "path": branch_prefix,
#         "user_request": f"{branch_prefix} 분기",
#         "ai_story": story_dict,
#         "is_ending": False,
#     })
#     st["current_index"] = int(branch_prefix.replace("page", ""))
#     st["last_page_path"] = branch_prefix
#     return story_dict

# def _generate_ending(st: Dict[str, Any]) -> Dict[str, Any]:
#     system_prompt = st["system_prompt"]
#     protagonist_name = st["protagonist_name"]
#     summary = "\n".join(
#         (ch.get("ai_story") or {}).get("story", "")
#         for ch in st["chapters"] if isinstance(ch.get("ai_story"), dict)
#     )
#     user_prompt = (
#         f"지금까지의 이야기 흐름 요약:\n\"{summary}\"\n\n"
#         f"이제 '{protagonist_name}'의 모험을 마무리하는 엔딩 장면을 작성해줘. "
#         "감정적 여운이 남도록 서술적이며 명확한 결말로 완결짓고, 선택지는 포함하지 마. "
#         "출력은 오직 JSON. 마크다운/설명 금지.\n"
#         "{\n"
#         '  "story": "<엔딩 내용(한국어)>",\n'
#         '  "image_prompt": "<Describe this ending scene in ENGLISH>",\n'
#         '  "scene_tags": ["...", "..."],\n'
#         '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
#         '  "choices": null\n'
#         "}"
#     )

#     msgs = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]
#     tok, model = TOK, LLM
#     prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
#     inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)
#     try:
#         embed_device = model.model.embed_tokens.weight.device
#     except Exception:
#         embed_device = next(model.parameters()).device
#     inputs = {k: v.to(embed_device) for k, v in inputs.items()}

#     with torch.inference_mode():
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=600,
#             do_sample=True, temperature=0.9, top_p=0.95,
#             pad_token_id=tok.eos_token_id,
#             use_cache=True,
#         )
#     reply = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#     cleaned = extract_json_object(strip_code_block(reply))
#     try:
#         story_dict = json.loads(cleaned)
#     except json.JSONDecodeError:
#         # 리페어 1회
#         repair_msg = (
#             "앞선 출력이 형식을 어겼습니다. 오직 JSON만 다시 출력하세요. "
#             '형식: {"story":"...","image_prompt":"...","scene_tags":["..."],'
#             '"character_state":{"emotion":"...","action":"...","location":"..."},"choices":null}'
#         )
#         msgs += [
#             {"role": "assistant", "content": reply},
#             {"role": "user", "content": repair_msg},
#         ]
#         prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
#         inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)
#         inputs = {k: v.to(embed_device) for k, v in inputs.items()}
#         with torch.inference_mode():
#             outputs = model.generate(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=320,
#                 do_sample=False,
#                 pad_token_id=tok.eos_token_id,
#                 use_cache=True,
#             )
#         reply = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#         cleaned = extract_json_object(strip_code_block(reply))
#         try:
#             story_dict = json.loads(cleaned)
#         except json.JSONDecodeError:
#             story_dict = {"error": reply}

#     end_path = f"page{st['current_index']}.end"
#     story_dict["is_ending"] = True
#     st["chapters"].append({
#         "path": end_path,
#         "user_request": "엔딩",
#         "ai_story": story_dict,
#         "is_ending": True,
#     })
#     st["finished"] = True
#     st["last_page_path"] = end_path
#     return story_dict


# # ---------- 외부로 노출할 API ----------
# def create_session(payload: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     payload: {
#       protagonist_name, protagonist_appearance, protagonist_characteristic,
#       start_location, era, genre, ENDING_POINT
#     }
#     """
#     import uuid
#     session_id = uuid.uuid4().hex
#     st = {
#         "session_id": session_id,
#         "protagonist_name": payload["protagonist_name"],
#         "protagonist_appearance": payload["protagonist_appearance"],
#         "protagonist_characteristic": payload["protagonist_characteristic"],
#         "start_location": payload["start_location"],
#         "era": payload["era"],
#         "genre": payload["genre"],
#         "ENDING_POINT": int(payload["ENDING_POINT"]),
#         "ending_count": 0,
#         "current_index": 0,
#         "chapters": [],
#         "finished": False,
#         "last_page_path": None,
#     }
#     st["system_prompt"] = build_system_prompt(st)
#     SESSIONS[session_id] = st

#     # page0 생성
#     page = _generate_branch(st, "page0", selected_choice="")
#     # 이미지 비동기 생성
#     img_dir = os.path.join(STATIC_ROOT, session_id)
#     os.makedirs(img_dir, exist_ok=True)
#     img_path = os.path.join(img_dir, "page0.png")
#     img_url = None
#     prompt = (page or {}).get("image_prompt", "")
#     if isinstance(page, dict) and "error" not in page and prompt:
#         img_url = f"/static/{session_id}/page0.png"
#         Thread(target=make_img_async, args=(PIPE, prompt, img_path), daemon=True).start()

#     return {
#         "session_id": session_id,
#         "page_index": 0,
#         "page": page,
#         "image_url": img_url,
#     }

# def choose(session_id: str, choice: int, custom_text: Optional[str] = None) -> Dict[str, Any]:
#     st = SESSIONS.get(session_id)
#     if not st:
#         raise ValueError("Invalid session_id")
#     if st.get("finished"):
#         return {"finished": True, "page_index": st["current_index"], "page": st["chapters"][-1]["ai_story"], "image_url": None}

#     # 선택 텍스트 결정
#     prev_story = (st["chapters"][-1]["ai_story"] if st["chapters"] else {}) or {}
#     prev_choices = prev_story.get("choices", {}) or {}
#     if choice == 4:
#         if not custom_text:
#             raise ValueError("custom_text is required for choice 4")
#         selected_text = custom_text
#     else:
#         prev_idx = st["current_index"]
#         key_dash   = f"page{prev_idx}-{choice}"
#         key_nodash = f"page{prev_idx}{choice}"
#         selected_text = prev_choices.get(key_dash) or prev_choices.get(key_nodash) or prev_choices.get(str(choice)) or "(선택지 없음)"

#     # 카운트/인덱스 진행
#     st["ending_count"] += 1
#     st["current_index"] += 1
#     branch_prefix = f"page{st['current_index']}"

#     # 엔딩 여부
#     if st["ending_count"] >= st["ENDING_POINT"]:
#         end_page = _generate_ending(st)
#         prompt = (end_page or {}).get("image_prompt", "")
#         img_url = None
#         if isinstance(end_page, dict) and "error" not in end_page and prompt:
#             img_dir = os.path.join(STATIC_ROOT, st["session_id"])
#             os.makedirs(img_dir, exist_ok=True)
#             img_path = os.path.join(img_dir, f"{branch_prefix}.end.png")
#             img_url = f"/static/{st['session_id']}/{branch_prefix}.end.png"
#             Thread(target=make_img_async, args=(PIPE, prompt, img_path), daemon=True).start()
#         return {"finished": True, "page_index": st["current_index"], "page": end_page, "image_url": img_url}

#     # 다음 분기 생성
#     page = _generate_branch(st, branch_prefix, selected_choice=selected_text)

#     # 이미지
#     prompt = (page or {}).get("image_prompt", "")
#     img_url = None
#     if isinstance(page, dict) and "error" not in page and prompt:
#         img_dir = os.path.join(STATIC_ROOT, st["session_id"])
#         os.makedirs(img_dir, exist_ok=True)
#         img_path = os.path.join(img_dir, f"{branch_prefix}.png")
#         img_url = f"/static/{st['session_id']}/{branch_prefix}.png"
#         Thread(target=make_img_async, args=(PIPE, prompt, img_path), daemon=True).start()

#     return {"finished": False, "page_index": st["current_index"], "page": page, "image_url": img_url}

# def get_state(session_id: str) -> Dict[str, Any]:
#     st = SESSIONS.get(session_id)
#     if not st:
#         raise ValueError("Invalid session_id")
#     return {
#         "session_id": session_id,
#         "current_index": st["current_index"],
#         "ending_count": st["ending_count"],
#         "ENDING_POINT": st["ENDING_POINT"],
#         "finished": st["finished"],
#         "last_page_path": st["last_page_path"],
#         "chapters_len": len(st["chapters"]),
#     }

# story_model.py
import os, json, torch
from threading import Thread, Lock
from typing import Dict, Any, Optional

from concurrent.futures import ThreadPoolExecutor
import time

from model_loader import get_text_model, get_image_pipe

# 전역(싱글톤)
TOK = None
LLM = None
PIPE = None
_INIT_DONE = False
_INIT_LOCK = Lock()
IMAGE_LOCK = Lock()
SESSIONS: Dict[str, Dict[str, Any]] = {}

STATIC_ROOT = os.path.abspath("static")
os.makedirs(STATIC_ROOT, exist_ok=True)


# ---------- 유틸 ----------
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

def generate_and_save_image(pipe, prompt: str, filename: str, seed: Optional[int]=None,
                            size: str="fast", offload_after=False):
    # 프리셋
    if size == "fast":
        H, W, steps, guidance = 832, 832, 20, 4.0
    elif size == "balanced":
        H, W, steps, guidance = 1024, 1024, 28, 5.0
    else:  # "quality"
        H, W, steps, guidance = 1152, 1152, 32, 5.5

    gen = None
    if seed is not None:
        gen = torch.Generator(device=pipe.device).manual_seed(int(seed))

    d = os.path.dirname(filename)
    if d: os.makedirs(d, exist_ok=True)
    with torch.inference_mode():
        image = pipe(
            clamp_prompt(prompt),
            height=H, width=W,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen
        ).images[0]
    image.save(filename)
    if offload_after:
        try:
            pipe.to("cpu"); torch.cuda.empty_cache()
        except Exception: pass
    return pipe


def make_img_async(pipe, prompt, path):
    try:
        with IMAGE_LOCK:
            generate_and_save_image(pipe, prompt, path, offload_after=False)
        print(f"[이미지 완료: {path}]")
    except Exception as e:
        print(f"[이미지 생성 실패: {e}]")


# ---------- 프롬프트 ----------
def build_system_prompt(st: Dict[str, Any]) -> str:
    return (
        f"너는 {st['genre']} 장르의 동화를 쓰는 작가야. 이야기는 {st['era']} 시대의 {st['start_location']}에서 시작되며, "
        f"주인공은 {st['protagonist_characteristic']} {st['protagonist_appearance']}인 '{st['protagonist_name']}'이야. "
        f"각 분기마다 사용자 선택에 따라 3개의 선택지를 제공하고, 마지막 1개는 사용자 입력용 고정 문구로 구성해줘. "
        f"이야기는 총 {st['ENDING_POINT']}장인 이야기이고 {st['ENDING_POINT']}페이지에 잘 끝나도록 이야기 길이를 조절해줘. "
        f"image_prompt에는 주인공 {st['protagonist_name']}의 {st['protagonist_appearance']}가 잘 묘사되어야 하고 반드시 image_prompt만 영어로 작성해야 해, "
        f"나머지 텍스트는 한국어로 작성하고, 배경인 {st['start_location']}, 시대 {st['era']}, 장르 {st['genre']}의 분위기도 잘 표현해야 해. "
        "형식은 다음 JSON 스키마를 따라야 해:\n"
        "{\n"
        '  "story": "...",\n'
        '  "image_prompt": "...",\n'
        '  "scene_tags": ["...", "..."],\n'
        '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
        '  "choices": {\n'
        '    "pageN1": "...",\n'
        '    "pageN2": "...",\n'
        '    "pageN3": "...",\n'
        '    "pageN4": "(당신이 직접 선택지를 입력해 보세요!)"\n'
        "  }\n"
        "}\n"
        "항상 순수 JSON만 응답해."
    )

# ---------- 모델/파이프 로딩 ----------
def init_models(gpu_text: Optional[int]=0, gpu_image: Optional[int]=1):
    global TOK, LLM, PIPE, _INIT_DONE
    if _INIT_DONE:
        print("[Init] Already initialized; skip.")
        return

    text_dev = f"cuda:{gpu_text}" if gpu_text is not None else "cpu"
    img_dev  = f"cuda:{gpu_image}" if gpu_image is not None else (text_dev if gpu_text is not None else "cpu")

    with _INIT_LOCK:
        if _INIT_DONE:
            return
        t0 = time.time()

        # 병렬 로딩
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_txt = ex.submit(get_text_model, text_dev)      # get_text_model(device=...) 시그니처면 아래처럼 래핑
            fut_img = ex.submit(get_image_pipe, img_dev)       # get_image_pipe(device=...)
            TOK, LLM = fut_txt.result()
            PIPE     = fut_img.result()

        _INIT_DONE = True
        print(f"[StoryModel] Models ready. text_dev={text_dev}, img_dev={img_dev} ({time.time()-t0:.1f}s)")

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

def _generate_branch(st: Dict[str, Any], branch_prefix: str, selected_choice: str = "") -> Dict[str, Any]:
    system_prompt = st["system_prompt"]
    ENDING_POINT = st["ENDING_POINT"]
    remaining = int(ENDING_POINT - st["ending_count"])
    ending_hint = (f"\n(이제 엔딩까지 {remaining} 장 남았습니다. 이야기의 복선 회수와 정리의 단서를 조금씩 드러내세요.)"
                   if remaining <= max(1, int(ENDING_POINT * 0.33)) else "")
    recent_ctx = build_recent_context(st)

    if branch_prefix == "page0":
        user_prompt = (
            f"'{st['protagonist_characteristic']}하고 {st['protagonist_appearance']}'인 "
            f"'{st['protagonist_name']}'이 어떻게 이 모험을 시작하게 되었는지 중심으로 "
            f'"{branch_prefix}"(프롤로그)를 작성해줘. '
            "image_prompt만 반드시 영어로 작성해. "
            "응답은 반드시 순수 JSON 형식으로만 작성하고, 주석/설명/마크다운은 절대 금지. "
            "아래 형식을 정확히 지켜줘:\n"
            "{\n"
            '  "story": "<서술적 이야기(한국어)>",\n'
            '  "image_prompt": "<Describe this scene in ENGLISH for image generation>",\n'
            '  "scene_tags": ["...", "..."],\n'
            '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
            '  "choices": {\n'
            f'    "{branch_prefix}-1": "행동 선택지 1",\n'
            f'    "{branch_prefix}-2": "행동 선택지 2",\n'
            f'    "{branch_prefix}-3": "행동 선택지 3",\n'
            f'    "{branch_prefix}-4": "(당신이 직접 선택지를 입력해 보세요!)"\n'
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
            "image_prompt는 반드시 영어로 작성해. "
            "응답은 반드시 순수 JSON 형식으로만 작성하고, 주석/설명/마크다운은 절대 금지. "
            "아래 형식을 지켜줘:\n"
            "{\n"
            '  "story": "<서술적 이야기>",\n'
            '  "image_prompt": "<Describe this scene in ENGLISH for image generation>",\n'
            '  "scene_tags": ["...", "..."],\n'
            '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
            '  "choices": {\n'
            f'    "{branch_prefix}-1": "행동 선택지 1",\n'
            f'    "{branch_prefix}-2": "행동 선택지 2",\n'
            f'    "{branch_prefix}-3": "행동 선택지 3",\n'
            f'    "{branch_prefix}-4": "(당신이 직접 선택지를 입력해 보세요!)"\n'
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

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=400,
            do_sample=True, temperature=0.9, top_p=0.95,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    reply = tok.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    cleaned = extract_json_object(strip_code_block(reply))
    try:
        story_dict = json.loads(cleaned)
    except json.JSONDecodeError:
        story_dict = {"error": reply}

    # 리페어 1회
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
                max_new_tokens=220,
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

    # 기록
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
    protagonist_name = st["protagonist_name"]
    summary = "\n".join(
        (ch.get("ai_story") or {}).get("story", "")
        for ch in st["chapters"] if isinstance(ch.get("ai_story"), dict)
    )
    user_prompt = (
        f"지금까지의 이야기 흐름 요약:\n\"{summary}\"\n\n"
        f"이제 '{protagonist_name}'의 모험을 마무리하는 엔딩 장면을 작성해줘. "
        "감정적 여운이 남도록 서술적이며 명확한 결말로 완결짓고, 선택지는 포함하지 마. "
        "출력은 오직 JSON. 마크다운/설명 금지.\n"
        "{\n"
        '  "story": "<엔딩 내용(한국어)>",\n'
        '  "image_prompt": "<Describe this ending scene in ENGLISH>",\n'
        '  "scene_tags": ["...", "..."],\n'
        '  "character_state": {"emotion": "...", "action": "...", "location": "..."},\n'
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
        outputs = model.generate(
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
        # 리페어 1회
        repair_msg = (
            "앞선 출력이 형식을 어겼습니다. 오직 JSON만 다시 출력하세요. "
            '형식: {"story":"...","image_prompt":"...","scene_tags":["..."],'
            '"character_state":{"emotion":"...","action":"...","location":"..."},"choices":null}'
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
    st["chapters"].append({
        "path": end_path,
        "user_request": "엔딩",
        "ai_story": story_dict,
        "is_ending": True,
    })
    st["finished"] = True
    st["last_page_path"] = end_path
    return story_dict


# ---------- 외부로 노출할 API ----------
def create_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: {
      protagonist_name, protagonist_appearance, protagonist_characteristic,
      start_location, era, genre, ENDING_POINT
    }
    """
    import uuid
    session_id = uuid.uuid4().hex
    st = {
        "session_id": session_id,
        "protagonist_name": payload["protagonist_name"],
        "protagonist_appearance": payload["protagonist_appearance"],
        "protagonist_characteristic": payload["protagonist_characteristic"],
        "start_location": payload["start_location"],
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
    # 이미지 비동기 생성
    img_dir = os.path.join(STATIC_ROOT, session_id)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "page0.png")
    img_url = None
    prompt = (page or {}).get("image_prompt", "")
    if isinstance(page, dict) and "error" not in page and prompt:
        img_url = f"/static/{session_id}/page0.png"
        Thread(target=make_img_async, args=(PIPE, prompt, img_path), daemon=True).start()

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
            img_url = f"/static/{st['session_id']}/{branch_prefix}.end.png"
            Thread(target=make_img_async, args=(PIPE, prompt, img_path), daemon=True).start()
        return {"finished": True, "page_index": st["current_index"], "page": end_page, "image_url": img_url}

    # 다음 분기 생성
    page = _generate_branch(st, branch_prefix, selected_choice=selected_text)

    # 이미지
    prompt = (page or {}).get("image_prompt", "")
    img_url = None
    if isinstance(page, dict) and "error" not in page and prompt:
        img_dir = os.path.join(STATIC_ROOT, st["session_id"])
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"{branch_prefix}.png")
        img_url = f"/static/{st['session_id']}/{branch_prefix}.png"
        Thread(target=make_img_async, args=(PIPE, prompt, img_path), daemon=True).start()

    return {"finished": False, "page_index": st["current_index"], "page": page, "image_url": img_url}

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
