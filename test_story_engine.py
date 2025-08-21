# test_story_engine.py
import os, sys, time, faulthandler, threading, traceback, torch
from pathlib import Path

print("=== test_story_engine.py ===")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"cuda device count      : {torch.cuda.device_count()}")

try:
    import AI.story_engine as SM
except Exception as e:
    print("[FAIL] import AI.story_engine:", e)
    sys.exit(1)

LOG = open("freeze.log", "w", encoding="utf-8")
faulthandler.enable(file=LOG)
faulthandler.dump_traceback_later(timeout=30, repeat=True, file=LOG)  # 30초마다 반복 덤프

# 2) 수동 덤프도 가능: Windows에선 Ctrl+Break(=SIGBREAK)로 덤프
try:
    import signal
    faulthandler.register(getattr(signal, "SIGBREAK", signal.SIGINT), file=LOG)
except Exception:
    pass

def dump_all_threads(tag="MANUAL"):
    LOG.write(f"\n===== {tag} DUMP =====\n"); LOG.flush()
    frames = sys._current_frames()
    for th in threading.enumerate():
        fid = th.ident
        LOG.write(f"\n--- Thread: {th.name} ({fid}) ---\n")
        tb = "".join(traceback.format_stack(frames.get(fid)))
        LOG.write(tb)
        LOG.flush()
        
# ---------- 초기화 ----------
try:
    gt = 0 if torch.cuda.device_count() >= 1 else None
    gi = 0 if torch.cuda.device_count() >= 1 else None  # lazy면 어차피 나중에 로드
    SM.init_models(gpu_text=gt, gpu_image=gi)
    print("[OK] init_models called")
    print(f"STATIC_ROOT: {SM.STATIC_ROOT}")
except Exception as e:
    print("[FAIL] init_models:", e)
    sys.exit(2)

# ---------- 세션 생성 ----------
try:
    payload = {
        "name": "윈터",
        "personality": "호기심 많고 용감한",
        "characteristics": "금발 단발머리의 작은 소녀",
        "location": "중세 성",
        "era": "중세",
        "genre": "판타지",
        "ENDING_POINT": 2
    }
    out = SM.create_session(payload)
    print("[OK] create_session returned keys:", list(out.keys()))
    # JSON 1회 응답 형태 확인
    page = out.get("page", {})
    print("[PAGE.story] len:", len((page or {}).get("story","")))
    print("[PAGE.choices] keys:", list((page or {}).get("choices", {}).keys())[:4])
    print("[PAGE.image_url]:", out.get("image_url"))
    sid = out.get("session_id")

    # 이미지 파일 존재 여부(동기 생성이 아니라면 아직 없을 수도 있음)
    if out.get("image_url"):
        img_path = Path(SM.STATIC_ROOT) / sid / "page0.png"
        exists = img_path.exists() and img_path.stat().st_size > 0
        print(f"[IMAGE EXISTS] {exists} → {img_path}")
    else:
        print("[INFO] image_url is None (정상일 수 있음: 동기 생성 미적용/프롬프트 없음 등)")

    print("=== PASS: story_engine create_session probe ===")

    # ---------- 제목 생성 테스트 ----------
    try:
        # generate_title을 테스트하기 위해 세션을 강제로 '완료' 상태로 만듭니다.
        # 실제 시나리오에서는 choose 함수를 통해 스토리가 진행되어야 합니다.
        st = SM.SESSIONS.get(sid)
        if st:
            st["finished"] = True
            # generate_title이 total_story를 찾을 수 있도록 더미 챕터를 추가합니다.
            st["chapters"].append({
                "path": "page_end",
                "user_request": "엔딩",
                "ai_story": {
                    "story": "이것은 테스트 스토리의 내용입니다. 제목 생성을 위한 충분한 텍스트입니다.",
                    "image_prompt": "test image prompt",
                    "scene_tags": [],
                    "character_state": {},
                    "choices": None,
                    "total_story": "이것은 테스트 스토리의 내용입니다. 제목 생성을 위한 충분한 텍스트입니다."
                },
                "is_ending": True,
            })
            
            title = SM.generate_title(sid)
            print(f"[OK] generate_title returned: {title}")
            assert isinstance(title, str) and len(title) > 0, "Generated title is empty or not a string"
            print("=== PASS: story_engine generate_title probe ===")
        else:
            print("[FAIL] generate_title: Session not found for testing.")
            sys.exit(4)

    except Exception as e:
        print("[FAIL] generate_title:", e)
        sys.exit(5)

except Exception as e:
    print("[FAIL] create_session:", e)
    sys.exit(3)
