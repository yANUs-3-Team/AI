# 🌟 몽글몽글 상상나래 (MGMG-AI)

본 프로젝트의 AI 파트는 **동화 줄거리 생성 모델**과 **동화 삽화 생성 모델**을 포함하며,  
사용자 설정값을 기반으로 **프롬프트 엔지니어링**을 거쳐 스토리와 이미지를 순차적으로 생성·저장하는 시스템입니다.

---

## ✨ 주요 기능

### 1. 동화 줄거리 생성
- **LLM 기반 스토리 생성**  
  SKT A.X 계열 모델을 활용하여 사용자 입력(주인공 정보, 시대, 장르, 엔딩 카운트 등)에 맞춘 맞춤형 스토리 생성  
- **프롬프트 엔지니어링**  
  - 프롤로그 생성 → 사용자 선택지 반영 → 다음 플롯 생성 → 엔딩 생성  
  - 각 페이지마다 **4개 선택지** 제공, 사용자 선택 반영  
- **JSON 기반 구조**  
  - `story`, `choices`, `path` 필드로 구성된 JSON 응답 반환  
  - 시스템에서 스토리 진행 및 엔딩 여부 판별 가능

### 2. 동화 삽화 생성
- **Stable Diffusion XL 기반 이미지 생성**  
  - LoRA(저장된 스타일/캐릭터) 적용을 통한 일관된 그림체 유지  
  - 플롯별 스토리와 장면 정보를 바탕으로 이미지 프롬프트 생성  
- **멀티 LoRA 지원**  
  - 예: `illu`(동화풍 스타일), `fantasy`(배경/분위기 강화) 동시 적용  
- **Web GUI 및 diffusers API 지원**

### 3. AI 파이프라인
1. **사용자 입력 수집** → 이름, 성격, 시대, 장르, 엔딩 카운트  
2. **스토리 생성 요청** → 프롬프트 엔지니어링 적용, JSON 응답 반환  
3. **이미지 생성 요청** → 스토리 내용을 기반으로 장면 삽화 생성  
4. **DB 저장** → 생성된 텍스트/이미지 URL 저장  
5. **반복 진행** → 엔딩 카운트 도달 시 스토리 종료

---

## 🛠️ 기술 스택

| 구분          | 기술 |
|--------------|------|
| Text Model   | SKT A.X 계열 LLM (Causal LM) |
| Image Model  | Stable Diffusion XL (diffusers) |
| Fine-tuning  | LoRA 학습 (Web GUI 기반) |
| Prompting    | 프롬프트 엔지니어링, JSON 구조 응답 |
| Framework    | PyTorch, diffusers, FastAPI |
| Language     | Python 3.10+ |

---

## 📂 프로젝트 구조

```
AI/
├── app.py               # FastAPI 서버 엔트리포인트
├── model_loader.py      # 텍스트/이미지 모델 로드 및 초기화
├── story_engine.py      # 스토리 생성, 프롬프트 엔지니어링 로직
├── requirements.txt     # Python 패키지 의존성
├── static/              # 생성된 이미지 저장 폴더
└── utils/               # 유틸 함수 (JSON 처리, 프롬프트 포맷팅 등)
```

---

## 🚀 시작하기

### 1. 사전 준비
- Python 3.10 이상
- NVIDIA GPU + CUDA 11.8 이상 (이미지 생성 시 권장)
- 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```
PyTorch는 CUDA 버전에 맞춰 별도 설치 필요:
```bash
# CUDA 12.1 예시
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# CPU-only
pip install torch torchvision torchaudio
```

### 3. 환경 변수 설정
`AI/.env` 파일 생성 (Git 추적 제외)

```env
# 텍스트 모델 캐시 경로
TEXT_MODEL_PATH=cache/AX

# 이미지 모델 캐시 경로
IMAGE_MODEL_PATH=stable-diffusion-xl-base-1.0

# LoRA 경로
LORA_PATH_1=loras/StorybookRedmondV2-KidsBook-KidsRedmAF.safetensors
LORA_PATH_2=loras/J_oil_pastels_XL.safetensors
```

### 4. 서버 실행
```bash
uvicorn app:app --reload
```
서버 실행 후 `http://localhost:8000`에서 API 호출 가능

---

## 📡 API 명세

### `POST /init`
동화의 프롤로그 또는 다음 페이지를 생성합니다.

**Request Body**
```json
{
  "name": "주인공",
  "personality": "성경",
  "era": "시대",
  "genre": "장르",
  "ending_count": 3,
  "current_path": "page0"
}
```

**Response Example**
```json
{
  "path": "page1",
  "story": "주인공은...",
  "choices": {
    "page1-1": "선택지1",
    "page1-2": "선택지2",
    "page1-3": "선택지3",
    "page1-4": "(직접 입력)"
  }
}
```

---

## 🔒 보안 및 성능 고려사항
- **모델 캐시**: 모델 로드 시 로컬 캐시 사용으로 API 응답속도 최적화
- **입력 값 검증**: 잘못된 형식의 프롬프트 방지
- **GPU 메모리 관리**: LoRA 적용 시 메모리 최적화 옵션 사용 (`torch_dtype=torch.float16`)
- **리퀘스트 큐잉**: 동시 다중 요청 시 큐 기반 처리로 안정성 확보
