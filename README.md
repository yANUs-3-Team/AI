# MGMG: AI 기반 스토리 및 이미지 생성 엔진

## 프로젝트 개요

MGMG는 사용자가 정의한 매개변수를 기반으로 동적인 스토리를 생성하고, 각 장면에 맞는 이미지를 함께 생성하는 AI 기반 애플리케이션입니다. FastAPI를 통해 API 형태로 기능을 제공하며, 텍스트 생성 모델과 이미지 생성 모델을 활용하여 풍부한 사용자 경험을 제공합니다.

## 주요 기능

*   **맞춤형 스토리 생성:** 주인공의 특성, 배경(시대, 장소), 장르 등을 설정하여 독창적인 스토리를 시작합니다.
*   **분기형 스토리 전개:** 사용자 선택에 따라 스토리가 다양한 방향으로 전개되며, 몰입감 있는 경험을 제공합니다.
*   **장면별 이미지 생성:** 스토리의 주요 장면에 어울리는 이미지를 자동으로 생성하여 시각적인 요소를 더합니다.
*   **스토리 제목 생성:** 완료된 스토리에 대한 적절한 제목을 AI가 생성합니다.
*   **FastAPI 기반 API:** RESTful API를 통해 다른 애플리케이션과의 연동 및 확장이 용이합니다.

## 설정 및 실행

### 1. 환경 설정

Python 3.9 이상 버전이 필요합니다. 가상 환경 사용을 권장합니다.

```bash
# 가상 환경 생성 (mgmg는 가상 환경 이름)
python -m venv mgmg

# 가상 환경 활성화
# Windows
.\mgmg\Scripts\activate
# Linux/macOS
source mgmg/bin/activate
```

### 2. 종속성 설치

프로젝트 루트 디렉토리에서 `requirements.txt`에 명시된 모든 종속성을 설치합니다.

```bash
pip install -r requirements.txt
```

**GPU 사용을 위한 PyTorch 설치 (선택 사항):**
만약 NVIDIA GPU를 사용하여 모델을 가속화하려면, CUDA 버전에 맞는 PyTorch를 설치해야 합니다. 예를 들어, CUDA 11.8을 사용하는 경우:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
자신의 CUDA 버전에 맞는 설치 명령은 [PyTorch 공식 웹사이트](https://pytorch.org/get-started/locally/)에서 확인할 수 있습니다.

### 3. 모델 준비

프로젝트는 `stable-diffusion-xl-base-1.0`과 같은 사전 훈련된 모델을 사용합니다. 필요한 모델 파일들이 `stable-diffusion-xl-base-1.0/` 디렉토리 내에 올바르게 위치하는지 확인하십시오. 모델 로딩은 애플리케이션 시작 시 자동으로 처리됩니다.

### 4. 애플리케이션 실행

프로젝트 루트 디렉토리에서 다음 명령어를 사용하여 FastAPI 서버를 시작합니다.

```bash
python app.py
```

서버가 성공적으로 시작되면, 기본적으로 `http://0.0.0.0:8000`에서 API를 사용할 수 있습니다.

## 프로젝트 구조

```
.
├── app.py                  # 메인 FastAPI 애플리케이션
├── requirements.txt        # Python 종속성 목록
├── AI/                     # AI 핵심 로직 및 유틸리티
│   ├── __init__.py         # AI 파일 모듈화
│   ├── model_loader.py     # AI 모델 로딩 관련 로직
│   ├── story_engine.py     # 스토리 생성 및 관리 핵심 로직
│   └── utils.py            # 공통 유틸리티 함수
├── static/                 # 생성된 이미지 파일 저장 디렉토리
├── cache/                  # 모델 캐싱 디렉토리
├── stable-diffusion-xl-base-1.0/ # 사전 훈련된 Stable Diffusion 모델 파일
├── loras/                  # LoRA 모델 저장 디렉토리 (선택 사항)
├── test/                   # 테스트 관련 파일
└── .gitignore              # Git 버전 관리 제외 파일
```

## API 명세서

### 기본 정보

*   **Base URL:** `http://0.0.0.0:8000` (기본값, 환경 변수 `HOST` 및 `PORT`로 변경 가능)
*   **API 버전:** 1.0.0
*   **인증:** 필요 없음

### 엔드포인트

#### 1. 상태 확인 (Health Check)

*   **GET** `/health`
*   **설명:** API 서버의 상태 및 모델 로딩 여부를 확인합니다.
*   **응답:**
    *   `200 OK`
    ```json
    {
      "ok": true,
      "cuda": true,       // CUDA 사용 가능 여부
      "gpus": 2,          // 감지된 GPU 수
      "models_ready": true // 모델 로딩 완료 여부
    }
    ```

#### 2. 새 스토리 세션 생성

*   **POST** `/sessions`
*   **설명:** 새로운 스토리 세션을 생성하고, 초기 스토리 페이지를 반환합니다.
*   **요청 본문 (`application/json`):**
    *   `SessionCreateIn` 스키마
    ```json
    {
      "name": "string",         // 주인공 이름
      "personality": "string",  // 주인공 성격
      "characteristics": "string", // 주인공 특징
      "location": "string",     // 스토리 배경 장소
      "era": "string",          // 스토리 배경 시대
      "genre": "string",        // 스토리 장르
      "ending_point": 1         // 스토리의 총 장 수 (1 이상)
    }
    ```
*   **응답:**
    *   `200 OK`
    *   `FlatPage` 스키마 (초기 스토리 페이지 정보)
    ```json
    {
      "page_number": 0,
      "story": "string",        // 생성된 스토리 내용
      "session_id": "string",   // 생성된 세션 ID
      "image": "string or null",// 생성된 이미지 URL (없을 수 있음)
      "choices_1": "string or null", // 선택지 1
      "choices_2": "string or null", // 선택지 2
      "choices_3": "string or null", // 선택지 3
      "choices_4": "string or null"  // 선택지 4
    }
    ```
    *   `400 Bad Request`: 요청 본문이 유효하지 않은 경우
    *   `500 Internal Server Error`: 서버 내부 오류

#### 3. 스토리 선택 진행

*   **POST** `/sessions/{session_id}/choose`
*   **설명:** 주어진 세션 ID에 대해 사용자의 선택에 따라 스토리를 다음 단계로 진행합니다.
*   **경로 매개변수:**
    *   `session_id`: `string` (스토리 세션 ID)
*   **요청 본문 (`application/json`):**
    *   `ChooseIn` 스키마
    ```json
    {
      "choice_id": "string" // 사용자가 선택한 선택지의 ID (예: "1", "2", "3", "4")
                            // 또는 사용자 정의 선택지 텍스트 (choice_id가 "4"일 경우)
    }
    ```
*   **응답:**
    *   `200 OK`
    *   `FlatPage` 스키마 (다음 스토리 페이지 정보)
    ```json
    {
      "finished": false,        // 스토리가 완료되었는지 여부
      "page_index": 1,          // 현재 페이지 인덱스
      "page": {},               // 스토리 내용 및 관련 데이터 (딕셔너리)
      "image_url": "string or null" // 생성된 이미지 URL (없을 수 있음)
    }
    ```
    *   `400 Bad Request`: 요청 본문이 유효하지 않거나, `choice_id`가 유효하지 않은 경우
    *   `404 Not Found`: `session_id`가 유효하지 않은 경우
    *   `500 Internal Server Error`: 서버 내부 오류

#### 4. 세션 상태 조회

*   **GET** `/sessions/{session_id}/state`
*   **설명:** 특정 스토리 세션의 현재 상태를 조회합니다.
*   **경로 매개변수:**
    *   `session_id`: `string` (스토리 세션 ID)
*   **응답:**
    *   `200 OK`
    ```json
    {
      "session_id": "string",
      "current_index": 0,     // 현재 스토리 페이지 인덱스
      "ending_count": 0,      // 엔딩까지 남은 카운트
      "ENDING_POINT": 2,      // 총 엔딩 포인트
      "finished": false,      // 스토리가 완료되었는지 여부
      "last_page_path": "string or null", // 마지막 페이지 경로
      "chapters_len": 1       // 현재까지 생성된 챕터 수
    }
    ```
    *   `404 Not Found`: `session_id`가 유효하지 않은 경우

#### 5. 스토리 제목 생성 요청

*   **POST** `/sessions/{session_id}/title`
*   **설명:** 완료된 스토리에 대한 제목 생성을 요청합니다. 스토리가 완료되지 않은 경우 오류를 반환합니다.
*   **경로 매개변수:**
    *   `session_id`: `string` (스토리 세션 ID)
*   **요청 본문:** 없음
*   **응답:**
    *   `200 OK`
    ```json
    {
      "title": "string" // 생성된 스토리 제목
    }
    ```
    *   `400 Bad Request`: `session_id`가 제공되지 않은 경우
    *   `404 Not Found`: `session_id`가 유효하지 않거나 스토리가 완료되지 않은 경우
    *   `500 Internal Server Error`: 서버 내부 오류
