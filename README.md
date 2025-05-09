## 📌 컨벤션 


### 커밋 메시지

| message | description |
| --- | --- |
| feat | 새로운 기능 추가, 기존 기능을 요구 사항에 맞추어 수정 |
| fix | 기능에 대한 버그 수정 |
| docs | 문서(주석) 수정 |
| style | 코드 스타일, 포맷팅에 대한 수정 |
| refact | 기능 변화가 아닌 코드 리팩터링 |
| test | 테스트 코드 추가/수정 |
| chore | 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore |

## 프로젝트 구조

```
├── src/
│   ├── __init__.py
│   ├── main.py                # entry point (FastAPI 객체 생성)
│   ├── api/                   # 라우팅 구성
│   │   ├── __init__.py
│   │   ├── image_route.py     # 이미지 처리 관련 API 엔드포인트
│   │   └── chat_route.py      # GPT 호출 관련 API 엔드포인트
│   ├── chat/
│   │   ├── __init__.py
│   │   └── gpt_handler.py
│   ├── image/         
│   │   ├── __init__.py
│   │   ├── trained_model/
│   │   │   ├── model.pth      # quickdraw 기반 분류 모델
│   │   ├── classifier.py      # quickdraw 기반 분류 기능
│   │   ├── model.py           # CNN 모델 정의
│   │   ├── img_caption.py     # BLIP 기반 captioning 기능 
│   │   ├── preprocessor.py    # 이미지 전처리
│   │   └── text_masking.py    # easyocr 기반 텍스트 마스킹 기능
```
