# ViaVox: 한국어 음성 및 화자 인식 기반 음성 노트

## 프로젝트 개요

ViaVox는 딥러닝을 활용하여 **한국어 음성 인식** 및 **화자 분리** 기능을 제공하는 음성 노트 애플리케이션입니다. 사용자는 음성을 녹음하고 이를 텍스트로 변환(STT)할 수 있으며, 개별 화자를 구별하고 대화 내용을 요약하는 기능을 제공합니다.

## 주요 기능

- **음성 인식(STT)**: Whisper-Large-V2 모델을 한국어 데이터셋을 사용하여 파인튜닝을 진행, 음성을 텍스트로 변환. (huggingface의 Junhoee/STT_Korean_Dataset 데이터 사용)
- **화자 분활**: Pyannote 모델을 한국어 데이터셋을 사용하여 파인튜닝을 진행, 개별 화자를 구분하여 발언자를 분활. (ai hub의 주요 영역별 회의 음성 데이터 사용)
- **대화 요약**: OpenAI GPT API를 사용하여 대화 또는 회의 내용을 요약.

- **데이터 출처**: (https://huggingface.co/datasets/Junhoee/STT_Korean_Dataset)
- **데이터 출처**: (https://www.aihub.or.kr/)
## 리포지토리 구조
```bash
├── .gitignore
├── Applying.ipynb
├── Finetune.ipynb
├── PipeTest.ipynb
├── Preprocess.ipynb
├── README.md
├── app.py
└── requirements.txt
```

### 리포지토리 설명
- **app/**: 프론트엔드 및 백엔드 소스 코드.
- **Finetune**:  화자 세그먼트(분할) 모델 파인튜닝 코드.
- **PipeTest**: 파인튜닝된 모델을 기반으로 나머지 파이프라인을 하이퍼파라미터 튜닝.
- **Complted**: 최종적으로 완성된 파인튜닝 모델과 파이프라인.
- **requirements.txt**: 의존성 파일.


## 설치 방법

### 요구 사항

- OpenAI GPT API Key
- HuggingFace Token

### 설치 절차

1. **리포지토리 클론**
```bash
   git clone https://github.com/yourusername/viavox.git
   cd viavox
```
2. **Python 패키지 설치**
```bash
  pip install -r requirements.txt
```
3. **Node.js 패키지 설치**
```bash
  npm install
```
4. **GPT API 키 설정 .env 파일을 생성하고, OpenAI GPT API 키를 추가하세요.**
```bash
  OPENAI_API_KEY=your-api-key
```


## 기여 방법

1. **프로젝트 포크**: 이 GitHub 프로젝트를 자신의 계정으로 포크합니다.
2. **새 브랜치 생성**: 개발할 기능 또는 수정할 내용에 대한 새 브랜치를 만듭니다.(git checkout -b feature/new-feature)
3. **변경 사항 커밋**: 개선사항이나 추가 기능을 개발하고 이를 커밋합니다.(git commit -m 'Add new feature')
4. **브랜치 푸시**: 변경사항을 GitHub에 푸시합니다.(git push origin feature/new-feature)
5. **Pull Request를 생성하세요**

e-mail : tobaky36@gmail.com

## 라이선스
이 프로젝트는 MIT 라이선스에 따라 배포됩니다.
