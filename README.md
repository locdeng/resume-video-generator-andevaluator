## 이력서/자소서 생성기 및 평가기 시스

텍스트 + 비디오 기반 자기소개 및 이력서 생성/평가 시스템

자기소개서(About Me) 자동 생성: OpenAI API 및 llama 사용

이력서/자기소개서 평가: KoBERT 기반 분류 모델

감정 분석: EfficientNet-B4 모델로 얼굴 표정 인식

자세 분석: MediaPipe를 통한 포즈/키포인트 추출

Streamlit UI 기반의 인터페이스

## 1) 주요 기능

자기소개서 생성 (About Me Generator):
OpenAI GPT + llama로 사용자 입력(경험, 기술 등)을 바탕으로 자연스러운 자기소개 생성

평가 (Evaluator – KoBERT):
이력서 및 자기소개서를 업로드 → KoBERT 모델이 A–E 등급 평가 및 피드백 제공
(평가 기준은 utils/ 폴더 내 JSON 파일에서 정의됨)

감정 분석 (EfficientNet-B4):
비디오 프레임 단위로 감정을 추정하고 시간 축에 따른 변화 시각화

자세 분석 (MediaPipe):
MediaPipe로 신체/얼굴 키포인트 추출 → 자세 안정성, 시선 처리, 제스처 평가

## 2) 시스템 아키텍처

### 전체 UI 흐름


<img width="1246" height="846" alt="Image" src="https://github.com/user-attachments/assets/645aadf9-8954-42e3-83b8-8a1fd7b9f727" />


### NLP 파이프라인 (이력서/자기소개서)


<img width="1246" height="846" alt="Image" src="https://github.com/user-attachments/assets/e4a79479-160d-4a30-a5d1-1058e5ba4b32" />


### 비디오 파이프라인 (감정 + 자세)


<img width="1246" height="846" alt="Image" src="https://github.com/user-attachments/assets/a76caa02-5aa1-4646-9d7e-e5c3e837ae6c" />


## 3) 이용 방법

```bash
git clone https://github.com/locdeng/resume-video-generator-andevaluator.git
cd resume-video-generator-andevaluator

pip install -r requirements.txt

streamlit run app_final.py
```
## 4) Demo Result video


### Main Screen


<img width="1294" height="632" alt="Image" src="https://github.com/user-attachments/assets/fa2db102-e7b4-4094-b113-46dbcb7a9e31" />


### 이력서 및 자기소개서 자동 생성


[![이력서 자동 생성](https://img.youtube.com/vi/RraWgu1p2tU/0.jpg)](https://youtu.be/RraWgu1p2tU)

[![자소서 자동 생성](https://img.youtube.com/vi/I17WnnDH0XQ/0.jpg)](https://youtu.be/I17WnnDH0XQ)


### 이력서 및 자기소개서 자동 평가


[![이력서 자동 평가](https://img.youtube.com/vi/WadoYxlbYwY/0.jpg)](https://youtu.be/WadoYxlbYwY)

[![자소서 자동 생성](https://img.youtube.com/vi/c7F-3nxsyys/0.jpg)](https://youtu.be/c7F-3nxsyys)


### 감정 분석


[![감정 분석](https://img.youtube.com/vi/mhEw9oLvvbk/0.jpg)](https://youtu.be/mhEw9oLvvbk)

[![감정 분석](https://img.youtube.com/vi/-yiNEdBY8ug/0.jpg)](https://youtu.be/-yiNEdBY8ug)


