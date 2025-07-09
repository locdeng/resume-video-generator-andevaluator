from openai import OpenAI
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
openai_client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key= os.getenv("OPENAI_API_KEY")
)
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= os.getenv("OPENROUTER_API_KEY"),
)

def build_resume_prompt(
    style, 이름, 생년월일, 이메일, 연락처, 주소,
    학교, 전공, 학력기간, 학점,
    경력사항, 기술역량, 자격증, 기타활동
):
    prompt = f"""
다음 정보를 바탕으로 한국어 이력서를 {style} 작성해 주세요:

아래 지원자 정보를 바탕으로 한국 기업 인사담당자가 읽기 좋은 이력서를 작성해 주세요.

✅ 조건
- 전문적이고 간결한 톤으로 작성해 주세요.
- 항목별로 구분된 한국어 이력서 포맷을 지켜 주세요.
- 각 항목에는 지원자의 정보를 명확하게 정리해 주세요.
- 경력사항 항목이 짧게 주어지면, 실제 이력서 경험 기술처럼 역할과 기여를 자연스럽게 1~2문장으로 보완해서 작성해 주세요.
- 기술 및 역량 항목도 간단 설명을 추가해 주세요.

[인적 사항]
- 이름: {이름}
- 생년월일: {생년월일}
- 이메일: {이메일}
- 연락처: {연락처}
- 주소: {주소}

[학력사항]
- 학교: {학교}
- 전공: {전공}
- 기간: {학력기간}
- 학점: {학점}

[경력사항]
{경력사항}

[기술 및 활동]
- 기술 및 역량: {기술역량}
- 자격증: {자격증}
- 기타 활동/수상내역: {기타활동}

포맷:
- 자기소개
- 학력사항
- 경력사항
- 기술 및 역량
- 기타 활동 및 수상내역
"""
    return prompt.strip()


def build_cover_letter_prompt(
    style,
    학교, 전공, 학력기간, 학점,
    경력사항, 기술역량, 자격증, 기타활동,
    지원회사, 지원직무, 채용공고, 기존이력서
):
    prompt = f"""
아래 정보를 바탕으로 한국어 자기소개서를 작성해 주세요.

✅ 목적:
- 지원 회사와 직무에 맞춘 실제 자기소개서 예문 생성
- 총 4개의 항목으로 구성
- 각 항목은 약 800자 분량으로 자세히 작성
- 자연스러운 문장 흐름 유지

✅ 구성 항목:
1. 성장과정
2. 성격의 장단점
3. 본인 역량 및 경력 사항
4. 지원동기 및 입사 후 포부

✅ 작성 조건:
- 각 항목 제목 포함
- 각 항목을 별도의 문단으로 작성
- 지원 회사와 직무의 요구사항(JD) 반영
- 내 기존 이력서 내용에서 중요한 경험/스킬 강조
- 내 기존 이력서 내용에서 중요한 경험/스킬을 반드시 반영
- 내 기존 이력서 내용이 JD와 어떻게 맞는지 연결
- JD 요구사항과 기존 이력서 내용을 분석해 강조
- 너무 반복적이지 않게 자연스럽게 서술
- 문장 연결 매끄럽게
- {style} 문체로 작성

[학력사항]
- 학교: {학교}
- 전공: {전공}
- 기간: {학력기간}
- 학점: {학점}

[경력사항]
{경력사항}

[기술 및 활동]
- 기술 및 역량: {기술역량}
- 자격증: {자격증}
- 기타 활동/수상내역: {기타활동}

[지원 정보]
- 지원 회사명: {지원회사}
- 지원 직무명: {지원직무}
- 채용 공고 내용(JD):
{채용공고}

[내 기존 이력서 내용(첨부 분석)]
{기존이력서}
"""
    return prompt.strip()




# def generate_resume_text(prompt):
#     response = openai_client.chat.completions.create(
#         model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#         messages=[
#             {"role": "system", "content": "당신은 한국어 이력서 작성 전문가입니다."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content


# def generate_cover_letter_text(prompt):
    
#     response = client.chat.completions.create(
#         model= "google/gemma-3-27b-it:free",
#         messages=[
#             {"role": "system", "content": "당신은 한국어 자기소개 작성 전문가입니다."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content

# model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        # model = "deepseek/deepseek-chat-v3-0324:free",

#  ✅ 입력 정보:

 # [인적 사항]
 # - 이름: {이름}
 # - 생년월일: {생년월일}
def generate_text(prompt, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "당신은 한국어 이력서 작성 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
def generate_text_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text