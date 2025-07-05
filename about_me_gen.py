from openai import OpenAI

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="af95f7404466444675d930ead2e9b67a8bfe3b3e2d8d0d16501a74107fa512d4"
)

def build_resume_prompt(
    style, 이름, 생년월일, 이메일, 연락처, 주소,
    학교, 전공, 학력기간, 학점,
    경력사항, 기술역량, 자격증, 기타활동
):
    prompt = f"""
다음 정보를 바탕으로 한국어 이력서를 {style} 작성해 주세요:

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

def generate_resume(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": "당신은 한국어 이력서 작성 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
