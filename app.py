from openai import OpenAI
import streamlit as st
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="af95f7404466444675d930ead2e9b67a8bfe3b3e2d8d0d16501a74107fa512d4"
)

st.title("🇰🇷 AI 이력서 생성기 (Wrtn 스타일)")

tabs = st.tabs(["1️⃣ 인적 사항", "2️⃣ 학력사항", "3️⃣ 경력사항", "4️⃣ 기술 및 활동", "5️⃣ 이력서 생성"])

with tabs[0]:
    st.header("1️⃣ 인적 사항 입력")
    이름 = st.text_input("이름")
    생년월일 = st.text_input("생년월일 (YYYY-MM-DD)")
    이메일 = st.text_input("이메일")
    연락처 = st.text_input("연락처")
    주소 = st.text_input("주소")

with tabs[1]:
    st.header("2️⃣ 학력사항 입력")
    학교 = st.text_input("학교명")
    전공 = st.text_input("전공")
    학력기간 = st.text_input("기간 (YYYY/MM - YYYY/MM)")
    학점 = st.text_input("학점")

with tabs[2]:
    st.header("3️⃣ 경력사항 입력")
    경력사항 = st.text_area("경력사항")

with tabs[3]:
    st.header("4️⃣ 기술 및 활동 입력")
    기술역량 = st.text_area("기술 및 역량")
    자격증 = st.text_area("자격증")
    기타활동 = st.text_area("기타 활동/수상내역")

with tabs[4]:
    st.header("5️⃣ 이력서 생성하기")
    style = st.selectbox("작성 스타일", ["간결하게", "감성적으로", "전문적으로"])

    if st.button("AI 이력서 생성하기"):
        with st.spinner("AI가 이력서를 생성 중입니다..."):
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


            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Ví dụ model together.ai mà bạn kiểm tra được quyền
                messages=[
                    {"role": "system", "content": "당신은 한국어 이력서 작성 전문가입니다."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.success("✅ 이력서 생성 완료!")
            st.markdown(response.choices[0].message.content)