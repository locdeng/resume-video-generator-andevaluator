<<<<<<< HEAD
#이력서 및 자소서 생성 -> KoGPT API를 이용한 이력서 및 자소서 생성
=======
from openai import OpenAI
from dotenv import load_dotenv
import os
import google.generativeai as genai
import streamlit as st
import fitz


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") 
genai.configure(api_key=api_key)
# openai_client = OpenAI(
#     base_url="https://api.together.xyz/v1",
#     api_key= os.getenv("OPENAI_API_KEY")
# )
# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key= os.getenv("OPENROUTER_API_KEY"),
# )

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


# -------------------------------
# 이력서 생성 페이지
# -------------------------------
def run_resume_generator():
    st.markdown("""
        <style>
        
         @font-face {
            font-family: 'SB_B';
            src: url('assets/fonts/SF.ttf') format('truetype');
        }
        
                /* Toàn bộ trang (nền đen) */
        html, body {
            background-color: #f0e8db !important;
            font-family: 'SF',sans-serif;
        }

        /* Nền vùng nội dung */
        [data-testid="stAppViewContainer"] {
            background-color: #f0e8db !important;
        }

        /* Nền container chính */
        [data-testid="stAppViewBlockContainer"] {
            background-color: #f0e8db !important;
            padding: 0rem 1rem; /* giảm padding nếu muốn */
            max-width: 100% !important;  /* full width */
        }

        /* Optional: Sidebar nếu bạn muốn cũng nền đen */
        [data-testid="stSidebar"] {
            background-color: #77C9D4 !important;
        }
        .intro-title {
            font-size: 48px;
            font-weight: 800;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: 30px;
        }
        .intro-sub {
            font-size: 18px;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .feature-box {
            background: #F2EFE7 ;
            padding: 30px;
            border-radius: 15px;
            margin: 10px 20px;
            color: #2b2b2b;
            border: 2px solid white;
            font-family: 'SF',sans-serif;
            text-align: center;
        }
        .feature-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
            font-family: 'SF',sans-serif;
            color: #2b2b2b;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            background-color: #F2EFE7;
            border: 2px solid white;
            color: #2b2b2b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="intro-title">AI 이력서 생성기</div>', unsafe_allow_html=True)
    

    tabs = st.tabs([
        "1️⃣ 인적 사항",
        "2️⃣ 학력사항",
        "3️⃣ 경력사항",
        "4️⃣ 기술 및 활동",
        "5️⃣ 이력서 생성"
    ])

    with tabs[0]:
        이름 = st.text_input("이름")
        생년월일 = st.text_input("생년월일 (YYYY-MM-DD)")
        이메일 = st.text_input("이메일")
        연락처 = st.text_input("연락처")
        주소 = st.text_input("주소")

    with tabs[1]:
        학교 = st.text_input("학교명")
        전공 = st.text_input("전공")
        학력기간 = st.text_input("기간 (YYYY/MM - YYYY/MM)")
        학점 = st.text_input("학점")

    with tabs[2]:
        경력사항 = st.text_area("경력사항")

    with tabs[3]:
        기술역량 = st.text_area("기술 및 역량")
        자격증 = st.text_area("자격증")
        기타활동 = st.text_area("기타 활동/수상내역")

    with tabs[4]:
        style = st.selectbox("작성 스타일", ["간결하게", "감성적으로", "전문적으로"])

        if st.button("AI 이력서 생성하기"):
            with st.spinner("AI가 이력서를 생성 중입니다..."):
                prompt = build_resume_prompt(
                    style, 이름, 생년월일, 이메일, 연락처, 주소,
                    학교, 전공, 학력기간, 학점,
                    경력사항, 기술역량, 자격증, 기타활동
                )
                # result = about_me_gen.generate_text(prompt,"google/gemini-2.0-flash-exp:free")
                result = generate_text_gemini(prompt)
                st.success("이력서 생성 완료!")
                st.markdown(result)

# -------------------------------
# 자소서 생성 페이지 (Placeholder)
# -------------------------------
def extract_text_from_file(uploaded_file):
    if uploaded_file is None or uploaded_file.size == 0:
        return ""

    uploaded_file.seek(0)
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    elif uploaded_file.name.endswith(".pdf"):
        try:
            uploaded_file.seek(0)
            text = ""
            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"⚠️ PDF 읽는 중 오류가 발생했습니다: {e}")
            return ""

    else:
        return ""
    
def run_cover_letter():
    
    st.markdown("""
        <style>
        
         @font-face {
            font-family: 'SB_B';
            src: url('assets/fonts/SF.ttf') format('truetype');
        }
        
                /* Toàn bộ trang (nền đen) */
        html, body {
            background-color: #f0e8db !important;
            font-family: 'SF',sans-serif;
        }

        /* Nền vùng nội dung */
        [data-testid="stAppViewContainer"] {
            background-color: #f0e8db !important;
        }

        /* Nền container chính */
        [data-testid="stAppViewBlockContainer"] {
            background-color: #f0e8db !important;
            padding: 0rem 1rem; /* giảm padding nếu muốn */
            max-width: 100% !important;  /* full width */
        }

        /* Optional: Sidebar nếu bạn muốn cũng nền đen */
        [data-testid="stSidebar"] {
            background-color: #77C9D4 !important;
        }
        .intro-title {
            font-size: 48px;
            font-weight: 800;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: 30px;
        }
        .intro-sub {
            font-size: 18px;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .feature-box {
            background: #F2EFE7 ;
            padding: 30px;
            border-radius: 15px;
            margin: 10px 20px;
            color: #2b2b2b;
            border: 2px solid white;
            font-family: 'SF',sans-serif;
            text-align: center;
        }
        .feature-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
            font-family: 'SF',sans-serif;
            color: #2b2b2b;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            background-color: #F2EFE7;
            border: 2px solid white;
            color: #2b2b2b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="intro-title">AI 자기소개서 생성 (맞춤형)</div>', unsafe_allow_html=True)

    
    st.markdown("""
    이 앱은 **텍스트 기반 PDF**만 지원합니다.  
    - 워드 등에서 작성한 PDF는 OK!  
    - 스캔 이미지 PDF는 지원하지 않습니다.
    """)

    with st.form("cover_letter_form"):
        # st.subheader("기본 정보")
        # 이름 = st.text_input("이름")
        # 생년월일 = st.text_input("생년월일 (YYYY-MM-DD)")

        st.subheader("지원 정보")
        지원회사 = st.text_input("지원 회사명")
        지원직무 = st.text_input("지원 직무명")
        채용공고 = st.text_area("채용 공고 내용 (JD 붙여넣기)")

        st.subheader("내 기존 이력서 업로드(옵션)")
        uploaded_file = st.file_uploader("이력서 파일 업로드 (.pdf, .txt 가능)", type=["pdf", "txt"])
        기존이력서 = extract_text_from_file(uploaded_file)

        with st.expander("추가 정보 (옵션)"):
            학교 = st.text_input("학교명")
            전공 = st.text_input("전공")
            학력기간 = st.text_input("기간 (YYYY/MM - YYYY/MM)")
            학점 = st.text_input("학점")

            경력사항 = st.text_area("경력사항")
            기술역량 = st.text_area("기술 및 역량")
            자격증 = st.text_area("자격증")
            기타활동 = st.text_area("기타 활동/수상내역")


        style = st.selectbox("작성 스타일", ["간결하게", "감성적으로", "전문적으로"])

        
        # show_debug = st.checkbox("🪄 Debug 모드 (생성 Prompt 보기)")

        submitted = st.form_submit_button("AI 자기소개서 생성하기")

    if submitted:
        with st.spinner("AI가 자기소개서를 생성 중입니다..."):
            기존이력서 = extract_text_from_file(uploaded_file)
            
            # st.subheader("[STEP 1] 업로드한 이력서에서 읽은 텍스트")
            if not 기존이력서.strip():
                st.warning("업로드한 파일에서 텍스트를 읽지 못했습니다. PDF가 이미지만 있거나 비어있을 수 있습니다.")

            prompt = build_cover_letter_prompt(
                style,
                학교, 전공, 학력기간, 학점,
                경력사항, 기술역량, 자격증, 기타활동,
                지원회사, 지원직무, 채용공고, 기존이력서
            )
            # if show_debug:
            #     st.subheader("✅ 생성된 Prompt (LLM Input)")
            #     st.code(prompt, language="markdown")
            # result = about_me_gen.generate_text(prompt, "google/gemma-3-27b-it:free")
            result = generate_text_gemini(prompt)
            st.success("자기소개서 생성 완료!")
            st.markdown(result)
>>>>>>> 16
