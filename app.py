import streamlit as st
import about_me_gen 
from streamlit_option_menu import option_menu
import fitz #PyMuPDF
# from PIL import Image
import pytesseract
# import pdf2image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------------------------------
# 홈 페이지
# -------------------------------
def run_home():
    st.title("🏠 AI 이력서 생성기 홈")
    st.markdown("""
    안녕하세요!  
    이 웹앱은 AI를 활용해 한국어 이력서를 자동으로 생성해 줍니다.  
    
    좌측 사이드바에서 **이력서 생성하기** 또는 **자소서 생성하기** 메뉴를 선택해 정보를 입력하고 생성해 보세요!
    """)



# -------------------------------
# 이력서 생성 페이지
# -------------------------------
import streamlit as st
import about_me_gen

# -------------------------------
# 이력서 생성 페이지 (with session_state tab navigation)
# -------------------------------
def run_resume_generator():
    st.title("📄 AI 이력서 생성기")

    # Initialize step and action state only once
    if "resume_step" not in st.session_state:
        st.session_state.resume_step = 0
    if "_action" not in st.session_state:
        st.session_state._action = ""

    steps = [
        "1️⃣ 인적 사항",
        "2️⃣ 학력사항",
        "3️⃣ 경력사항",
        "4️⃣ 기술 및 활동",
        "5️⃣ 이력서 생성"
    ]

    # Render visual tab headers
    cols = st.columns(len(steps))
    for i, label in enumerate(steps):
        with cols[i]:
            if i == st.session_state.resume_step:
                st.markdown(f"<div style='padding:8px;background:black;color:white;text-align:center;border-radius:5px'>{label}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:8px;background:white;color:black;text-align:center;border-radius:5px;border:1px solid black'>{label}</div>", unsafe_allow_html=True)

    st.write("")

    # Tab control content
    def set_action(act):
        st.session_state._action = act

    action = st.session_state._action
    st.session_state._action = ""

    if st.session_state.resume_step == 0:
        st.subheader("1️⃣ 인적 사항")
        이름 = st.text_input("이름", key="이름")
        생년월일 = st.text_input("생년월일 (YYYY-MM-DD)", key="생년월일")
        이메일 = st.text_input("이메일", key="이메일")
        연락처 = st.text_input("연락처", key="연락처")
        주소 = st.text_input("주소", key="주소")

        if st.button("➡️ 다음"):
            st.session_state.data = {
                "이름": 이름, "생년월일": 생년월일, "이메일": 이메일,
                "연락처": 연락처, "주소": 주소
            }
            st.session_state.resume_step += 1

    elif st.session_state.resume_step == 1:
        st.subheader("2️⃣ 학력사항")
        학교 = st.text_input("학교명", key="학교")
        전공 = st.text_input("전공", key="전공")
        학력기간 = st.text_input("기간 (YYYY/MM - YYYY/MM)", key="학력기간")
        학점 = st.text_input("학점", key="학점")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ 이전"):
                st.session_state.resume_step -= 1
        with col2:
            if st.button("➡️ 다음"):
                st.session_state.data.update({
                    "학교": 학교, "전공": 전공,
                    "학력기간": 학력기간, "학점": 학점
                })
                st.session_state.resume_step += 1

    elif st.session_state.resume_step == 2:
        st.subheader("3️⃣ 경력사항")
        경력사항 = st.text_area("경력사항", key="경력사항")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ 이전"):
                st.session_state.resume_step -= 1
        with col2:
            if st.button("➡️ 다음"):
                st.session_state.data.update({"경력사항": 경력사항})
                st.session_state.resume_step += 1

    elif st.session_state.resume_step == 3:
        st.subheader("4️⃣ 기술 및 활동")
        기술역량 = st.text_area("기술 및 역량", key="기술역량")
        자격증 = st.text_area("자격증", key="자격증")
        기타활동 = st.text_area("기타 활동/수상내역", key="기타활동")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ 이전"):
                st.session_state.resume_step -= 1
        with col2:
            if st.button("➡️ 다음"):
                st.session_state.data.update({
                    "기술역량": 기술역량,
                    "자격증": 자격증,
                    "기타활동": 기타활동
                })
                st.session_state.resume_step += 1

    elif st.session_state.resume_step == 4:
        st.subheader("5️⃣ 이력서 생성")
        style = st.selectbox("작성 스타일", ["간결하게", "감성적으로", "전문적으로"], key="작성스타일")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ 이전"):
                st.session_state.resume_step -= 1
        with col2:
            if st.button("✅ AI 이력서 생성하기"):
                with st.spinner("AI가 이력서를 생성 중입니다..."):
                    d = st.session_state.data
                    prompt = about_me_gen.build_resume_prompt(
                        style,
                        d["이름"], d["생년월일"], d["이메일"], d["연락처"], d["주소"],
                        d["학교"], d["전공"], d["학력기간"], d["학점"],
                        d["경력사항"], d["기술역량"], d["자격증"], d["기타활동"]
                    )
                    result = about_me_gen.generate_text_gemini(prompt)
                    st.success("✅ 이력서 생성 완료!")
                    st.markdown(result)

# -------------------------------
# MAIN APP
# -------------------------------
def main():
    with st.sidebar:
        selected = st.selectbox("📌 메뉴 선택", ["홈", "이력서 생성하기"])

    if selected == "홈":
        st.title("🏠 홈 페이지")
        st.write("AI 기반 이력서/자소서 생성기입니다.")
    elif selected == "이력서 생성하기":
        run_resume_generator()

if __name__ == "__main__":
    main()


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
    st.title("📝 AI 자기소개서 생성기 (맞춤형)")
    st.markdown("""
    ✅ 이 앱은 **텍스트 기반 PDF**만 지원합니다.  
    - 워드 등에서 작성한 PDF는 OK!  
    - 스캔 이미지 PDF는 지원하지 않습니다.
    """)

    with st.form("cover_letter_form"):
        # st.subheader("✅ 기본 정보")
        # 이름 = st.text_input("이름")
        # 생년월일 = st.text_input("생년월일 (YYYY-MM-DD)")

        st.subheader("✅ 지원 정보")
        지원회사 = st.text_input("지원 회사명")
        지원직무 = st.text_input("지원 직무명")
        채용공고 = st.text_area("채용 공고 내용 (JD 붙여넣기)")

        st.subheader("✅ 내 기존 이력서 업로드(옵션)")
        uploaded_file = st.file_uploader("이력서 파일 업로드 (.pdf, .txt 가능)", type=["pdf", "txt"])
        기존이력서 = extract_text_from_file(uploaded_file)

        with st.expander("✅ 추가 정보 (옵션)"):
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

        submitted = st.form_submit_button("✅ AI 자기소개서 생성하기")

    if submitted:
        with st.spinner("AI가 자기소개서를 생성 중입니다..."):
            기존이력서 = extract_text_from_file(uploaded_file)
            
            # st.subheader("✅ ✅ 📌 [STEP 1] 업로드한 이력서에서 읽은 텍스트")
            if not 기존이력서.strip():
                st.warning("⚠️ 업로드한 파일에서 텍스트를 읽지 못했습니다. PDF가 이미지만 있거나 비어있을 수 있습니다.")

            prompt = about_me_gen.build_cover_letter_prompt(
                style,
                학교, 전공, 학력기간, 학점,
                경력사항, 기술역량, 자격증, 기타활동,
                지원회사, 지원직무, 채용공고, 기존이력서
            )
            # if show_debug:
            #     st.subheader("✅ 생성된 Prompt (LLM Input)")
            #     st.code(prompt, language="markdown")
            # result = about_me_gen.generate_text(prompt, "google/gemma-3-27b-it:free")
            result = about_me_gen.generate_text_gemini(prompt)
            st.success("✅ 자기소개서 생성 완료!")
            st.markdown(result)
            



# -------------------------------
# MAIN APP
# -------------------------------
def main():
    with st.sidebar:
        selected = option_menu(
            "📌 메뉴 선택",
            ["홈", "이력서 생성하기", "자소서 생성하기"],
            icons=["house", "file-earmark-person", "pencil-square"],
            menu_icon="cast",
            default_index=0
        )

    if selected == "홈":
        run_home()

    elif selected == "이력서 생성하기":
        run_resume_generator()

    elif selected == "자소서 생성하기":
        run_cover_letter()

    else:
        st.error("⚠️ 잘못된 메뉴입니다.")


# -------------------------------
if __name__ == "__main__":
    main()
def run_resume_generator():
    st.title("📄 AI 이력서 생성기")

    if "resume_step" not in st.session_state:
        st.session_state.resume_step = 1

    if st.session_state.resume_step == 1:
        st.subheader("1️⃣ 인적 사항")
        st.session_state.이름 = st.text_input("이름", key="이름")
        st.session_state.생년월일 = st.text_input("생년월일 (YYYY-MM-DD)", key="생년월일")
        st.session_state.이메일 = st.text_input("이메일", key="이메일")
        st.session_state.연락처 = st.text_input("연락처", key="연락처")
        st.session_state.주소 = st.text_input("주소", key="주소")

        if st.button("➡️ 다음"):
            st.session_state.resume_step += 1

    elif st.session_state.resume_step == 2:
        st.subheader("2️⃣ 학력사항")
        st.session_state.학교 = st.text_input("학교명", key="학교")
        st.session_state.전공 = st.text_input("전공", key="전공")
        st.session_state.학력기간 = st.text_input("기간 (YYYY/MM - YYYY/MM)", key="학력기간")
        st.session_state.학점 = st.text_input("학점", key="학점")

        if st.button("⬅️ 이전"):
            st.session_state.resume_step -= 1
        if st.button("➡️ 다음"):
            st.session_state.resume_step += 1

    elif st.session_state.resume_step == 3:
        st.subheader("3️⃣ 경력사항")
        st.session_state.경력사항 = st.text_area("경력사항", key="경력사항")

        if st.button("⬅️ 이전"):
            st.session_state.resume_step -= 1
        if st.button("➡️ 다음"):
            st.session_state.resume_step += 1

    elif st.session_state.resume_step == 4:
        st.subheader("4️⃣ 기술 및 활동")
        st.session_state.기술역량 = st.text_area("기술 및 역량", key="기술역량")
        st.session_state.자격증 = st.text_area("자격증", key="자격증")
        st.session_state.기타활동 = st.text_area("기타 활동/수상내역", key="기타활동")

        if st.button("⬅️ 이전"):
            st.session_state.resume_step -= 1
        if st.button("➡️ 다음"):
            st.session_state.resume_step += 1

    elif st.session_state.resume_step == 5:
        st.subheader("5️⃣ 이력서 생성")
        style = st.selectbox("작성 스타일", ["간결하게", "감성적으로", "전문적으로"])

        if st.button("⬅️ 이전"):
            st.session_state.resume_step -= 1

        if st.button("✅ AI 이력서 생성하기"):
            with st.spinner("AI가 이력서를 생성 중입니다..."):
                prompt = about_me_gen.build_resume_prompt(
                    style,
                    st.session_state.이름, st.session_state.생년월일,
                    st.session_state.이메일, st.session_state.연락처, st.session_state.주소,
                    st.session_state.학교, st.session_state.전공,
                    st.session_state.학력기간, st.session_state.학점,
                    st.session_state.경력사항,
                    st.session_state.기술역량, st.session_state.자격증, st.session_state.기타활동
                )
                result = about_me_gen.generate_text_gemini(prompt)
                st.success("✅ 이력서 생성 완료!")
                st.markdown(result)




