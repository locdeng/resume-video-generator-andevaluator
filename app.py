import streamlit as st
import about_me_gen
from streamlit_option_menu import option_menu


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
def run_resume_generator():
    st.title("📄 AI 이력서 생성기")

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

        if st.button("✅ AI 이력서 생성하기"):
            with st.spinner("AI가 이력서를 생성 중입니다..."):
                prompt = about_me_gen.build_resume_prompt(
                    style, 이름, 생년월일, 이메일, 연락처, 주소,
                    학교, 전공, 학력기간, 학점,
                    경력사항, 기술역량, 자격증, 기타활동
                )
                result = about_me_gen.generate_resume(prompt)
                st.success("✅ 이력서 생성 완료!")
                st.markdown(result)

# -------------------------------
# 자소서 생성 페이지 (Placeholder)
# -------------------------------
def run_cover_letter():
    st.title("📝 AI 자소서 생성기")
    st.markdown("""
    🚧 준비 중입니다!  
    곧 업데이트 될 예정입니다.
    """)


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
