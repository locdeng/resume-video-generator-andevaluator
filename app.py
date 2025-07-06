import streamlit as st
from about_me_evaluate import resume_predict_tab
# from other_tab import other_tab_function

tab = st.sidebar.selectbox("기능", ["자소사 평가", "기타 기능"])

if tab == "자소서 평가":
    resume_predict_tab()
elif tab == "기타 기능":
    st.write("🚧 개발 중입니다.")
