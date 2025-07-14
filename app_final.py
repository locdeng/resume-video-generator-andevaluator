import streamlit as st
from streamlit_option_menu import option_menu
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from about_me_evaluate import resume_predict_tab
from resume import show_evaluation_ui
from about_me_gen import run_cover_letter, run_resume_generator
from video_pose_analyze import run_realtime_analysis, run_uploaded_video_analysis

# -------------------- 1. Menu state setup --------------------
if "menu" not in st.session_state:
    st.session_state["menu"] = "Home"

# -------------------- 2. Sidebar --------------------
# with st.sidebar:
#     selected = option_menu(
#         menu_title="기능 선택",
#         options=["Home", "자소서 평가", "동영상 분석"],
#         icons=["house", "file-earmark-text", "camera-video"],
#         default_index=["Home", "자소서 평가", "동영상 분석"].index(st.session_state["menu"])
#     )
# st.session_state["menu"] = selected

# -------------------- 3. Page: Home --------------------
if st.session_state["menu"] == "Home":
    st.markdown("""
        <style>
        .intro-title {
            font-size: 48px;
            font-weight: 800;
            color: #00ffc8;
            text-align: center;
            margin-top: 30px;
        }
        .intro-sub {
            font-size: 18px;
            color: #aaa;
            text-align: center;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .feature-box {
            background: #2b2b3c;
            padding: 30px;
            border-radius: 15px;
            margin: 10px 20px;
            color: white;
            text-align: center;
        }
        .feature-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #ff4b4b;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            background-color: transparent;
            border: 2px solid #ff4b4b;
            color: #ff4b4b;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="intro-title">AI 기반 분석 도구</div>', unsafe_allow_html=True)
    st.markdown('<div class="intro-sub">자소서 평가 및 얼굴 감정/자세 실시간 분석</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title"> 이력서 및 자소서 자동 생성</div>
            AI를 활용해 한국어 이력서와 자소서를 자동으로 생성해 줍니다.  .
        </div>
        """, unsafe_allow_html=True)

        if st.button("👉 자소서 및 이력서 자동 생성로 이동"):
            st.session_state["menu"] = "이력서 및 자소서 자동 생성"
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title"> 이력서 및 자소서 평가</div>
            자기소개서와 이력서를 항목별로 평가하고 피드백을 제공합니다.
        </div>
        """, unsafe_allow_html=True)

        if st.button("👉 이력서 및 자소서 평가로 이동"):
            st.session_state["menu"] = "이력서 및 자소서 평가"
            st.rerun()
            
    with col3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title"> 동영상 분석</div>
            웹캠 또는 비디오 파일에서 감정 및 포즈를 분석합니다.
        </div>
        """, unsafe_allow_html=True)

        if st.button("👉 동영상 분석으로 이동"):
            st.session_state["menu"] = "동영상 분석"
            st.rerun()

# -------------------- 4. Page: Resume Evaluation --------------------
elif st.session_state["menu"] == "이력서 및 자소서 자동 생성":
    st.title("이력서 및 자소서 자동 생성")
    
    tab1, tab2 = st.tabs(["이력서 자동 생성", "자소서 자동 생성 "])

    with tab1:
        run_resume_generator()

    with tab2:
        run_cover_letter
        
    if st.button("⬅️ 홈으로 돌아가기"):
        st.session_state["menu"] = "Home"
        st.rerun()


elif st.session_state["menu"] == "이력서 및 자소서 평가":
    st.title("자소서 및 이력서 평가 시스템")
    
    tab1, tab2 = st.tabs(["✍️ 자소서 평가", "📄 이력서 평가"])

    with tab1:
        resume_predict_tab()

    with tab2:
        show_evaluation_ui()
        
    if st.button("⬅️ 홈으로 돌아가기"):
        st.session_state["menu"] = "Home"
        st.rerun()

# -------------------- 5. Page: Video Analysis --------------------
elif st.session_state["menu"] == "동영상 분석":
    st.title("실시간 감정 + 자세 분석")
    mode = st.radio("모드를 선택하세요:", ["실시간 웹캠", "비디오 파일 업로드"], horizontal=True)

    if mode == "실시간 웹캠":
        st.info("웹캠을 사용해 실시간 분석을 시작합니다. 아래에서 허용을 눌러주세요.")
        run_realtime_analysis()
    elif mode == '비디오 파일 업로드':
        st.info("비디오 파일을 업로드하여 분석합니다.")
        run_uploaded_video_analysis()
        
    if st.button("⬅️ 홈으로 돌아가기"):
        st.session_state["menu"] = "Home"
        st.rerun()


 
