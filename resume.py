# evaluator.py
import re
import datetime
import pandas as pd
import streamlit as st
import fitz  # pip install PyMuPDF
import docx2txt

from config import NATIONAL_CERTIFICATES, PRIVATE_CERTIFICATES, FAKE_OR_UNCERTIFIED, \
    EXTERNAL_EDUCATION_KEYWORDS, INTERNAL_EDUCATION_KEYWORDS, SELF_DEVELOPMENT_KEYWORDS

def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

def extract_recent_years(text):
    current_year = datetime.datetime.now().year
    years = re.findall(r'(\d{4})\s*년', text)
    return any(current_year - int(y) <= 3 for y in years if y.isdigit())

def contains_any(text, keyword_list):
    return any(keyword in text for keyword in keyword_list)

def grade_education(text):
    text_lower = text.lower()
    if "phd" in text_lower or "박사" in text_lower:
        degree_score = "A"
    elif "master" in text_lower or "석사" in text_lower:
        degree_score = "B"
    elif "bachelor" in text_lower or "학사" in text_lower:
        degree_score = "C"
    elif "associate" in text_lower or "전문대" in text_lower:
        degree_score = "D"
    else:
        degree_score = "E"

    gpa_match = re.search(r"(\d\.\d{1,2})\s*/\s*4\.0", text)
    if gpa_match:
        gpa = float(gpa_match.group(1))
        if gpa >= 4.0:
            gpa_score = "A"
        elif gpa >= 3.5:
            gpa_score = "B"
        elif gpa >= 3.0:
            gpa_score = "C"
        elif gpa >= 2.5:
            gpa_score = "D"
        else:
            gpa_score = "E"
    else:
        gpa_score = "E"

    return {"최종 학위 수준": degree_score, "학점": gpa_score}

def grade_experience(text):
    if re.search(r"(10\s*years|10년)", text):
        year_score = "A"
    elif re.search(r"(6\s*[-~]?\s*9\s*years|6~9년)", text):
        year_score = "B"
    elif re.search(r"(3\s*[-~]?\s*5\s*years|3~5년)", text):
        year_score = "C"
    elif re.search(r"(1\s*[-~]?\s*2\s*years|1~2년)", text):
        year_score = "D"
    else:
        year_score = "E"

    if "award" in text.lower() or "성과" in text or "수상" in text:
        result_score = "A"
    elif "led" in text.lower() or "리더" in text or "팀장" in text:
        result_score = "B"
    elif "responsible" in text.lower() or "담당" in text:
        result_score = "C"
    else:
        result_score = "E"

    return {"총 경력 연수": year_score, "성과 및 기여도": result_score}

def grade_certification(text):
    score_cert = "E"
    score_level = "E"
    score_recent = "E"

    if contains_any(text, NATIONAL_CERTIFICATES):
        score_cert = "A"
        score_level = "A"
    elif contains_any(text, PRIVATE_CERTIFICATES):
        score_cert = "C"
        score_level = "C"
    elif contains_any(text, FAKE_OR_UNCERTIFIED):
        score_cert = "D"
        score_level = "D"

    if extract_recent_years(text):
        score_recent = "A"
    elif re.search(r"(201[8-9]|2020|2021)", text):
        score_recent = "B"
    else:
        score_recent = "E"

    return {
        "직무 관련 자격증": score_cert,
        "자격증 수준": score_level,
        "자격증 최신성": score_recent
    }

def grade_education_history(text):
    score_pro = "E"
    score_internal = "E"
    score_self = "E"

    if contains_any(text, EXTERNAL_EDUCATION_KEYWORDS):
        score_pro = "A" if extract_recent_years(text) else "B"

    if contains_any(text, INTERNAL_EDUCATION_KEYWORDS):
        if "발표" in text:
            score_internal = "A"
        elif "정기" in text:
            score_internal = "B"
        elif "교육" in text:
            score_internal = "C"

    if contains_any(text, SELF_DEVELOPMENT_KEYWORDS):
        if re.search(r"(지속|꾸준|계속|계획적)", text):
            score_self = "A"
        elif re.search(r"(기록|블로그|이력)", text):
            score_self = "B"
        elif re.search(r"(시도|도전)", text):
            score_self = "C"

    return {
        "전문교육 수료 여부": score_pro,
        "기업 내 교육 참여": score_internal,
        "자기개발 노력": score_self
    }

def evaluate_resume(text):
    edu = grade_education(text)
    exp = grade_experience(text)
    cert = grade_certification(text)
    edu_hist = grade_education_history(text)

    result_df = pd.DataFrame([
        {"분류": "학력", "세부 항목": k, "등급": v} for k, v in edu.items()
    ] + [
        {"분류": "경력", "세부 항목": k, "등급": v} for k, v in exp.items()
    ] + [
        {"분류": "자격증", "세부 항목": k, "등급": v} for k, v in cert.items()
    ] + [
        {"분류": "교육이력", "세부 항목": k, "등급": v} for k, v in edu_hist.items()
    ])

    return result_df

def color_grade(val):
    colors = {
        "A": "background-color: #c6f6d5",
        "B": "background-color: #fefcbf",
        "C": "background-color: #fbd38d",
        "D": "background-color: #feb2b2",
        "E": "background-color: #fc8181"
    }
    return colors.get(val, "")

def show_evaluation_ui():
    st.header("📄 이력서 평가 시스템")
    uploaded_file = st.file_uploader("이력서를 업로드하세요 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file:
        text = extract_text(uploaded_file)
        st.success("파일 업로드 및 내용 추출 완료!")
        result_df = evaluate_resume(text)

        styled_df = result_df.style.applymap(color_grade, subset=["등급"])
        st.subheader("🔍 평가 결과")
        st.dataframe(styled_df, use_container_width=True)
        
        with st.expander("📄 Mở xem file CSV"):
    
            csv_path = 'resume_standart.csv'  
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df)
            except FileNotFoundError:
                st.error(f"Not found: {csv_path}")

        csv = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("결과 CSV 다운로드", data=csv, file_name="resume_evaluation.csv", mime="text/csv")
