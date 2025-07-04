# Streamlit UI 
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_kogpt():
    tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt")
    model = AutoModelForCausalLM.from_pretrained("kakaobrain/kogpt")
    return tokenizer, model

tokenizer, model = load_kogpt()

# ------------------------------
# app.py
# KoGPT 자기소개서 / 이력서 Generator
# ------------------------------

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------
# Load model & tokenizer
# ------------------------------
@st.cache_resource
def load_kogpt():
    tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt")
    model = AutoModelForCausalLM.from_pretrained("kakaobrain/kogpt")
    return tokenizer, model

tokenizer, model = load_kogpt()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🇰🇷 KoGPT 자기소개서 / 이력서 생성기")
st.markdown("""
한국어 입력 정보를 바탕으로 **자기소개서**나 **이력서**를 AI가 자동으로 작성합니다.  
아래에 정보를 입력하고 버튼을 눌러 보세요!
""")

st.sidebar.title("🛠️ 옵션")
max_tokens = st.sidebar.slider("생성 최대 길이", 100, 1024, 512, step=50)
temperature = st.sidebar.slider("창의성 (Temperature)", 0.5, 1.5, 0.8, step=0.1)

# ------------------------------
# User Inputs
# ------------------------------
doc_type = st.selectbox("문서 종류 선택:", ["자기소개서", "이력서"])
name = st.text_input("이름:")
age = st.text_input("나이:")
skills = st.text_area("보유 기술/역량:")
experience = st.text_area("경력 사항:")
goal = st.text_area("목표/지원 동기:")

# ------------------------------
# Prompt Builder
# ------------------------------
def build_prompt(doc_type, name, age, skills, experience, goal):
    prompt = f"""
아래 정보를 바탕으로 {doc_type}를 한국어로 자연스럽게 작성해 주세요.

- 이름: {name}
- 나이: {age}
- 보유 기술/역량: {skills}
- 경력 사항: {experience}
- 목표/지원 동기: {goal}

작성된 {doc_type}:
"""
    return prompt.strip()

# ------------------------------
# Generation Button
# ------------------------------
if st.button("✅ 생성하기"):
    if not name or not age:
        st.warning("⚠️ 이름과 나이를 입력해 주세요!")
    else:
        with st.spinner("KoGPT가 문서를 생성 중입니다... 잠시만 기다려 주세요."):
            user_prompt = build_prompt(doc_type, name, age, skills, experience, goal)
            
            inputs = tokenizer.encode(user_prompt, return_tensors="pt")
            
            output = model.generate(
                inputs,
                max_length=max_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            
            result_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Prompt까지 포함된 결과가 나오므로 잘라서 깔끔하게
            result_only = result_text[len(user_prompt):].strip()
            
            st.success(f"🎉 {doc_type} 생성 완료!")
            st.text_area("✍️ 생성된 문서", value=result_only, height=300)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("🛠️ Made with KakaoBrain KoGPT + Streamlit")
