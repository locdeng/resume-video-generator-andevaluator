import os
import gdown
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import streamlit as st

# Tải model nếu chưa có
def download_model():
    os.makedirs("saved_model", exist_ok=True)
    model_path = "saved_model/model.safetensors"
    if not os.path.exists(model_path):
        st.info("🔽 모델 파일을 Google Drive에서 다운로드 중입니다...")
        url = "https://drive.google.com/uc?id=1kAmiJNISGAPYtvfQXNg47nWNvC09pWVE"
        gdown.download(url, output=model_path, quiet=True)
    # else:
    #     st.success("✅ 모델 파일이 이미 존재합니다.")

download_model()

# Load model và tokenizer
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")
model.eval()

# Load label encoder
with open("saved_model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def score_to_grade(score):
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    return mapping.get(score, None)


# Dự đoán
def predict(text, subcat):
    prompt = f"{subcat}:\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_score = torch.argmax(probs, dim=1).item()
        pred_grade = score_to_grade(pred_score)
    return pred_grade, probs[0][pred_score].item()



# Tab UI
def resume_predict_tab():
    
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
    
    st.markdown('<div class="intro-title">자소서 평가 시스템</div>', unsafe_allow_html=True)
    
    
    categories = {'성장과정' : ['가치관', '가족/환경의 영향', '전환점/특별한 경험'], 
                  '성격의 장단점': ['정점 기술', '단점 기술 및 극복 노력', '대인관계 성향'], 
                  '학창시설': ['학업 태도', '활동 참여', '성취 및 도전 경험'], 
                  '지원동기': ['회사/직무 이해도', '지원 이유의 명확성', '직무 적합성 강조'] , 
                  '입사 후 포부': ['단기 목표', '장기 목표', '기여 방안'], 
                  '직무 경험': ['실무', '직무 관련 능력', '문제 해결력'] }
    
    user_inputs = {}
    for cat in categories.keys():
        text = st.text_area(f"✍️ {cat} 입력:", key=cat, height=150)
        user_inputs[cat] = text

    if st.button("🔍 평가 시작"):
        st.markdown("---")
        st.subheader("📊 평가 결과")
        for cat, text in user_inputs.items():
            if text.strip():
                st.markdown(f"#### 📁 {cat}")
                for subcat in categories[cat]:
                    label, score = predict(text, subcat)
                    st.write(f"🔸 {subcat} → **{label}** ({score:.2%})")
            else:
                st.warning(f"⚠️ `{cat}` 항목은 비어 있습니다.")