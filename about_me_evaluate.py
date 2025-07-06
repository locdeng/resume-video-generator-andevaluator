#자소서 + 이력서 펼가 -> KoBERT 이용

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

@st.cache_resource
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    with open("saved_model/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    model.eval()
    return model, tokenizer, le

def predict(text, model, tokenizer, le):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
    return le.inverse_transform([pred])[0], probs[0][pred].item()

def resume_predict_tab():
    st.title("📄 Resume 평가 시스템")
    model, tokenizer, le = load_model_and_tokenizer()

    text = st.text_area("📝 Paste your resume text here:", height=200)

    if st.button("🔍 Evaluate"):
        if text.strip() == "":
            st.warning("⚠️ 텍스트를 입력해주세요.")
        else:
            label, score = predict(text, model, tokenizer, le)
            st.success(f"📌 예측 결과: **{label}** ({score:.2%})")
