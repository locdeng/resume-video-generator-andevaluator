from dotenv import load_dotenv
import os
import google.generativeai as genai
from about_me_gen import generate_text_gemini



def make_prompt_from_features(features):
    prompt = (
        "Bạn là một chuyên gia tuyển dụng.\n"
        "Hãy đánh giá mức độ tự tin của ứng viên dựa trên thông số sau:\n\n"
        f"- Tốc độ nói: {features['speech_rate']} từ/phút\n"
        f"- Filler words: {features['filler_count']} lần\n"
        f"- Số lần ngừng nghỉ: {features['num_pauses']}\n"
        f"- Độ lớn giọng: {features['loudness']} dBFS\n"
        f"- Độ cao trung bình: {features['pitch_mean']} Hz\n"
        f"- Độ biến thiên độ cao: {features['pitch_variation']} Hz\n"
        f"- Nội dung nói: \"{features['transcript']}\"\n\n"
        "Hãy trả lời ngắn gọn, dễ hiểu cho ứng viên biết họ tự tin hay lo lắng, và gợi ý cách cải thiện."
    )
    return prompt
