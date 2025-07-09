from dotenv import load_dotenv
import os
import google.generativeai as genai
from about_me_gen import generate_text_gemini
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

features = {
    "speech_rate": 175,
    "filler_count": 12,
    "num_pauses": 8,
    "loudness": -23,
    "pitch_mean": 150,
    "pitch_variation": 17,
    "transcript": """안녕하세요. 저는 명지대학교에서 관광경영을 전공하고 있는 27살 브 타이 호앙입니다.
    관광 산업에 대한 깊은 관심으로 한국에 유학하게 되었고, 학교에서는 관광 마케팅과 서비스 기획 수업을 중점적으로 수강했습니다.
    최근에는 소규모 여행 코스 기획 프로젝트에 참여하면서 실제 고객 니즈를 반영한 일정 구성과 발표 경험도 쌓았습니다.
    저는 사람들과 소통하는 것을 좋아하고, 다양한 문화를 이해하며 고객 맞춤 서비스를 제공하는 데 강점이 있습니다.
    앞으로는 한국 관광 산업 현장에서 실무 경험을 쌓고, 글로벌 감각을 갖춘 전문가로 성장하고 싶습니다. 감사합니다.
    """
}
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
prompt = make_prompt_from_features(features)
response = generate_text_gemini(prompt)

print("INSIGHT TỪ GEMINI:")
print(response)