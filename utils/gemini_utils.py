from dotenv import load_dotenv
import os
import google.generativeai as genai
from about_me_gen import generate_text_gemini



def make_prompt_from_features(features):
    prompt = (
        "당신은 채용 전문가입니다.\n"
        "다음 정보를 바탕으로 지원자의 자신감 정도를 평가해 주세요:\n\n"
        f"- 말하기 속도: {features['speech_rate']} 단어/분\n"
        f"- 군더더기 단어 사용 횟수: {features['filler_count']} 회\n"
        f"- 멈춤 횟수: {features['num_pauses']} 회\n"
        f"- 음량: {features['loudness']} dBFS\n"
        f"- 평균 음높이: {features['pitch_mean']} Hz\n"
        f"- 음높이 변화량: {features['pitch_variation']} Hz\n"
        f"- 말한 내용: \"{features['transcript']}\"\n\n"
        "지원자에게 자신감 있는지 긴장했는지 짧고 이해하기 쉽게 알려 주고, 개선 방법도 제안해 주세요."
    )
    return prompt

def build_insight_prompt_ko(pose_counts, hand_counts, emotion_counts):
    prompt = "당신은 면접 분석 전문가입니다. 아래는 면접 영상에서 지원자의 자세(Pose), 손 제스처(Hand Gesture), 감정(Emotion)에 대한 통계 수치입니다.\n\n"

    prompt += "⭐ Pose Summary:\n"
    for k, v in pose_counts.items():
        prompt += f"- {k}: {v} 프레임\n"

    prompt += "\n⭐ Hand Gesture:\n"
    for k, v in hand_counts.items():
        prompt += f"- {k}: {v} 프레임\n"

    prompt += "\n⭐ Emotion:\n"
    for k, v in emotion_counts.items():
        prompt += f"- {k}: {v} 프레임\n"

    prompt += (
        "\n\n👉 위 데이터를 기반으로 **한국어로 4~6문장 정도의 간결한 인사이트 분석**을 작성해주세요. "
        "✅ 문장은 각 항목별로 보기 쉽게 **• (bullet-point) 형태**로 구분해 주세요. "
        "✅ 지원자의 자세, 손 제스처, 감정의 특징과 전반적인 인상을 전문적이고 명확하며 이해하기 쉽게 설명해 주세요. "
        "✅ 기업의 면접관이 참고할 수 있도록 정중하고 분석적인 어조로 작성해 주세요."
    )

    return prompt

