import streamlit as st
import tempfile
import librosa
import numpy as np
from utils.audio_preprocessing import extract_audio, preprocess_audio
from utils.transcription import load_whisper, split_audio_chunks
from utils.audio_analysis import (
    analyze_speech_rate, count_filler_words, sentence_length,
    detect_pauses, measure_loudness, analyze_pitch
)
from dotenv import load_dotenv
import os
import google.generativeai as genai
from about_me_gen import generate_text_gemini
from utils.gemini_utils import make_prompt_from_features

# Load .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Load Whisper once (global to avoid reloading)
asr = load_whisper()

def run_transcription_app():
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
    
    """
    Streamlit App UI for Korean Speech Transcription and Analysis
    """
    # st.title("🎙️ Whisper를 활용한 한국어 음성 전사")
    st.markdown('<div class="intro-title">동영상 음성 분석</div>', unsafe_allow_html=True)
    st.markdown("""
    - 한국어에만 적용
    - 비디오 (.mp4) 또는 오디오 (.wav/.mp3) 업로드  
    - 오디오 자동 추출  
    - 노이즈 제거 + 볼륨 정규화  
    - 긴 오디오를 자동으로 30초 청크로 분할  
    - Whisper 모델로 한국어 전사
    """)

    uploaded_file = st.file_uploader("📤 비디오 또는 오디오 업로드", type=["mp4", "wav", "mp3"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name

        st.success(f"✅ Uploaded: {uploaded_file.name}")

        # STEP 1: Extract audio if video
        if uploaded_file.type.startswith("video/"):
            st.info("🔄 비디오에서 오디오 추출 중...")
            raw_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            extract_audio(input_path, raw_audio_path)
            st.success("🎵 오디오 추출 완료!")
        else:
            raw_audio_path = input_path

        # STEP 2: Preprocess audio
        st.info("✨ Preprocessing (Noise Reduction + Normalize)...")
        preprocessed_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        preprocess_audio(raw_audio_path, preprocessed_audio_path)
        st.success("✅ Audio preprocessed!")

        # STEP 3: Transcription
        if st.button("📝 Transcribe Now"):
            with st.spinner("⏳ 오디오를 30초 청크로 분할 중..."):
                chunk_files = split_audio_chunks(preprocessed_audio_path)
                st.success(f"✅ {len(chunk_files)} 개의 청크 생성 완료.")

            full_transcript = ""
            all_chunks = []
            total_audio_length = 0
            all_pauses = []
            all_loudness = []
            all_pitches = []
            offset = 0

            speech_rates = []
            filler_counts = []
            num_sentences_all = []
            avg_sentence_lengths = []

            for idx, chunk_file in enumerate(chunk_files, 1):
                st.info(f"🔊 Transcribing chunk {idx}/{len(chunk_files)}...")

                result = asr(
                    chunk_file,
                    generate_kwargs={"language": "korean"},
                    return_timestamps=True
                )

                full_transcript += result["text"].strip() + "\n\n"
                duration = librosa.get_duration(filename=chunk_file)
                total_audio_length += duration

                speech_rate = analyze_speech_rate(result["text"], duration)
                filler_count = count_filler_words(result["text"])
                num_sentences, avg_sentence_length = sentence_length(result["text"])

                speech_rates.append(speech_rate)
                filler_counts.append(filler_count)
                num_sentences_all.append(num_sentences)
                avg_sentence_lengths.append(avg_sentence_length)

                clean_chunks = []
                for chunk in result["chunks"]:
                    start, end = chunk["timestamp"]
                    text = chunk["text"].strip()
                    if start is not None and end is not None and text:
                        clean_chunks.append({
                            "start": start + offset,
                            "end": end + offset,
                            "text": text
                        })

                all_chunks.extend(clean_chunks)
                offset += duration

                pauses = detect_pauses(chunk_file)
                loudness = measure_loudness(chunk_file)
                pitch_mean, pitch_variation = analyze_pitch(chunk_file)

                all_pauses.extend(pauses)
                all_loudness.append(loudness)
                all_pitches.append((pitch_mean, pitch_variation))

                st.write(f"✅ 말 속도: {speech_rate:.2f} 단어/분")
                st.write(f"✅ 메우는 말 횟수: {filler_count}")
                st.write(f"✅ 문장 수: {num_sentences}, 평균 문장 길이: {avg_sentence_length:.2f}")
                st.write(f"✅ 멈춤: {len(pauses)}회")
                st.write(f"✅ 평균 볼륨: {loudness:.2f} dBFS")
                st.write(f"✅ 음높이 (평균, 변동): {pitch_mean:.2f} Hz, {pitch_variation:.2f} Hz")

            # 요약
            total_pauses = len(all_pauses)
            avg_speech_rate = np.mean(speech_rates)
            avg_filler_count = np.sum(filler_counts)
            avg_loudness = np.mean(all_loudness)
            avg_pitch_mean = np.mean([p[0] for p in all_pitches if p[0] is not None])
            avg_pitch_variation = np.mean([p[1] for p in all_pitches if p[1] is not None])

            st.subheader("🔍 전체 오디오 분석 요약")
            st.write(f"✅ 총 멈춤 횟수: {total_pauses}")
            st.write(f"✅ 말 속도 평균: {avg_speech_rate:.2f} 단어/분")
            st.write(f"✅ 총 메우는 말 횟수: {avg_filler_count}")
            st.write(f"✅ 평균 볼륨: {avg_loudness:.2f} dBFS")
            st.write(f"✅ 음높이 평균: {avg_pitch_mean:.2f} Hz")
            st.write(f"✅ 음높이 변동: {avg_pitch_variation:.2f} Hz")

            st.subheader("📜 Full Transcript (Korean)")
            st.text_area("전체 스크립트:", full_transcript.strip(), height=300)

            st.subheader("🕒 Segmented with Timestamps")
            for c in all_chunks:
                st.markdown(f"**[{c['start']:.1f}s - {c['end']:.1f}s]**: {c['text']}")

            st.download_button(
                "💾 Download Transcript",
                data=full_transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )

            # Save results to session_state
            st.session_state["analysis_result"] = {
                "speech_rate": avg_speech_rate,
                "filler_count": avg_filler_count,
                "num_pauses": total_pauses,
                "loudness": avg_loudness,
                "pitch_mean": avg_pitch_mean,
                "pitch_variation": avg_pitch_variation,
                "transcript": full_transcript
            }

        # AI Insight button
        if "analysis_result" in st.session_state:
            if st.button("✨ AI 인사이트 요청"):
                with st.spinner("🔍 물어보고 있는중..."):
                    prompt = make_prompt_from_features(st.session_state["analysis_result"])
                    result = generate_text_gemini(prompt)
                st.subheader("📈 AI Insight")
                st.write(result)
