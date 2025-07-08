import streamlit as st
import tempfile
import ffmpeg
from transformers import pipeline
from pydub import AudioSegment, effects
import librosa
import noisereduce as nr
import numpy as np

# ------------------------------
# Load Whisper once
# ------------------------------
@st.cache_resource
def load_whisper():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=0  # Nếu có GPU
    )

asr = load_whisper()

# ------------------------------
# Extract audio from video
# ------------------------------
def extract_audio(video_path, audio_out_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_out_path, ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )

# ------------------------------
# Preprocess audio: Noise Reduction + Normalize
# ------------------------------
def preprocess_audio(input_path, output_path, target_dbfs=-20):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)

    audio_segment = AudioSegment(
        (reduced_noise * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    normalized_audio = effects.normalize(audio_segment, headroom=-target_dbfs)
    normalized_audio.export(output_path, format="wav")

# ------------------------------
# Split audio into ~30s chunks
# ------------------------------
def split_audio_chunks(audio_path, chunk_length_ms=30000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        tmp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(tmp_chunk.name, format="wav")
        chunks.append(tmp_chunk.name)
    return chunks

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎙️ Korean Speech-to-Text with Whisper")
st.markdown("""
✅ Upload video (.mp4) or audio (.wav/.mp3)  
✅ Auto-extract audio  
✅ Preprocess (Noise Reduction + Normalize)  
✅ Split long audio (>30s)  
✅ Transcribe in Korean with Whisper (with Timestamps)
""")

uploaded_file = st.file_uploader("📤 Upload video or audio", type=["mp4", "wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # STEP 1: Extract audio if video
    if uploaded_file.type.startswith("video/"):
        st.info("🔄 Extracting audio from video...")
        raw_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        extract_audio(input_path, raw_audio_path)
        st.success("🎵 Audio extracted!")
    else:
        raw_audio_path = input_path

    # STEP 2: Preprocess audio
    st.info("✨ Preprocessing (Noise Reduction + Normalize)...")
    preprocessed_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    preprocess_audio(raw_audio_path, preprocessed_audio_path)
    st.success("✅ Audio preprocessed!")

    # STEP 3: Transcription
    if st.button("📝 Transcribe Now"):
        with st.spinner("⏳ Splitting audio into 30s chunks..."):
            chunk_files = split_audio_chunks(preprocessed_audio_path)
            st.success(f"✅ {len(chunk_files)} chunks created.")

        full_transcript = ""
        all_chunks = []

        for idx, chunk_file in enumerate(chunk_files, 1):
            st.info(f"🔊 Transcribing chunk {idx}/{len(chunk_files)}...")

            result = asr(
                chunk_file,
                generate_kwargs={"language": "korean"},
                return_timestamps=True
            )

            full_transcript += result["text"].strip() + "\n\n"

            clean_chunks = []
            for chunk in result["chunks"]:
                start, end = chunk["timestamp"]
                text = chunk["text"].strip()
                if start is not None and end is not None and text:
                    clean_chunks.append({"start": start, "end": end, "text": text})

            # ✅ Fix: append parsed chunks
            all_chunks.extend(clean_chunks)

        # Display results
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
