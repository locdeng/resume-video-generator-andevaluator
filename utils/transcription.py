from transformers import pipeline
import tempfile
from pydub import AudioSegment

# ---------------------------------
# 1️⃣ Load model once
# ---------------------------------
def load_whisper(model_name="openai/whisper-large-v3", device=0):
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device
    )

# ---------------------------------
# 2️⃣ Split audio chunks (~30s)
# ---------------------------------
def split_audio_chunks(audio_path, chunk_length_ms=30000):
    audio = AudioSegment.from_file(audio_path)
    total_length = len(audio)
    chunks = []
    start = 0
    
    while start < total_length:
        end = min(start + chunk_length_ms, total_length)  # Đảm bảo chunk cuối không quá dài
        chunk = audio[start:end]
        tmp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(tmp_chunk.name, format="wav")
        chunks.append(tmp_chunk.name)
        start = end  # Cập nhật lại vị trí bắt đầu của chunk tiếp theo
    return chunks

# ---------------------------------
# 3️⃣ Transcribe one chunk
# ---------------------------------
def transcribe_chunk(asr_pipeline, audio_file, language="korean"):
    return asr_pipeline(
        audio_file,
        generate_kwargs={"language": language},
        return_timestamps=True  # Đảm bảo bật tham số này để lấy timestamp
    )