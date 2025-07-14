import ffmpeg
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects
import tempfile
import numpy as np

def extract_audio(video_path, audio_out_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_out_path, ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    
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