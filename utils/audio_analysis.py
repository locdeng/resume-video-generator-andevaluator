from pydub import silence, AudioSegment
import librosa
import numpy as np

def analyze_speech_rate(text, audio_length):
    words = len(text.split())
    wpm = words / (audio_length / 60)
    return wpm

def count_filler_words(text, fillers=["음", "어", "그"]):
    return sum(text.count(f) for f in fillers)

def sentence_length(text):
    sentences = text.split(".")
    if not sentences or len(sentences) == 0:
        return 0, 0
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    return len(sentences), avg_len

def detect_pauses(audio_path, silence_thresh=-40, min_silence_len=500):
    audio = AudioSegment.from_file(audio_path)
    silent_chunks = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    pauses = [(start/1000, end/1000) for start, end in silent_chunks]
    return pauses
def measure_loudness(audio_path):
    audio = AudioSegment.from_file(audio_path)
    return audio.dBFS
def analyze_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) == 0:
        return None
    return np.mean(pitches), np.std(pitches)