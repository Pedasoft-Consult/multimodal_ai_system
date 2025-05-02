# src/preprocessing/audio/preprocessor.py
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class AudioPreprocessor:
    def __init__(self, sample_rate=16000, model_name="facebook/wav2vec2-base-960h"):
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def load_audio(self, audio_path):
        """Load audio file and resample if necessary"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio

    def extract_mfcc(self, audio, n_mfcc=13):
        """Extract MFCC features from audio"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        # Normalize features
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        return mfccs

    def extract_spectrogram(self, audio):
        """Extract spectrogram from audio"""
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        # Convert to log scale (dB)
        log_spec = librosa.power_to_db(spectrogram, ref=np.max)
        return log_spec

    def data_augmentation(self, audio):
        """Apply simple audio augmentation (pitch and speed)"""
        # Pitch shift (up)
        pitch_up = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=2)
        # Pitch shift (down)
        pitch_down = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=-2)
        # Time stretch (faster)
        faster = librosa.effects.time_stretch(audio, rate=1.2)

        return {
            'original': audio,
            'pitch_up': pitch_up,
            'pitch_down': pitch_down,
            'faster': faster
        }

    def extract_wav2vec_embeddings(self, audio):
        """Extract embeddings using Wav2Vec2 model"""
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean of the hidden states as embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def process_audio(self, audio_path, extract_mfcc=True, extract_spec=True, extract_embeddings=True):
        """Full preprocessing pipeline for audio"""
        audio = self.load_audio(audio_path)
        result = {'audio': audio}

        if extract_mfcc:
            result['mfcc'] = self.extract_mfcc(audio)

        if extract_spec:
            result['spectrogram'] = self.extract_spectrogram(audio)

        if extract_embeddings:
            result['embeddings'] = self.extract_wav2vec_embeddings(audio)

        return result