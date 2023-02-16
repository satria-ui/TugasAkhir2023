from os import listdir
import os
import time
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
from IPython.display import Audio, display
import librosa.display
import random
import torchaudio
import torch
import joblib

class CremaD:
    def __init__(self, path: str, sample_rate: int, num_samples: int, device = "cuda"):
        self.path = path
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = device

    def getData(self):
        if os.path.isdir(self.path):
            audio_path = []
            audio_emotion = []
            directory_path = listdir(self.path)

            for audio in directory_path:
                audio_path.append(self.path+audio)
                emotion = audio.split("_")

                if emotion[2] == "ANG":
                    audio_emotion.append("angry")
                elif emotion[2] == "FEA":
                    audio_emotion.append("fear")
                elif emotion[2] == "DIS":
                    audio_emotion.append("disgust")
                elif emotion[2] == "HAP":
                    audio_emotion.append("happy")
                elif emotion[2] == "NEU":
                    audio_emotion.append("neutral")
                elif emotion[2] == "SAD":
                    audio_emotion.append("sad")

            emotion_dataset = pd.DataFrame(audio_emotion, columns=['Emotions'])
            audio_path_dataset = pd.DataFrame(audio_path, columns=['Path'])
            dataset = pd.concat([audio_path_dataset, emotion_dataset], axis= 1)
            return dataset
        elif os.path.isfile(self.path):
            audio_emotion = []
            audio_path = [self.path]
            emotion = self.path.split("_")

            if emotion[2] == "ANG":
                audio_emotion.append("angry")
            elif emotion[2] == "FEA":
                audio_emotion.append("fear")
            elif emotion[2] == "DIS":
                audio_emotion.append("disgust")
            elif emotion[2] == "HAP":
                audio_emotion.append("happy")
            elif emotion[2] == "NEU":
                audio_emotion.append("neutral")
            elif emotion[2] == "SAD":
                audio_emotion.append("sad")

            emotion_dataset = pd.DataFrame(audio_emotion, columns=['Emotions'])
            audio_path_dataset = pd.DataFrame(audio_path, columns=['Path'])
            dataset = pd.concat([audio_path_dataset, emotion_dataset], axis= 1)
            return dataset
        else:
            return("Wrong Path File")

    def __len__(self):
        return len(self.getData())

    def __getitem__(self, index):
        audio_path = f"{self.getData().Path[index]}"
        label_mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
        new_dataset = self.getData().replace(({"Emotions": label_mapping}))
        label = new_dataset.Emotions[index]
        signal, sr = torchaudio.load(audio_path)
        signal = signal.to(self.device)
        # pre-processing
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        signal = self.cut_signal(signal)
        processed_signal = self.right_padding(signal)
        # transformation
        transformation = Transformation(self.target_sample_rate, self.device).MFCC()
        transformed_signal = transformation(processed_signal)
        return transformed_signal,label

    def plot(self, title, index):
        audio_path = f"{self.getData().Path[index]}"
        emotion = f"{self.getData().Emotions[index]}"
        np.seterr(divide = 'ignore')

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sr

        figure, axes = plt.subplots(num_channels, 1, figsize=(15,4), facecolor=(.9,.9,.9))
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            if title == "Waveform":
                axes[c].plot(time_axis, waveform[c], linewidth=1, color = "pink")
                axes[c].grid(True)
                axes[c].set_xlabel('Time (s)')
                axes[c].set_ylabel('Amplitude')
                axes[c].set_title(emotion)
            elif title == "MFCC":
                mfcc_signal, label = self.__getitem__(index=index)
                mfcc_signal = mfcc_signal.cpu()
                axes[c].set_title("Mel-Frequency Cepstrum")
                axes[c].set_ylabel("Features")
                axes[c].set_xlabel("Frame")
                im = axes[c].imshow(librosa.power_to_db(mfcc_signal[0]), origin="lower", aspect="auto")
                figure.colorbar(im, ax=axes[c], format = "%+2.f dB")
            else:
                Pxx, freqs, bins, im = axes[c].specgram(waveform[c], Fs=sr, cmap = "plasma")
                figure.colorbar(im,ax= axes[c],format="%+2.f dB")
                axes[c].set_title(emotion)
                axes[c].set_xlabel('Time (s)')
                axes[c].set_ylabel('Frequency')
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
        plt.show(block=False)

    def plot_waveform(self, index):
        self.plot(title="Waveform", index=index)

    def plot_spectogram(self, index):
        self.plot(title="Spectrogram", index=index)

    def plot_mfcc(self, index):
        self.plot(title="MFCC", index=index)

    def audio_info(self, index):
        audio_path = f"{self.getData().Path[index]}"
        info = torchaudio.info(audio_path, format="wav")
        return print(info)

    def play_audio(self, index):
        audio_path = f"{self.getData().Path[index]}"
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.numpy()
        num_channels, num_frames = waveform.shape
        if num_channels == 1:
            display(Audio(waveform[0], rate = sr))
        elif num_channels == 2:
            display(Audio((waveform[0], waveform[1]), rate = sr))
        else:
            raise ValueError("Waveform with more than 2 channels are not supported")

    # def plot_mel_spectogram(self, index):


    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)

        return signal

    def mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim = 0, keepdim = True)

        return signal

    def right_padding(self,signal):
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            missing_samples = self.num_samples - signal_length
            pad_number = (0, missing_samples)
            signal = torch.nn.functional.pad(signal, pad_number)
        return signal

    def cut_signal(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal


class Transformation:
    def __init__(self, sr, device):
        self.sr = sr
        self.device = device
    def mel_spectogram(self):
        mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate= self.sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
            ).to(self.device)
        return mel_spectogram

    def MFCC(self):
        n_mels = 80
        n_mfcc = int(n_mels*(2/3))
        n_fft = int(self.sr * 0.02)
        hop_length = n_fft // 2
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "mel_scale": "htk",
                },
            ).to(self.device)
        return mfcc_transform


class figures:
    def __init__(self, path: str, emotion: str):
        self.dataset = CremaD(path).getData()
        self.path = list(self.dataset["Path"][self.dataset["Emotions"] == emotion])
        self.idx = 0
        # self.idx = random.randint(0, len(path))
        self.emotion = emotion
        self.data, self.sampling_rate = librosa.load(self.path[self.idx])

    def waveplot(self):
        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(self.emotion, size=14)
        librosa.display.waveshow(self.data,sr=self.sampling_rate,color='pink')
        return plt.show()

    def getAudio(self):
        print(f"This is a recording of {self.path[self.idx]} from index {self.idx} of {self.emotion} dataset")
        return Audio(self.path[self.idx])

    def spectogram(self, display="hz"):
        x = librosa.stft(self.data)
        # convert to db
        xdb = librosa.amplitude_to_db(abs(x))
        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(self.emotion, size=14)
        librosa.display.specshow(xdb,sr=self.sampling_rate, x_axis='time', y_axis=display)

        return plt.colorbar(format="%+2.f dB")

class audio_extraction:
    def __init__(self, path: str):
        self.dataset = CremaD(path, sample_rate=22050, num_samples=22050).getData()
        self.file = self.dataset["Path"]

    def mfcc_formula(audio):
        y,sr = librosa.load(audio, duration=4, offset=0.5)
        n_fft = int(sr * 0.02)
        hop_length = n_fft // 2
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).T, axis=0)

        return mfcc

    def extract_audio(self):
        X_mfcc = self.file.apply(lambda x: audio_extraction.mfcc_formula(x))
        X = [item for item in X_mfcc]
        X = np.array(X)
        y = self.dataset["Emotions"]

        extracted_audio = pd.DataFrame(X)
        extracted_audio["Emotions"] = y
        return extracted_audio

class load_model:
    def __init__(self, audio:str, model:str):
        self.model = model
        self.audio = audio

    def reverse_label_encoder(data):
        mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
        reverse_mapping_dict = {v: k for k, v in mapping.items()}
        keys = list(reverse_mapping_dict.values())

        # decoded_data = np.array([reverse_mapping_dict[i] for i in data])

        result = {f'{keys[i]}': data[0, i] for i in range(len(keys))}

        return result

    def getModelPrediction(self):
        try:
            loaded_model = pickle.load(open(f"./ML_Model/{self.model}", 'rb'))
            if not os.path.isfile(self.audio):
                raise ValueError("Wrong Audio Path")
            else:
                pass

        except (ValueError, FileNotFoundError) as e:
            return e

        dataset = audio_extraction(self.audio).extract_audio()
        X = dataset.drop(labels='Emotions', axis= 1)
        scaler = joblib.load('./Scaler/Z-ScoreScaler.joblib')
        x_scaled = scaler.transform(X)
        x_test = pd.DataFrame(x_scaled)

        prediction = loaded_model.predict_proba(x_test)
        prediction = load_model.reverse_label_encoder(prediction)
        return prediction


