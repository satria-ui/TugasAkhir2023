from os import listdir
import os
import time
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
from IPython.display import Audio
import librosa.display
import random
import torchaudio
import torch
import joblib

class data_loader:
    def __init__(self, path: str, sample_rate: int, num_samples: int):
        self.path = path
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples

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
        label = f"{self.getData().Emotions[index]}"
        signal, sr = torchaudio.load(audio_path)
        # pre-processing
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        signal = self.cut_signal(signal)
        signal = self.right_padding(signal)
        # transformation = Transformation(self.target_sample_rate).mel_spectogram()
        # signal = transformation(signal)
        return signal,label,sr

    def waveplot(self, index):
        data, emotion, sample_rate = self.__getitem__(index)
        path = f"{self.getData().Path[index]}"
        emotion = f"{self.getData().Emotions[index]}"
        print(f"This is a recording of {path} from index {index} of {emotion} emotion from the dataset")

        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(emotion, size=14)
        librosa.display.waveshow(data.numpy(),sr=sample_rate,color='pink')
        plt.show()

        time.sleep(1)
        return Audio(path)

    def spectogram(self, index, display):
        path = f"{self.getData().Path[index]}"
        emotion = f"{self.getData().Emotions[index]}"
        data, sample_rate = librosa.load(path)
        x = librosa.stft(data)
        # convert to db
        xdb = librosa.amplitude_to_db(abs(x))
        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(emotion, size=14)
        librosa.display.specshow(xdb,sr=sample_rate, x_axis='time', y_axis=display)

        return plt.colorbar(format="%+2.f dB")


    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
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
    def __init__(self, sr):
        self.sr = sr
    def mel_spectogram(self):
        mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate= self.sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
            )
        return mel_spectogram

class figures:
    def __init__(self, path: str, emotion: str):
        self.dataset = data_loader(path).getData()
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
        self.dataset = data_loader(path, sample_rate=22050, num_samples=22050).getData()
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


