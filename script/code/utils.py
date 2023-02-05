from os import listdir
import pandas as pd
from matplotlib import pyplot as plt
import librosa
import numpy as np
import librosa.display

class data_loader:
    def __init__(self, path):
        self.path = path
    def getData(self):
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

class figures:
    def waveplot(data,sr,emotion):
        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(emotion, size=14)
        librosa.display.waveshow(data,sr=sr,color='pink')
        plt.show()

    def spectogram_linear(data,sr,emotion):
        x = librosa.stft(data)
        # convert to db
        xdb = librosa.amplitude_to_db(abs(x))
        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(emotion, size=14)
        librosa.display.specshow(xdb,sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format="%+2.f dB")

    def spectogram_log(data,sr,emotion):
        x = librosa.stft(data)
        # convert to db
        xdb = librosa.amplitude_to_db(abs(x))
        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(emotion, size=14)
        librosa.display.specshow(xdb,sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format="%+2.f dB")

class audio_extraction:
    def __init__(self, file):
        self.file = file

    def getMFCC(self):
        y,sr = librosa.load(self.file, duration=4, offset=0.5)
        n_fft = int(sr * 0.02)
        hop_length = n_fft // 2
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).T, axis=0)

        return mfcc