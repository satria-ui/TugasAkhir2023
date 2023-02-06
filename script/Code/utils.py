from os import listdir
import os
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
from IPython.display import Audio
import librosa.display
import random
from joblib import load

class data_loader:
    def __init__(self, path: str):
        self.path = path
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

class figures:
    def __init__(self, path: str, emotion: str):
        self.dataset = data_loader(path).getData()
        self.path = list(self.dataset["Path"][self.dataset["Emotions"] == emotion])
        self.idx = random.randint(0, len(path))
        self.emotion = emotion
        self.data, self.sampling_rate = librosa.load(self.path[self.idx])

    def waveplot(self):
        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(self.emotion, size=14)
        librosa.display.waveshow(self.data,sr=self.sampling_rate,color='pink')
        return plt.show()

    def getAudio(self):
        print(f"This is a recording of {self.path[self.idx]} from {self.idx} index of {self.emotion} dataset")
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
        self.dataset = data_loader(path).getData()
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

        decoded_data = np.array([reverse_mapping_dict[i] for i in data])

        return decoded_data

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
        scaler = load('./Scaler/Z-ScoreScaler.joblib')
        x_scaled = scaler.transform(X)
        x_test = pd.DataFrame(x_scaled)

        prediction = loaded_model.predict(x_test)
        prediction = load_model.reverse_label_encoder(prediction)
        return prediction


