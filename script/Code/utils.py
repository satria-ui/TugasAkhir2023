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
    def __init__(self, path: str, sample_rate: int, num_samples: int, duration: int, device = "cpu"):
        self.path = path
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples
        self.target_duration = duration
        self.device = device

    def getWaveform(self):
        # label_mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
        if os.path.isdir(self.path):
            counter = 0
            audio_path = []
            audio_waveforms = []
            audio_emotion = []
            items = listdir(self.path)
            directory_path = [item for item in items if os.path.isfile(os.path.join(self.path, item))]

            for audio in directory_path:
                audio_path.append(self.path+audio)
                waveform, _ = librosa.load(self.path+audio, duration=self.target_duration, sr=self.target_sample_rate, offset=0.5)
                # make sure waveform vectors are homogenous by defining explicitly
                waveform_homo = np.zeros((int(self.target_sample_rate*self.target_duration)))
                waveform_homo[:len(waveform)] = waveform
                
                emotion = audio.split("_")

                audio_waveforms.append(waveform_homo)

                if emotion[2] == "ANG":
                    audio_emotion.append("0")
                elif emotion[2] == "FEA":
                    audio_emotion.append("1")
                elif emotion[2] == "DIS":
                    audio_emotion.append("2")
                elif emotion[2] == "HAP":
                    audio_emotion.append("3")
                elif emotion[2] == "NEU":
                    audio_emotion.append("4")
                elif emotion[2] == "SAD":
                    audio_emotion.append("5")

                counter += 1
                print('\r'+f' Processed {counter}/{len(directory_path)} audio samples',end='')
            
            return audio_waveforms, audio_emotion
        
        elif os.path.isfile(self.path):
            waveform, _ = librosa.load(self.path, duration=self.target_duration, sr=self.target_sample_rate, offset=0.4)
            # make sure waveform vectors are homogenous by defining explicitly
            waveform_homo = np.zeros((int(self.target_sample_rate*self.target_duration)))
            waveform_homo[:len(waveform)] = waveform

            emotion = self.path.split("_")

            if emotion[2] == "ANG":
                audio_emotion = "0"
            elif emotion[2] == "FEA":
                audio_emotion = "1"
            elif emotion[2] == "DIS":
                audio_emotion = "2"
            elif emotion[2] == "HAP":
                audio_emotion = "3"
            elif emotion[2] == "NEU":
                audio_emotion = "4"
            elif emotion[2] == "SAD":
                audio_emotion = "5"

            return waveform_homo, audio_emotion
        else:
            return "Wrong Audio Path"

    def getData(self):
        if os.path.isdir(self.path):
            audio_path = []
            audio_emotion = []
            items = listdir(self.path)
            directory_path = [item for item in items if os.path.isfile(os.path.join(self.path, item))]

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
            return("Wrong Audio Path")

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

    def plot(self, title):
        if os.path.isfile(self.path):
            waveform, emotion_idx = self.getWaveform()
        else:
            return "Please select single file of audio"
        
        mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
        reverse_mapping_dict = {v: k for k, v in mapping.items()}
        
        if int(emotion_idx) in reverse_mapping_dict:
            emotion = reverse_mapping_dict[int(emotion_idx)]
        else:
            return "Unknown emotion value"

        plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.title(emotion, size=14)

        if title == "Waveform":
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            librosa.display.waveshow(waveform,sr=self.target_sample_rate,color='pink')
        
        elif title == "Spectogram":
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency')
            spec = librosa.stft(waveform)
            spec_db = librosa.amplitude_to_db(abs(spec))
            librosa.display.specshow(spec_db, sr=self.target_sample_rate, x_axis='time', y_axis='log', cmap='plasma')
            plt.colorbar(format="%+2.f dB")
        
        elif title == "MFCC":
            plt.xlabel('Frame')
            plt.ylabel('MFCC Coefficients')

            waveform_np = np.array(waveform, dtype=np.float64)
            m = 1
            n = waveform_np.shape[0]  # number of columns
            X_2d = waveform_np.reshape((m, n))
            features_train = []

            print('Waveforms:')
            features = self.get_features(X_2d, features_train, self.target_sample_rate)
            features = np.array(features)
            # average across frequency axis to get a 2D array
            features2d = np.squeeze(features)
            print(f'\n\nFeatures set: {len(features_train)} total, {len(features_train)} samples')
            print(f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

            librosa.display.specshow(features2d, x_axis='frames', cmap='viridis')
            plt.gca().set_ylabel('MFCC Coefficients', labelpad=10)
            plt.colorbar(format="%+2.f dB")
            plt.tight_layout()
        
        return plt.show()

    def plot_waveform(self):
        self.plot(title="Waveform")

    def plot_spectogram(self):
        self.plot(title="Spectogram")

    def plot_mfcc(self):
        self.plot(title="MFCC")

    def audio_info(self):
        info = torchaudio.info(self.path, format="wav")
        return print(info)

    def play_audio(self):
        if os.path.isfile(self.path):
            waveform, _ = self.getWaveform()
        else:
            return "Please select single file of audio"
        
        display(Audio(waveform, rate = self.target_sample_rate))
    
    def get_features(self, waveforms, features, sample_rate):
        # initialize counter to track progress
        file_count = 0

        # process each waveform individually to get its MFCCs
        for waveform in waveforms:
            mfccs = Transformation.feature_mfcc(waveform, sample_rate)
            features.append(mfccs)
            file_count += 1
            # print progress
            print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')

        # return all features from list of waveforms
        return features

    def extract_audio_svm(self):
        waveform, emotion_idx = self.getWaveform()
        waveform_np = np.array(waveform, dtype=np.float64)
        emotion_np = np.array(emotion_idx, dtype=int)
        
        mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
        reverse_mapping_dict = {v: k for k, v in mapping.items()}
 
        if os.path.isfile(self.path):
            emotions = reverse_mapping_dict[int(emotion_idx)]
            m = 1
            n = waveform_np.shape[0]  # number of columns
            waveform_np = waveform_np.reshape((m, n))
            features_train = []
        elif os.path.isdir(self.path):
            emotions = [reverse_mapping_dict[val] for val in emotion_np]
            features_train = []
        else:
            return "Data input is neither file, nor directory"

        print('\n\nWaveforms:')
        features = self.get_features(waveform_np, features_train, self.target_sample_rate)
        print(f'\n\nFeatures set: {len(features_train)} total samples')
        print(f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')
        mfcc_signal = np.array(features)
        mean_data = np.mean(mfcc_signal, axis=-1)
        df = pd.DataFrame(mean_data)
        df["Emotions"] = emotions
        return df


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
    
    def feature_mfcc(waveform, sample_rate, winlen=512, window='hamming', n_mels=40, n_mfcc=20, hop_length=256):
        # Preprocessing: pad waveform with zeros to be at least 1 second long
        waveform = librosa.util.fix_length(waveform, size = sample_rate)
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mels=n_mels, n_mfcc=n_mfcc,
                                    win_length=winlen, window=window, hop_length=hop_length)
        return mfcc
    # def feature_mfcc(waveform, sample_rate, n_mfcc = 40, fft = 1024, winlen = 512, window='hamming', mels=128):
    #     mfc_coefficients=librosa.feature.mfcc(
    #                         y=waveform, 
    #                         sr=sample_rate, 
    #                         n_mfcc=n_mfcc,
    #                         n_fft=fft, 
    #                         win_length=winlen, 
    #                         window=window, 
    #                         #hop_length=hop, 
    #                         n_mels=mels, 
    #                         fmax=sample_rate/2) 
    #     return mfc_coefficients


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
        self.cmd = CremaD(path, sample_rate=22050, num_samples=22050)
        self.dataset = self.cmd.getData()
        self.file = self.dataset["Path"]

    def mfcc_formula(self):
        signal = []
        for idx in range(len(self.file)):
            mfcc_signal, _ = self.cmd[idx]
            signal.append(mfcc_signal.numpy().reshape(mfcc_signal.size(1), mfcc_signal.size(2)))

        return signal
        # y,sr = librosa.load(audio)
        # n_fft = int(sr * 0.02)
        # hop_length = n_fft // 2
        # mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).T, axis=0)

        # return mfcc

    def extract_audio(self):
        mfcc_signal = self.mfcc_formula()
        X = np.mean(mfcc_signal, axis=-1)
        # concatenated_frames = torch.cat(mfcc_signal, dim=0)
        # features_list = []
        # for i in range(concatenated_frames.size(0)):
        #     sample_features = concatenated_frames[i].view(-1).numpy()
        #     features_list.append(sample_features)
        
        # X_mfcc = self.file.apply(lambda x: audio_extraction.mfcc_formula(x))
        # X = [item for item in X_mfcc]
        # X = np.array(X)
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

        SAMPLE_RATE = 22050
        NUM_SAMPLE = 22050
        DURATION = 3
        
        dataset = CremaD(path=self.audio, sample_rate=SAMPLE_RATE, duration=DURATION, num_samples=NUM_SAMPLE).extract_audio_svm()
        X = dataset.drop(labels='Emotions', axis= 1)
        scaler = joblib.load('./Scaler/Z-ScoreScaler.joblib')
        x_scaled = scaler.transform(X)
        x_test = pd.DataFrame(x_scaled)

        prediction = loaded_model.predict_proba(x_test)
        prediction = load_model.reverse_label_encoder(prediction)
        return prediction   

