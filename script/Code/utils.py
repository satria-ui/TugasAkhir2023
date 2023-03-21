from os import listdir
import os
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import librosa
from sklearn.model_selection import train_test_split
import numpy as np
from IPython.display import Audio, display
import librosa.display
import random
import torchaudio
from torch import nn
import torch
import joblib

class CremaD:
    def __init__(self, path: str, sample_rate: int, num_samples: int, duration: int, device = "cuda"):
        self.path = path
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples
        self.target_duration = duration
        self.device = device
    
    def getWaveformRavdess(self):
        if os.path.isdir(self.path):
            counter = 0
            audio_path = []
            audio_waveforms = []
            audio_emotion = []
            items = os.listdir(self.path)
            directory_path = [item for item in items if os.path.isfile(os.path.join(self.path, item))]

            for audio in directory_path:
                audio_path.append(self.path + audio)
                waveform, _ = librosa.load(self.path+audio, duration=self.target_duration, sr=self.target_sample_rate, offset=0.8)
                # waveform, _ = librosa.load(self.path+audio)

                # make sure waveform vectors are homogenous by defining explicitly
                waveform_homo = np.zeros((int(self.target_sample_rate*self.target_duration)))
                waveform_homo[:len(waveform)] = waveform
                
                emotion = int(audio.split("-")[2])
                audio_waveforms.append(waveform_homo)

                # label_mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
                if emotion == int(5):
                    audio_emotion.append("0")
                elif emotion == int(6):
                    audio_emotion.append("1")
                elif emotion == int(7):
                    audio_emotion.append("2")
                elif emotion == int(3):
                    audio_emotion.append("3")
                elif emotion == int(1):
                    audio_emotion.append("4")
                elif emotion == int(4):
                    audio_emotion.append("5")

                counter += 1
                print('\r'+f' Processed {counter}/{len(directory_path)} audio waveforms',end='')
            
            return audio_waveforms, audio_emotion
        
        elif os.path.isfile(self.path):
            waveform, _ = librosa.load(self.path, sr=self.target_sample_rate, duration=self.target_duration, offset=0.8)
            # make sure waveform vectors are homogenous by defining explicitly
            waveform_homo = np.zeros((int(self.target_sample_rate*self.target_duration)))
            waveform_homo[:len(waveform)] = waveform

            emotion = int(self.path.split("-")[2])

            # label_mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
            if emotion == int(5):
                audio_emotion = "0"
            elif emotion == int(6):
                audio_emotion = "1"
            elif emotion == int(7):
                audio_emotion = "2"
            elif emotion == int(3):
                audio_emotion = "3"
            elif emotion == int(1):
                audio_emotion = "4"
            elif emotion == int(4):
                audio_emotion = "5"
        
            return waveform_homo, audio_emotion
        else:
            return "Wrong Audio Path"

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
                waveform, _ = librosa.load(self.path+audio, duration=self.target_duration, sr=self.target_sample_rate, offset=0.3)
                # waveform, _ = librosa.load(self.path+audio)
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
                print('\r'+f' Processed {counter}/{len(directory_path)} audio waveforms',end='')
            
            return audio_waveforms, audio_emotion
        
        elif os.path.isfile(self.path):
            waveform, _ = librosa.load(self.path, duration=self.target_duration, sr=self.target_sample_rate, offset=0.3)
            # waveform_homo, _ = librosa.load(self.path, duration = self.target_duration)
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
    def getDataRavdess(self):
        if os.path.isdir(self.path):
            file_emotion = []
            file_path = []

            items = os.listdir(self.path)

            for x in items:
                part = x.split('-')
                file_emotion.append(int(part[2]))
                file_path.append(self.path+x)

            path_df = pd.DataFrame(file_path, columns=['Path'])
            emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
            Ravdess_df = pd.concat([path_df, emotion_df], axis=1)

            Ravdess_df.Emotions.replace({1:'Neutral', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust'}, inplace=True)
            return Ravdess_df
        else:
            return "Wrong audio path"

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
            waveform, emotion_idx = self.getWaveformRavdess()
        else:
            return "Please select single file of audio"
        
        mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
        reverse_mapping_dict = {v: k for k, v in mapping.items()}
        
        if int(emotion_idx) in reverse_mapping_dict:
            emotion = reverse_mapping_dict[int(emotion_idx)]
        else:
            return "Unknown emotion value"

        # plt.figure(figsize=(15,4), facecolor=(.9,.9,.9))
        plt.figure(figsize=(4,2))
        plt.title(emotion, size=8)

        if title == "Waveform":
            plt.xlabel('Time (s)', fontsize=8)
            plt.xticks(fontsize=8)
            plt.ylabel('Amplitude', fontsize=8)
            plt.yticks(fontsize=8)
            librosa.display.waveshow(waveform,sr=self.target_sample_rate,color='pink')
        
        elif title == "Spectogram":
            plt.xlabel('Time (s)', fontsize=8)
            plt.xticks(fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
            plt.yticks(fontsize=8)
            spec = librosa.stft(waveform)
            spec_db = librosa.amplitude_to_db(abs(spec))
            librosa.display.specshow(spec_db, sr=self.target_sample_rate, x_axis='time', y_axis='log', cmap='plasma')
            cbar = plt.colorbar(format="%+2.f dB")
            cbar.ax.tick_params(labelsize=8)
        
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
            waveform, _ = self.getWaveformRavdess()
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
            print('\r'+f' Processed {file_count}/{len(waveforms)} waveform features',end='')

        # return all features from list of waveforms
        return features
    
    def create_readyToTrain_data(self, test_data, train_data):
        self.path = test_data
        print(f"Path is now {self.path}")
        # waveforms_testing, emotions_testing = self.getWaveform()
        waveforms_testing, emotions_testing = self.getWaveformRavdess()
        
        self.path = train_data
        print(f"\nPath is now {self.path}")
        # waveforms_training, emotions_training = self.getWaveform()
        waveforms_training, emotions_training = self.getWaveformRavdess()
        
        #################### Split Train and Validation Data ####################
        print("\n\nSplitting train and validation data...\n")
        waveforms_testing = np.array(waveforms_testing, dtype=np.float64)
        emotions_testing = np.array(emotions_testing, dtype=int)
        waveforms_training = np.array(waveforms_training, dtype=np.float64)
        emotions_training = np.array(emotions_training, dtype=int)

        X_train, X_valid, y_train, y_valid = train_test_split(waveforms_training, emotions_training, test_size=0.1, random_state=123, stratify=emotions_training)
        X_test = waveforms_testing
        y_test = emotions_testing
        (unique_train, counts_train) = np.unique(y_train, return_counts=True)
        (unique_valid, counts_valid) = np.unique(y_valid, return_counts=True)
        (unique_test, counts_test) = np.unique(y_test, return_counts=True)
        print(f'Training waveforms:{X_train.shape}, y_train:{y_train.shape}')
        print(f'Validation waveforms:{X_valid.shape}, y_valid:{y_valid.shape}')
        print(f'Test waveforms:{X_test.shape}, y_test:{y_test.shape}')
        print(f"\nTrain Set Data : {len(y_train)}")
        print("Train Emotion Count")
        print(unique_train, counts_train)
        print(f"Validation Set Data : {len(y_valid)}")
        print("Validation Emotion Count")
        print(unique_valid, counts_valid)
        print(f"Test Set Data : {len(y_test)}")
        print("Test Emotion Count")
        print(unique_test, counts_test)

        #################### EXTRACT FEATURES ####################
        features_train, features_valid, features_test = [],[],[]
        print('\nExtracting Train waveforms:')
        features_train = self.get_features(X_train, features_train, self.target_sample_rate)

        print('\n\nExtracting Validation waveforms:')
        features_valid = self.get_features(X_valid, features_valid, self.target_sample_rate)

        print('\n\nExtracting Test waveforms:')
        features_test = self.get_features(X_test, features_test, self.target_sample_rate)

        print(f'\n\nExtracted Features set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, and {len(features_test)} test data')
        print(f'Features shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

        #################### FORMAT DATA FOR TENSOR ####################
        X_train = np.expand_dims(features_train,1)
        X_valid = np.expand_dims(features_valid, 1)
        X_test = np.expand_dims(features_test,1)

        y_train = np.array(y_train, dtype=int)
        y_valid = np.array(y_valid, dtype=int)
        y_test = np.array(y_test, dtype=int)
        
        #################### FEATURE SCALING ####################
        scaler = StandardScaler()

        #### Scale the training data ####
        BATCH,CHANNEL,WIDTH,HEIGHT = X_train.shape
        X_train = np.reshape(X_train, (BATCH,-1)) 
        X_train = scaler.fit_transform(X_train)
        X_train = np.reshape(X_train, (BATCH,CHANNEL,WIDTH,HEIGHT))

        ##### Scale the validation set ####
        BATCH,CHANNEL,WIDTH,HEIGHT = X_valid.shape
        X_valid = np.reshape(X_valid, (BATCH,-1))
        X_valid = scaler.transform(X_valid)
        X_valid = np.reshape(X_valid, (BATCH,CHANNEL,WIDTH,HEIGHT))

        #### Scale the test set ####
        BATCH,CHANNEL,WIDTH,HEIGHT = X_test.shape
        X_test = np.reshape(X_test, (BATCH,-1))
        X_test = scaler.transform(X_test)
        X_test = np.reshape(X_test, (BATCH,CHANNEL,WIDTH,HEIGHT))

        # check shape of each set again
        print(f'\nShape of 4D feature array for input tensor: {X_train.shape} train, {X_valid.shape} validation, {X_test.shape} test')
        joblib.dump(scaler, "./Scaler/CNNScaler.joblib")

        #################### SAVE READY TO TRAIN DATA ####################
        filename = "./Scaler/CREMA-D_ready_data.npy"
        with open(filename, 'wb') as f:
            np.save(f, X_train)
            np.save(f, X_valid)
            np.save(f, X_test)
            np.save(f, y_train)
            np.save(f, y_valid)
            np.save(f, y_test)

        print(f'\nFeatures and labels saved to {filename}')
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def load_readyToTrain_data(self, train_file):
        with open(train_file, 'rb') as f:
            X_train = np.load(f)
            X_valid = np.load(f)
            X_test = np.load(f)
            y_train = np.load(f)
            y_valid = np.load(f)
            y_test = np.load(f)
        print("Data Loaded with shape:")
        print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
        print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
        print(f'X_test:{X_test.shape}, y_test:{y_test.shape}') 
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def extract_audio_svm(self):
        waveform, emotion_idx = self.getWaveformRavdess()
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

class DeepLearning:
    def __init__(self, sample_rate: int, duration: int, device = "cuda"):
        self.target_sample_rate = sample_rate
        self.target_duration = duration
        self.device = device
    
    def train(self, train_file, model, num_epochs):
        CMD = CremaD(path="./dataset/", sample_rate=self.target_sample_rate, duration=self.target_duration, num_samples=22050)
        X_train, X_valid, X_test, y_train, y_valid, y_test = CMD.load_readyToTrain_data(train_file)
        train_size = X_train.shape[0]
        minibatch = 32

        print(f'\n{self.device} selected')

        # instantiate model and move to GPU for training
        model = model().to(self.device) 
        optimizer = torch.optim.SGD(model.parameters(),lr=0.001, weight_decay=0.001, momentum=0.8)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=5, verbose=True)
        print('Number of trainable params: ',sum(p.numel() for p in model.parameters()))

        # instantiate the checkpoint save function
        save_checkpoint = self.make_save_checkpoint()
        # instantiate the training step function 
        train_step = self.train_single_epoch(model, optimizer, criterion)
        # instantiate the validation loop function
        validate = self.validate_single_epoch(model, criterion)

        # instantiate lists to hold scalar performance metrics to plot later
        train_losses = []
        valid_losses = []
        train_accuracy = []
        valid_accuracy = []

        print("\nStart Training...")
        for epoch in range(num_epochs):
            
            # set model to train phase
            model.train()         
            
            # shuffle entire training set in each epoch to randomize minibatch order
            train_indices = np.random.permutation(train_size) 
            
            # shuffle the training set for each epoch:
            X_train = X_train[train_indices,:,:,:] 
            y_train = y_train[train_indices]

            # instantiate scalar values to keep track of progress after each epoch so we can stop training when appropriate 
            epoch_acc = 0 
            epoch_loss = 0
            num_iterations = int(train_size / minibatch)
            
            # create a loop for each minibatch of 32 samples:
            for i in range(num_iterations):
                
                # we have to track and update minibatch position for the current minibatch
                # if we take a random batch position from a set, we almost certainly will skip some of the data in that set
                # track minibatch position based on iteration number:
                batch_start = i * minibatch 
                # ensure we don't go out of the bounds of our training set:
                batch_end = min(batch_start + minibatch, train_size) 
                # ensure we don't have an index error
                actual_batch_size = batch_end-batch_start 
                
                # get training minibatch with all channnels and 2D feature dims
                X = X_train[batch_start:batch_end,:,:,:] 
                # get training minibatch labels 
                Y = y_train[batch_start:batch_end] 

                # instantiate training tensors
                X_tensor = torch.tensor(X, device=self.device).float() 
                Y_tensor = torch.tensor(Y, dtype=torch.long,device=self.device)
                
                # Pass input tensors thru 1 training step (fwd+backwards pass)
                loss, acc = train_step(X_tensor,Y_tensor) 
                
                # aggregate batch accuracy to measure progress of entire epoch
                epoch_acc += acc * actual_batch_size / train_size
                epoch_loss += loss * actual_batch_size / train_size
                
                # keep track of the iteration to see if the model's too slow
                print('\r'+f'Epoch {epoch+1}: iteration {i}/{num_iterations}',end='')
            
            # scheduler.step(epoch_loss)
            # create tensors from validation set
            X_valid_tensor = torch.tensor(X_valid,device=self.device).float()
            Y_valid_tensor = torch.tensor(y_valid,dtype=torch.long,device=self.device)
            
            # calculate validation metrics to keep track of progress; don't need predictions now
            valid_loss, valid_acc, _ = validate(X_valid_tensor,Y_valid_tensor)
            
            # accumulate scalar performance metrics at each epoch to track and plot later
            train_losses.append(epoch_loss)
            valid_losses.append(valid_loss)
            train_accuracy.append(epoch_acc.item())
            valid_accuracy.append(valid_acc.item())
            
            # Save checkpoint of the model
            checkpoint_filename = './Checkpoint/cnnModel-{:03d}.pkl'.format(epoch+1)
            save_checkpoint(optimizer, model, epoch, checkpoint_filename)
            
            # keep track of each epoch's progress
            print(f'\nEpoch {epoch+1} --- loss:{epoch_loss:.2f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.2f}, Validation accuracy:{valid_acc:.2f}%')
        print("Finished.")
        return train_losses, valid_losses, train_accuracy, valid_accuracy
    
    def train_single_epoch(self, model, optimizer, criterion):
        # define the training step of the training phase
        def train_step(X,Y):  
            # forward pass
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            
            # compute loss on logits because nn.CrossEntropyLoss implements log softmax
            loss = criterion(input=output_logits, target=Y)
            # loss = criterion(output_logits, Y) 
            
            # compute gradients for the optimizer to use 
            loss.backward()
            
            # update network parameters based on gradient stored (by calling loss.backward())
            optimizer.step()
            
            # zero out gradients for next pass
            # pytorch accumulates gradients from backwards passes (convenient for RNNs)
            optimizer.zero_grad() 
            return loss.item(), accuracy*100
        return train_step
    
    def validate_single_epoch(self, model, criterion):
        def validate(X,Y):
            # don't want to update any network parameters on validation passes: don't need gradient
            # wrap in torch.no_grad to save memory and compute in validation phase: 
            with torch.no_grad(): 
                # set model to validation phase i.e. turn off dropout and batchnorm layers 
                model.eval()
                # get the model's predictions on the validation set
                output_logits, output_softmax = model(X)
                predictions = torch.argmax(output_softmax,dim=1)
                # calculate the mean accuracy over the entire validation set
                accuracy = torch.sum(Y==predictions)/float(len(Y))
                # compute error from logits (nn.crossentropy implements softmax)
                loss = criterion(input=output_logits, target=Y)
                # loss = criterion(output_logits,Y)
            return loss.item(), accuracy*100, predictions
        return validate
    
    def make_save_checkpoint(self): 
        def save_checkpoint(optimizer, model, epoch, filename):
            checkpoint_dict = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint_dict, filename)
        return save_checkpoint
    
    def load_checkpoint(self, optimizer, model, filename):
        checkpoint_dict = torch.load(filename)
        epoch = checkpoint_dict['epoch']
        model.load_state_dict(checkpoint_dict['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
        return epoch

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
    
    # def feature_mfcc(waveform, sample_rate, winlen=512, window='hamming', n_mels=40, n_mfcc=20, hop_length=256):
    #     # Preprocessing: pad waveform with zeros to be at least 1 second long
    #     waveform = librosa.util.fix_length(waveform, size = sample_rate)
    #     # Compute MFCCs
    #     mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mels=n_mels, n_mfcc=n_mfcc,
    #                                 win_length=winlen, window=window, hop_length=hop_length)
    #     return mfcc
    def feature_mfcc(waveform, sample_rate, n_mfcc = 40, fft = 1024, winlen = 512, window='hamming', mels=128):
        mfc_coefficients=librosa.feature.mfcc(
                            y=waveform, 
                            sr=sample_rate, 
                            n_mfcc=n_mfcc,
                            n_fft=fft, 
                            win_length=winlen, 
                            window=window, 
                            #hop_length=hop, 
                            n_mels=mels, 
                            fmax=sample_rate/2) 
        return mfc_coefficients


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
        DURATION = 2
        
        dataset = CremaD(path=self.audio, sample_rate=SAMPLE_RATE, duration=DURATION, num_samples=NUM_SAMPLE).extract_audio_svm()
        X = dataset.drop(labels='Emotions', axis= 1)
        scaler = joblib.load('./Scaler/SVMScaler.joblib')
        x_scaled = scaler.transform(X)
        x_test = pd.DataFrame(x_scaled)

        prediction = loaded_model.predict_proba(x_test)
        prediction = load_model.reverse_label_encoder(prediction)
        return prediction   
