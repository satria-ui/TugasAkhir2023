import os
import numpy as np
import gradio as gr
from datetime import date
import librosa
import torch
from Code.TransformerCnn import TransformerCNNNetwork
from Code.LeNet import LeNet
import pandas as pd
import pickle
import joblib

def reverse_label_encoder(data):
    mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
    reverse_mapping_dict = {v: k for k, v in mapping.items()}
    keys = list(reverse_mapping_dict.values())

    result = {f'{keys[i]}': data[0, i] for i in range(len(keys))}

    return result

def mfcc_formula(audio):
    y,sr = librosa.load(audio, duration=4, offset=0.5)
    n_fft = int(sr * 0.02)
    hop_length = n_fft // 2
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).T, axis=0)

    extracted_audio = pd.DataFrame(mfcc)
    return extracted_audio.T

def svm_prediction(input):
    loaded_model = pickle.load(open("./ML_Model/svm_model.sav", 'rb'))

    dataset = mfcc_formula(input)
    scaler = joblib.load('./Scaler/Z-ScoreScaler.joblib')
    x_scaled = scaler.transform(dataset)
    x_test = pd.DataFrame(x_scaled)

    prediction = loaded_model.predict_proba(x_test)
    prediction = reverse_label_encoder(prediction)
    return prediction

def LeNet_prediction(input):
    model_path = "./ML_Model_UI/Lenet_ravdess_model.pkl"
    scaler_path = './Scaler_UI/LenetScalerRAVDESS.joblib'
    ############################## LOAD MODEL ##############################
    ## instantiate empty model and populate with params from binary 
    model = LeNet().to("cuda")
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=0.001, momentum=0.8)
    load_checkpoint(optimizer, model, model_path)

    print(f'Loaded model from {model_path}')
    ############################## PRE-PROCESS AUDIO ##############################
    SAMPLE_RATE = 48000
    DURATION = 3.73

    features_test = []
    emotion_list = ['angry', 'fear', 'disgust', 'happy', 'neutral', 'sad']

    waveform, _ = librosa.load(input, duration=DURATION, sr=SAMPLE_RATE)
    waveform_homo = np.zeros((int(SAMPLE_RATE*DURATION)))
    waveform_homo[:len(waveform)] = waveform

    array_wave = np.array(waveform_homo, dtype=np.float64)
    m = 1  # number of rows
    n = array_wave.shape[0]  # number of columns
    X_2d = array_wave.reshape((m, n))
    print(X_2d.shape)
    features_test = get_features(X_2d, features_test, SAMPLE_RATE)

    XTest = np.expand_dims(features_test,1)

    scaler = joblib.load(scaler_path)
    BATCH,CHANNEL,WIDTH,HEIGHT = XTest.shape
    XTest = np.reshape(XTest, (BATCH,-1))
    XTest = scaler.transform(XTest)
    XTest = np.reshape(XTest, (BATCH,CHANNEL,WIDTH,HEIGHT))
    print(f'Shape of 4D feature array for input tensor: {XTest.shape}')

    # ############################## MAKE PREDICTION ##############################
    X_test_tensor = torch.tensor(XTest,device="cuda").float()
    output_prediction, output_softmax = make_prediction(model,X_test_tensor)

    output_dict = {}
    for i in range(len(emotion_list)):
        output_dict[emotion_list[i]] = output_softmax[0][i].item()
    print(output_dict)
    return output_dict

def TransformerCNN_prediction(input):
    model_path = "./ML_Model_UI/Transformer_ravdess_model.pkl"
    scaler_path = './Scaler_UI/TransformerCNNScalerRAVDESS.joblib'
    ############################## LOAD MODEL ##############################
    ## instantiate empty model and populate with params from binary 
    model = TransformerCNNNetwork().to("cpu")
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=0.001, momentum=0.8)
    load_checkpoint(optimizer, model, model_path)

    print(f'Loaded model from {model_path}')
    ############################## PRE-PROCESS AUDIO ##############################
    SAMPLE_RATE = 22050
    DURATION = 5.26

    features_test = []
    emotion_list = ['angry', 'fear', 'disgust', 'happy', 'neutral', 'sad']

    waveform, _ = librosa.load(input, duration=DURATION, sr=SAMPLE_RATE)
    waveform_homo = np.zeros((int(SAMPLE_RATE*DURATION)))
    waveform_homo[:len(waveform)] = waveform

    array_wave = np.array(waveform_homo, dtype=np.float64)
    m = 1  # number of rows
    n = array_wave.shape[0]  # number of columns
    X_2d = array_wave.reshape((m, n))
    print(X_2d.shape)
    features_test = get_features(X_2d, features_test, SAMPLE_RATE)

    XTest = np.expand_dims(features_test,1)

    scaler = joblib.load(scaler_path)
    BATCH,CHANNEL,WIDTH,HEIGHT = XTest.shape
    XTest = np.reshape(XTest, (BATCH,-1))
    XTest = scaler.transform(XTest)
    XTest = np.reshape(XTest, (BATCH,CHANNEL,WIDTH,HEIGHT))
    print(f'Shape of 4D feature array for input tensor: {XTest.shape}')

    # ############################## MAKE PREDICTION ##############################
    X_test_tensor = torch.tensor(XTest,device="cpu").float()
    output_prediction, output_softmax = make_prediction(model,X_test_tensor)

    output_dict = {}
    for i in range(len(emotion_list)):
        output_dict[emotion_list[i]] = output_softmax[0][i].item()
    print(output_dict)
    return output_dict

def main():
    today = date.today()
    tgl = today.strftime("%B %d, %Y")
    desc = (
        f"<p id='Name'>by Adam Satria</p><br><p>{tgl}</p>"
        )

    gr.Interface(
    TransformerCNN_prediction,
    inputs = [
        gr.Audio(source="upload", type="filepath"),
    ],
    outputs = [
        gr.Label(),
    ],
    examples=[
        [os.path.join(os.path.dirname(__file__),"audio/recording1.wav")],
        [os.path.join(os.path.dirname(__file__),"audio/recording2.wav")],
        [os.path.join(os.path.dirname(__file__),"audio/sad.wav")],
        [os.path.join(os.path.dirname(__file__),"audio/happy.wav")],
        [os.path.join(os.path.dirname(__file__),"audio/angry.wav")],
        [os.path.join(os.path.dirname(__file__),"audio/disgust.wav")],
        [os.path.join(os.path.dirname(__file__),"audio/angry-savee.wav")],
    ],
    title = "English Audio Emotion Recognition",
    description = desc,
    allow_flagging = "never",
    css = (
        "p{color: gray; text-align: center; font-size: 	1.2500em;}"
        "#Name{color: lightgrey; text-align: center; font-size: 1.875em; margin-bottom: 0px;}"
        "footer{visibility: hidden;}"
    )
    ).launch(share=True)

def feature_mfcc(waveform, sample_rate, n_mfcc = 40, fft = 1024, winlen = 512, window='hamming', mels=128):
    mfc_coefficients=librosa.feature.mfcc(
                        y=waveform, 
                        sr=sample_rate, 
                        n_mfcc=n_mfcc,
                        n_fft=fft, 
                        win_length=winlen, #fft//2
                        window=window, 
                        # hop_length=512, #winlen//4
                        n_mels=mels, 
                        fmax=sample_rate/2) 
    return mfc_coefficients

def get_features(waveforms, features, sample_rate):
    file_count = 0
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1

        print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')
    return features

def load_checkpoint( optimizer, model, filename):
    checkpoint_dict = torch.load(filename, map_location=torch.device("cpu"))
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

def make_prediction(model, data):
    with torch.no_grad(): 
        model.eval()
        _, output_softmax = model(data)
        predictions = torch.argmax(output_softmax,dim=1)
        print(output_softmax)
        return predictions, output_softmax

if __name__ == "__main__":
    main()