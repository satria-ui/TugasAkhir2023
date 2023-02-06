from utils import audio_extraction
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import pickle
from joblib import dump

def manual_label_encoder(data):
    mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
    if type(data) == pd.core.series.Series:
        encoded_data = data.map(mapping)
    else:
        encoded_data = [mapping[i] for i in data]

    return encoded_data

def main():
    path = "../dataset/"
    print("Extracting Audio...")
    dataset = audio_extraction(path = path).extract_audio()
    print("Done")

    dataset['Emotions'] = manual_label_encoder(dataset['Emotions'])

    X = dataset.drop(labels='Emotions', axis= 1)
    Y = dataset['Emotions']

    print("Creating Scaler...")
    scaler = StandardScaler().fit(X)
    dump(scaler, './Scaler/Z-ScoreScaler.joblib')
    print(f"Mean: {scaler.mean_}\n")
    print(f"Scale: {scaler.scale_}")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    # # GridSearching
    # # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

    # # grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
    # # grid.fit(x_train,y_train)

    print("Training...")
    SVM_model = svm.SVC(kernel='linear')
    SVM_model.fit(x_train_scaled, y_train)
    print("Saving Model...")
    filename = '../ML_Model/svm_model.sav'
    pickle.dump(SVM_model, open(filename, 'wb'))
    print("Done.")

    y_pred=SVM_model.predict(x_test_scaled)

    report=classification_report(y_test, y_pred)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

    print("\nModel Summary:\n")
    print("Model:{}    Accuracy: {:.2f}%".format(type(SVM_model).__name__ , accuracy*100))
    print(report)

    print("The Model's Prediction ")
    print("<<<===========================================>>>")
    df = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
    print(df.head(10))

if __name__ == '__main__':
    main()