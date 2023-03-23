from utils import CremaD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib

def manual_label_encoder(data):
    mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
    if type(data) == pd.core.series.Series:
        encoded_data = data.map(mapping)
    else:
        encoded_data = [mapping[i] for i in data]

    return encoded_data

def reverse_label_encoder(data):
    mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
    reverse_mapping_dict = {v: k for k, v in mapping.items()}
    return [reverse_mapping_dict[label] for label in data]

def main():
    SAMPLE_RATE = 16000
    NUM_SAMPLE = 22050
    DURATION = 5
    train_path = "../dataset/train_CREMA_85/"
    test_path = "../dataset/test_CREMA_15/"

    print("Extracting Audio...\n")
    train_data = CremaD(path=train_path, sample_rate=SAMPLE_RATE, duration=DURATION, num_samples=NUM_SAMPLE).extract_audio_svm()
    test_data = CremaD(path=test_path, sample_rate=SAMPLE_RATE, duration=DURATION, num_samples=NUM_SAMPLE).extract_audio_svm()
    print("Done")

    test_data['Emotions'] = manual_label_encoder(test_data['Emotions'])
    train_data['Emotions'] = manual_label_encoder(train_data['Emotions'])

    X_test = test_data.drop(labels='Emotions', axis= 1)
    y_test = test_data['Emotions']
    X_train = train_data.drop(labels='Emotions', axis= 1)
    y_train = train_data["Emotions"]

    print("Creating Scaler...")
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, '../Scaler/SVMScaler.joblib')
    print(f"Mean: {scaler.mean_}\n")
    print(f"Scale: {scaler.scale_}")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ############################## GridSearching ##############################
    # # {'C': 10, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}
    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 3, 5, 7, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,'scale', 'auto'],'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'probability':[True, False], 'class_weight': [None, 'balanced'],
    #           'shrinking': [True, False], 'degree': [2,3,4]}

    # grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
    # grid.fit(X_train_scaled,y_train)

    # print(" Results from Grid Search " )
    # print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
    # print("\n The best score across ALL searched params:\n",grid.best_score_)
    # print("\n The best parameters across ALL searched params:\n",grid.best_params_)
    
    # grid_predictions = grid.predict(X_test_scaled)
    
    # y_pred_str = reverse_label_encoder(grid_predictions)
    # y_test_str = reverse_label_encoder(y_test)
    
    # print(classification_report(y_test_str, y_pred_str))

    # print("The Model's Prediction ")
    # print("<<<===========================================>>>")
    # df = pd.DataFrame({'Actual': y_test_str, 'Predict': y_pred_str})
    # print(df.head(20))

    print("Training...")
    SVM_model = svm.SVC(C=10, gamma=0.01, kernel='rbf', probability=True)
    SVM_model.fit(X_train_scaled, y_train)
    print("Saving Model...")
    filename = '../ML_Model/svm_model.sav'
    pickle.dump(SVM_model, open(filename, 'wb'))
    print("Done.")

    y_pred=SVM_model.predict(X_test_scaled)
    
    y_pred_str = reverse_label_encoder(y_pred)
    y_test_str = reverse_label_encoder(y_test)

    report=classification_report(y_test_str, y_pred_str)
    accuracy=accuracy_score(y_true=y_test_str, y_pred=y_pred_str)

    print("\nModel Summary:\n")
    print("Model:{}    Accuracy: {:.2f}%".format(type(SVM_model).__name__ , accuracy*100))
    print(report)

    print("The Model's Prediction ")
    print("<<<===========================================>>>")
    df = pd.DataFrame({'Actual': y_test_str, 'Predict': y_pred_str})
    print(df[41:61])

if __name__ == '__main__':
    main()