from utils import audio_extraction, data_loader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import pickle

dataset = data_loader("./dataset/").getData()
X_mfcc = dataset["Path"].apply(lambda x: audio_extraction(x).getMFCC())
print("First Data MFCC Values\n")
print(X_mfcc[0])

# X = [item for item in X_mfcc]
# X = np.array(X)
# y = dataset["Emotions"]
# extracted_audio = pd.DataFrame(X)
# extracted_audio["Emotions"] = y

# label_encoder = LabelEncoder()
# # 0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sad"
# extracted_audio['Emotions'] = label_encoder.fit_transform(extracted_audio['Emotions'])

# X = extracted_audio.drop(labels='Emotions', axis= 1)
# Y = extracted_audio['Emotions']

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(X_train)
# x_test = scaler.transform(X_test)

# # GridSearching
# # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

# # grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
# # grid.fit(x_train,y_train)


# SVM_model = svm.SVC(kernel='linear')
# SVM_model.fit(x_train, y_train)

# filename = 'SVM_Model.sav'
# pickle.dump(SVM_model, open(filename, 'wb'))

# y_pred=SVM_model.predict(x_test)

# report = classification_report(y_test, y_pred)
# accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

# print("\nModel:{}    Accuracy: {:.2f}%".format(type(SVM_model).__name__ , accuracy*100))
# print(report)

# print("The Model's Prediction ")
# print("<<<===========================================>>>")
# df = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
# print(df)