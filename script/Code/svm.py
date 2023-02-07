from utils import audio_extraction
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import pickle

def main():
    path = "../dataset/"
    print("Extracting Audio...")
    dataset = audio_extraction(path = path).extract_audio()
    print("Done")

    label_encoder = LabelEncoder()
    dataset['Emotions'] = label_encoder.fit_transform(dataset['Emotions'])
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label Mapping:\n")
    print(le_name_mapping)

    X = dataset.drop(labels='Emotions', axis= 1)
    Y = dataset['Emotions']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)

    # GridSearching
    param_grid = {'C': [0.1,1,10], 'gamma': [1,0.1,0.01,0.001,'scale', 'auto'],'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'probability':[True, False]}

    grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
    grid.fit(x_train,y_train)
    
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid.best_params_)

    # print("Training...")
    # SVM_model = svm.SVC(kernel='linear')
    # SVM_model.fit(x_train, y_train)
    # print("Saving Model...")
    # filename = '../ML_Model/SVM_Model.sav'
    # pickle.dump(SVM_model, open(filename, 'wb'))
    # print("Done.")

    # y_pred=SVM_model.predict(x_test)

    # report=classification_report(y_test, y_pred)
    # accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

    # print("Model Summary:\n")
    # print("Model:{}    Accuracy: {:.2f}%".format(type(SVM_model).__name__ , accuracy*100))
    # print(report)

    # print("The Model's Prediction ")
    # print("<<<===========================================>>>")
    # df = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
    # print(df.head(10))

if __name__ == '__main__':
    main()