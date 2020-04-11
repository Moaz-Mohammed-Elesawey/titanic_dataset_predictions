import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  #, AdaBoostClassifier
from sklearn import svm
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.metrics import accuracy_score


def change_sex(x):
    if x == 'male':
        return 0
    elif x == 'female':
        return 1


def change_Embark(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    elif x == 'Q':
        return 2


def get_train_data_ready(file_name='data/train.csv'):
    data = pd.read_csv(file_name).copy()
    data.set_index('PassengerId', inplace=True)
    data.fillna(0, inplace=True)
    data.drop(["Name", 'Cabin', 'Ticket'], axis=1, inplace=True)
    data['Age'] = data['Age'] / 80.000
    data['Fare'] = data['Fare'] / 512.329200
    data['Sex'] = data['Sex'].apply(change_sex)
    data['Embarked'] = data['Embarked'].apply(change_Embark)
    survived = np.array(data.pop('Survived'))
    data = np.array(data)
    return data, survived

X, Y = get_train_data_ready()



X = np.nan_to_num(X)
Y = np.nan_to_num(Y)

def get_test_data_ready(file_name='data/test.csv'):
    data = pd.read_csv(file_name).copy()
    data.set_index('PassengerId', inplace=True)
    data.fillna(0, inplace=True)
    data.drop(["Name", 'Cabin', 'Ticket'], axis=1, inplace=True)
    data['Age'] = data['Age'] / 80.000
    data['Fare'] = data['Fare'] / 512.329200
    data['Sex'] = data['Sex'].apply(change_sex)
    data['Embarked'] = data['Embarked'].apply(change_Embark)
    data = np.array(data)
    return data

test_data = get_test_data_ready()

print(X.shape, Y.shape, test_data.shape)


#######  RandomForestClassifier  ########
RF_clf = RandomForestClassifier(n_estimators=10, max_depth=6, n_jobs=1, verbose=0)
RF_clf.fit(X, Y)
RF_predictions = RF_clf.predict(test_data)


####### svm.SVC ####
SVC_clf = svm.SVC()
SVC_clf.fit(X, Y)
SVC_predictions = SVC_clf.predict(test_data)


output_1 = pd.DataFrame({"PassengerId":np.arange(892, 1310),
                         "Survived":RF_predictions
                        }).to_csv('./predictions/RF_predictions.csv', index=False)

output_2 = pd.DataFrame({"PassengerId":np.arange(892, 1310),
                         "Survived":SVC_predictions
                        }).to_csv('./predictions/SVC_predictions.csv', index=False)
