import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
from sklearn.ensemble import VotingClassifier
import joblib


# from chefboost import Chefboost as chef

# Function importing Dataset



def importdata():
    rawdata = pd.read_csv(r'../Model/DSL-StrongPasswordData.csv')
    # print(rawdata.head())
    # data = pd.DataFrame()
    # subject,sessionIndex,rep,H.period,DD.period.t,UD.period.t,H.t,DD.t.i,UD.t.i,H.i,DD.i.e,UD.i.e,H.e,DD.e.five,UD.e.five,H.five,DD.five.Shift.r,UD.five.Shift.r,H.Shift.r,DD.Shift.r.o,UD.Shift.r.o,H.o,DD.o.a,UD.o.a,H.a,DD.a.n,UD.a.n,H.n,DD.n.l,UD.n.l,H.l,DD.l.Return,UD.l.Return,H.Return
    # data = pd.DataFrame()
    data = rawdata.drop(['sessionIndex', 'rep', ], axis=1)
    # data = rawdata[['subject', 'H.period', 'H.t', 'H.i', 'H.e', 'H.five', 'H.Shift.r', 'H.o', 'H.a', 'H.n', 'H.l', 'H.Return']]
    # print (data.head())
    # print("Mean: ", data.mean(axis=0))
    # print("Standard Deviation: ", data.std(axis=0))
    # print(data.shape)
    # pd.set_option('display.max_columns', 500)
    # print(data.describe())
    # data.hist()
    # data.plot(kind='density',subplots=True,sharex=False)
    # plt.show()
    return data


#1 Function to perform training with giniIndex.
def train_using_gini(x_train, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(x_train, y_train)
    filename = 'clf_gini'
    joblib.dump(clf_gini, filename)
    return clf_gini


#2 Function to perform training with entropy.
def tarin_using_entropy(x_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)
    # Performing training
    clf_entropy.fit(x_train, y_train)
    filename = 'clf_entropy'
    joblib.dump(clf_entropy, filename)
    return clf_entropy


#3 Function to perform training with decision tree
def train_usign_decicionTree(x_train, y_train):
    clf_decisionTree = DecisionTreeClassifier()
    clf_decisionTree = clf_decisionTree.fit(x_train, y_train)
    filename = 'clf_decisionTree'
    joblib.dump(clf_decisionTree, filename)
    return clf_decisionTree


#4 Function to perform training with DecisionTreeRegressor
def train_usign_decicionTreeRgress(x_train, y_train):
    clf_decisionTreeRegress = DecisionTreeRegressor()
    clf_decisionTreeRegress = clf_decisionTreeRegress.fit(x_train, y_train)
    filename = 'clf_decisionTreeRegress'
    joblib.dump(clf_decisionTreeRegress, filename)
    return clf_decisionTreeRegress


#5 Function to perform training with decision tree
def train_usign_KNeighborsClassifier(x_train, y_train):
    clf_neigh = KNeighborsClassifier(n_neighbors=1)
    clf_neigh = clf_neigh.fit(x_train, y_train)
    filename = 'clf_neigh'
    joblib.dump(clf_neigh, filename)
    return clf_neigh


#6 Function to perform training with Support Vector Machines
def train_usign_svm(x_train, y_train):
    clf_svm = svm.SVC()
    clf_svm = clf_svm.fit(x_train, y_train)
    filename = 'clf_svm'
    joblib.dump(clf_svm, filename)
    return clf_svm

#7 Function to test K Neighbors Classifier k value
def optimize_k_KNeighborsClassifier(x_train, x_test, y_train, y_test):
    accuracyTest = []
    accuracyTrain = []
    maxK = 81
    # for i in range(1, maxK):
    #     print (i)
    #     knn = KNeighborsClassifier(n_neighbors=i, )
    #     knn.fit(x_train, y_train)
    #     pred_i = knn.predict(x_test)
    #     accuracyTest.append(metrics.accuracy_score(y_test, pred_i)*100)
    #     # print(pred_i)
    #     # print(y_test)
    #     # print(metrics.accuracy_score(y_test, pred_i))
    #     pred_i = knn.predict(x_train)
    #     accuracyTrain.append(metrics.accuracy_score(y_train, pred_i)*100)
    #     # error.append(np.mean(pred_i != y_test))
    # print(accuracyTest)
    # print(accuracyTrain)
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 80), accuracyTest, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Testing Accuracy K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('accuracy')
    # plt.show()
    accuracyTest = [1.0947712418300652, 1.2581699346405228, 1.1764705882352942, 1.2091503267973858, 1.3398692810457515,
                    1.4869281045751632, 1.5849673202614378, 1.715686274509804, 1.7483660130718954, 1.6339869281045754,
                    1.7647058823529411, 1.8627450980392157, 1.9281045751633987, 1.9444444444444444, 1.8627450980392157,
                    1.8790849673202614, 1.8300653594771243, 1.7647058823529411, 1.8137254901960786, 1.9281045751633987,
                    2.107843137254902, 2.3039215686274512, 2.6143790849673203, 3.022875816993464, 3.496732026143791,
                    4.330065359477124,
                    5.1797385620915035, 6.2745098039215685, 7.630718954248366, 9.330065359477125, 10.506535947712418,
                    12.401960784313726, 13.49673202614379, 15.081699346405228, 15.702614379084967, 15.42483660130719,
                    14.722222222222223, 14.248366013071895, 13.071895424836603, 12.124183006535947, 11.470588235294118,
                    10.473856209150327, 9.558823529411764, 8.970588235294118, 8.235294117647058, 7.549019607843137,
                    7.140522875816993,
                    6.73202614379085, 6.2254901960784315, 5.9640522875816995, 5.604575163398693, 5.245098039215686,
                    5.049019607843137,
                    4.673202614379085, 4.477124183006536, 4.23202614379085, 4.101307189542483, 3.8562091503267975,
                    3.6111111111111107,
                    3.480392156862745, 3.3823529411764706, 3.316993464052288, 3.1209150326797386, 3.104575163398693,
                    2.9575163398692808, 2.7450980392156863, 2.630718954248366, 2.5653594771241828, 2.4836601307189543,
                    2.3856209150326797, 2.2549019607843137, 2.238562091503268, 2.173202614379085, 2.091503267973856,
                    2.07516339869281,
                    2.07516339869281, 2.026143790849673, 1.9607843137254901, 1.9281045751633987, 1.8790849673202614]
    accuracyTrain = [100.0, 47.51400560224089, 31.98879551820728, 24.530812324929972, 20.23109243697479,
                     17.54201680672269, 15.749299719887954, 14.404761904761903, 13.38935574229692, 12.829131652661063,
                     12.282913165266105, 11.680672268907562, 11.183473389355743, 10.784313725490197, 10.490196078431373,
                     10.175070028011204, 9.873949579831933, 9.6218487394958, 9.57983193277311, 9.586834733893559,
                     9.642857142857144, 9.957983193277311, 9.971988795518207, 10.609243697478991, 11.694677871148459,
                     12.885154061624648, 14.481792717086837, 16.778711484593835, 19.327731092436977, 22.38795518207283,
                     25.609243697478988, 29.054621848739497, 32.53501400560224, 35.16106442577031, 36.09243697478991,
                     36.17647058823529, 35.44117647058824, 34.397759103641455, 33.60644257703081, 32.28991596638656,
                     31.07843137254902, 29.565826330532214, 28.42436974789916, 27.33893557422969, 26.176470588235297,
                     25.34313725490196, 24.327731092436974, 23.515406162464984, 22.780112044817926, 21.96078431372549,
                     21.288515406162464, 20.53921568627451, 19.92997198879552, 19.28571428571429, 18.80252100840336,
                     18.22128851540616, 17.69607843137255, 17.170868347338935, 16.72268907563025, 16.34453781512605,
                     16.029411764705884, 15.749299719887954, 15.301120448179272, 14.873949579831933, 14.523809523809526,
                     14.180672268907562, 13.914565826330533, 13.571428571428571, 13.305322128851541, 13.011204481792719,
                     12.76610644257703, 12.556022408963585, 12.296918767507002, 12.051820728291316, 11.855742296918766,
                     11.568627450980392, 11.323529411764707, 11.141456582633054, 10.945378151260504, 10.686274509803921]
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, maxK), accuracyTest, 'ro', linestyle='dashed', label="test accuracy")
    plt.plot(range(1, maxK), accuracyTrain, 'bo', linestyle='dashed', label="Training accuracy")

    plt.title('Training and Testing Accuracy with k value')
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(np.arange(0, maxK, 2))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 80), accuracyTrain, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Testing Accuracy K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('accuracy')
    # plt.show()





# Function to make predictions
def prediction(X_test, clf):
    # Predicton on test with giniIndex
    loaded_model = joblib.load(clf)
    # y_pred = clf_object.predict(X_test)
    y_pred = loaded_model.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    # print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    # print ("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    print(accuracy)
    # print("Report : ", classification_report(y_test, y_pred))
    return accuracy


def show_result(x_train, x_test, y_train, y_test, clf):
    print("Training accuracy")
    y_pred = prediction(x_train, clf)
    accuracy_train = cal_accuracy(y_train, y_pred)
    print("\n Test accuracy")
    y_pred = prediction(x_test, clf)
    accuracy_test = cal_accuracy(y_test, y_pred)
    return [accuracy_train, accuracy_test]


def compareAlgo(x_train, x_test, y_train, y_test):
    # X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(x_train, y_train)
    clf_entropy = tarin_using_entropy(x_train, y_train)
    clf_decisionTree = train_usign_decicionTree(x_train, y_train)
    clf_neigh = train_usign_KNeighborsClassifier(x_train, y_train)
    clf_svm = train_usign_svm(x_train, y_train)

    # find_k_KNeighborsClassifier(x_train, x_test, y_train, y_test)

    accuracy_train = []
    accuracy_test = []
    # print("Real values:")
    # print(y_test)

    classifireList = ['Gini', 'Entropy', 'DecisionTree', 'KNeighbors', 'SVM']
    # for i in range(0, 3):
    #     print("\n\n", classifireList[i])
    #     accuracy = show_result(x_train, x_test, y_train, y_test, clf_gini)
    #     accuracy_train.append(accuracy[0])
    #     accuracy_test.append(accuracy[1])

    print("\n\n clf_gini")
    accuracy = show_result(x_train, x_test, y_train, y_test, clf_gini)
    accuracy_train.append(accuracy[0])
    accuracy_test.append(accuracy[1])

    print("\n\n clf_entropy")
    accuracy = show_result(x_train, x_test, y_train, y_test, clf_entropy)
    accuracy_train.append(accuracy[0])
    accuracy_test.append(accuracy[1])

    print("\n\n clf_decisionTree")
    accuracy = show_result(x_train, x_test, y_train, y_test, clf_decisionTree)
    accuracy_train.append(accuracy[0])
    accuracy_test.append(accuracy[1])

    print("\n\n clf_KNeighborsClassifier")
    accuracy = show_result(x_train, x_test, y_train, y_test, clf_neigh)
    accuracy_train.append(accuracy[0])
    accuracy_test.append(accuracy[1])

    print("\n\n clf_Support Vector Machines")
    accuracy = show_result(x_train, x_test, y_train, y_test, clf_svm)
    accuracy_train.append(accuracy[0])
    accuracy_test.append(accuracy[1])

    #
    print('accuracy_test= ', accuracy_test)
    print('accuracy_train= ', accuracy_train)

    plt.figure(figsize=(12, 6))

    # plt.plot(classifireList, accuracy_test, 'ro' , classifireList, accuracy_train, 'go', linestyle='dashed')
    plt.plot(classifireList, accuracy_test, 'ro', linestyle='dashed', label="test accuracy")
    plt.plot(classifireList, accuracy_train, 'go', linestyle='dashed', label="training accuracy")

    plt.title('Training and Testing Accuracy for hold time attributes')
    plt.xlabel('Classifire')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.yticks(np.arange(0, 101, 5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

# function to combine classifier
def combine_clfs(x_train, y_train):
    clf_decisionTree = train_usign_decicionTree(x_train, y_train)
    clf_neigh = train_usign_KNeighborsClassifier(x_train, y_train)

    ensemble = VotingClassifier(estimators=[('Decision Tree', clf_decisionTree), ('KNN', clf_neigh)], voting='soft',
                                weights=[2, 1])
    eclf = ensemble.fit(x_train, y_train)
    # save the model to disk
    filename = 'clf_combine'
    joblib.dump(eclf, filename)
    return eclf


# create model request
def createModel():
    data = importdata()
    y = data.subject
    x = data.drop('subject', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # clf_gini = train_using_gini(x_train, y_train)
    # clf_entropy= tarin_using_entropy(x_train, y_train)
    # clf_decisionTree = train_usign_decicionTree()
    # clf_decisionTreeRegress = train_usign_decicionTreeRgress(x_train, y_train)
    # clf_neigh = train_usign_KNeighborsClassifier(x_train, y_train)
    # clf_svm = train_usign_svm(x_train, y_train)

    clf_combine = combine_clfs(x_train, y_train)


# Driver code
def main():
    data = importdata()
    y = data.subject
    x = data.drop('subject', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

    print (data.head())
    # print("Mean: ", data.mean(axis=0))
    print(data.shape)
    # pd.set_option('display.max_columns', 500)
    # print(data.describe())

    print("x details")

    print(x_test.head())
    print(x_test.shape)


    # print(x_train.head())
    # print(x_test.head())
    # print(y_train.shape)
    # compareAlgo(x_train, x_test, y_train, y_test)
    # clf_svm = train_usign_svm(x_train, y_train)
    # clf_neigh = train_usign_KNeighborsClassifier(x_train, y_train)
    # show_result(x_train, x_test, y_train, y_test, clf_neigh)
    # optimize_k_KNeighborsClassifier(x_train, x_test, y_train, y_test)
    # clf_combine = combine_clfs(x_train, y_train)
    # show_result(x_train, x_test, y_train, y_test, 'clf_combine')
    # print("classify")


# Calling main function
# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    print("classify is being run directly")
    main()
else:
    print("classify is being imported")
