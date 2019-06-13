from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


def MNB(x_train, y_train, x_test, y_test):
    predmnb = predictMNB(x_train, y_train, x_test)
    print("Confusion Matrix for MNB: ")
    print(confusion_matrix(y_test, predmnb))
    print("Score:", round(accuracy_score(y_test, predmnb) * 100, 2))
    print("Classification report:\n", classification_report(y_test, predmnb))


def SVM(x_train, y_train, x_test, y_test):
    svm = SVC(random_state=101)
    svm.fit(x_train, y_train)
    predsvm = svm.predict(x_test)
    print("Confusion Matrix for SVM: ")
    print(confusion_matrix(y_test, predsvm))
    print("Score:", round(accuracy_score(y_test, predsvm) * 100, 2))
    print("Classification report:\n", classification_report(y_test, predsvm))


def DTC(x_train, y_train, x_test, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    preddtc = dtc.predict(x_test)
    print("Confusion Matrix for DTC: ")
    print(confusion_matrix(y_test, preddtc))
    print("Score:", round(accuracy_score(y_test, preddtc) * 100, 2))
    print("Classification report:\n", classification_report(y_test, preddtc))


def predictMNB(x_train, y_train, text):
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    prediction = mnb.predict(text)
    return prediction


def predictSVM(x_train, y_train, text):
    svm = SVC(random_state=101)
    svm.fit(x_train, y_train)
    prediction = svm.predict(text)
    return prediction