from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


def MNB(x_train, y_train, x_test, y_test):
    predmnb = predictMNB(x_train, y_train, x_test)
    confusionMatrix = confusion_matrix(y_test, predmnb)
    accuracy = round(accuracy_score(y_test, predmnb) * 100, 2)
    classificationReport = classification_report(y_test, predmnb)

    print("Confusion Matrix for MNB: ")
    print(confusionMatrix)
    print("Score:", accuracy)
    print("Classification report:\n", classificationReport)

    result = "Confusion Matrix for MNB:\n" + confusionMatrix.__str__() + "\n\n"
    result += "Score: " + accuracy.__str__() + "\n\n"
    result += "Classification report:\n" + classificationReport.__str__() + "\n"
    return result


def SVM(x_train, y_train, x_test, y_test):
    svm = SVC(random_state=101)
    svm.fit(x_train, y_train)
    predsvm = svm.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, predsvm)
    accuracy = round(accuracy_score(y_test, predsvm) * 100, 2)
    classificationReport = classification_report(y_test, predsvm)

    print("Confusion Matrix for SVM: ")
    print(confusionMatrix)
    print("Score:", accuracy)
    print("Classification report:\n", classificationReport)

    result = "Confusion Matrix for SVM:\n" + confusionMatrix.__str__() + "\n\n"
    result += "Score: " + accuracy.__str__() + "\n\n"
    result += "Classification report:\n" + classificationReport.__str__() + "\n"
    return result


def DTC(x_train, y_train, x_test, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    preddtc = dtc.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, preddtc)
    accuracy = round(accuracy_score(y_test, preddtc) * 100, 2)
    classificationReport = classification_report(y_test, preddtc)

    print("Confusion Matrix for DTC: ")
    print(confusionMatrix)
    print("Score:", accuracy)
    print("Classification report:\n", classificationReport)

    result = "Confusion Matrix for DTC:\n" + confusionMatrix.__str__() + "\n\n"
    result += "Score: " + accuracy.__str__() + "\n\n"
    result += "Classification report:\n" + classificationReport.__str__() + "\n"
    return result


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