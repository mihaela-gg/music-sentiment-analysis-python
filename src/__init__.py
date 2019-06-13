from src.reader_writer import read_csv
from src.pre_processing import pre_process_data_text
from src.ml_algorithms import MNB, SVM, DTC, predictMNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    training_data = read_csv('resources/Train.csv')
    test_data = read_csv('resources/Test.csv')

    x_train = training_data['Lyrics']
    y_train = training_data['Sentiment']
    x_test = test_data['Lyrics']
    y_test = test_data['Sentiment']

    # print(x_train)
    x_train = x_train.copy()
    x_test = x_test.copy()

    for i in range(0, x_train.__len__()):
        x_train[i] = pre_process_data_text(x_train[i])

    # print(x_train)

    for i in range(0, x_test.__len__()):
        x_test[i] = pre_process_data_text(x_test[i])

    # vocab = CountVectorizer().fit(x_train)
    # x_train = vocab.transform(x_train)
    # x_test = vocab.transform(x_test)

    vectorizer = TfidfVectorizer().fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

    MNB(x_train, y_train, x_test, y_test)
    print("-----------------------------")
    SVM(x_train, y_train, x_test, y_test)
    print("-----------------------------")
    DTC(x_train, y_train, x_test, y_test)

