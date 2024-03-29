from src.reader_writer import read_csv
from src.pre_processing import pre_process_data_text
from src.ml_algorithms import MNB, SVM, DTC, predictMNB, predictSVM, predictDTC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

training_data = read_csv('resources/Train.csv')
test_data = read_csv('resources/Test.csv')

x_train = training_data['Lyrics']
y_train = training_data['Sentiment']
x_test = test_data['Lyrics']
y_test = test_data['Sentiment']

for i in range(0, x_train.__len__()) :
    x_train[i] = pre_process_data_text(x_train[i])

for i in range(0, x_test.__len__()) :
    x_test[i] = pre_process_data_text(x_test[i])

vectorizer = TfidfVectorizer().fit(x_train)
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)


def predictMNBrunnerText(text):
    text = pre_process_data_text(text)
    print(text)
    vectorText = vectorizer.transform([text])
    return predictMNB(x_train, y_train, vectorText)


def predictMNBrunnerVector(vector):
    for i in range(0, vector.__len__()):
        vector[i] = pre_process_data_text(vector[i])
    vectorText = vectorizer.transform(vector)
    return predictMNB(x_train, y_train, vectorText)

