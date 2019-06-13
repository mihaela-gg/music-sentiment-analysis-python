# usual imports
import re
import string
import base64
# nltk imports
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def remove_encoded_characters(text):
    return text.encode("ascii", "ignore").decode("utf-8")


def remove_non_letters_digits(text):
    return re.sub(r' [^a-zA-Z]+', '', text)


def remove_punctuation(text):
    # words = word_tokenize(text, 'english')
    # new_text = ''
    # filtered_text = [word for word in words if not word in string.punctuation]
    # for word in filtered_text:
    #     new_text += word
    # return new_text
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    new_text = ''
    filtered_text = [word for word in words if not word in stop_words]

    for word in filtered_text:
        new_text += word + ' '

    return new_text


def stemming(text):
    ps = PorterStemmer()
    new_text = ''
    words = word_tokenize(text)

    for word in words:
        new_text += ps.stem(word) + ' '

    return new_text
    # words = [ps.stem(word) for word in text]
    # return words


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    new_text = " "
    words = word_tokenize(text)

    for word in words:
        word = lemmatizer.lemmatize(word)
        new_text += word + " "

    return new_text
    # words = [lemmatizer.lemmatize(word) for word in text]
    # return words


def pre_process_data_text(text):
    text = remove_encoded_characters(text)
    text = remove_punctuation(text)
    text = remove_non_letters_digits(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text
