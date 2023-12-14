# Regular expression
import re
# import string library for text preprocessing
import string

# Natural Language tookit
import nltk
import numpy as np
from keras.models import load_model
# import keras
from keras.preprocessing import text, sequence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# obtain additional stopwords from nltk

# download models for sentence tokenization and word tokenization
nltk.download('punkt')
# download stopwords
nltk.download('stopwords')
# database of English words and their semantic relationship
nltk.download('wordnet')

model = load_model('trained_nlp_model.h5')


# Convert text to lowercase
def convert_to_lower_case(text):
    return text.lower()


def remove_punctuation(text):
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)

    # Remove punctuation using the translation table
    text_without_punct = text.translate(translator)

    return text_without_punct


def tokenize(text):
    # Tokenization
    return word_tokenize(text)


# Remove stopwords
def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


def lemmatize(tokens):
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) if lemmatizer.lemmatize(word) is not None else word for word in tokens]


# Remove special characters and numbers
def remove_special_chars(tokens):
    return [re.sub('[^A-Za-z]+', '', word) for word in tokens]


# function to do preprocessing text
def preprocess_text(text):
    lower_text = convert_to_lower_case(text)  # converting to lowercase
    removed_punctuation = remove_punctuation(lower_text)  # removing punctuation
    tokens = tokenize(removed_punctuation)  # tokenize words
    tokens = remove_stop_words(tokens)  # removing stop words
    tokens = lemmatize(tokens)  # lemmatize tokens
    tokens = remove_special_chars(tokens)  # remove special characters from token

    preprocessed_text = ' '.join(tokens)  # final preprocessed text
    return preprocessed_text


def token_sequence(news_text):
    # constants
    MAX_FEATURES = 15000

    # Create a tokenizer to tokenize the words and create sequences of tokenized words
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(news_text)
    text_sequences = tokenizer.texts_to_sequences(news_text)

    padded_text = sequence.pad_sequences(text_sequences, maxlen=300)

    return padded_text


def fake_no_fake(prompt):
    preprocessed_text = preprocess_text(str(prompt))
    padded_data = token_sequence(preprocessed_text)

    pred = model.predict(padded_data)
    binary_pred = (pred > 0.005).astype(int)
    class_label = "fake" if binary_pred[0] == 1 else "not fake"
    print("Predicted Class Label:", class_label)
    return class_label
