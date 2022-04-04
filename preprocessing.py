from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re



stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    text = str(text)
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    #removes URLS
    text = re.sub('http\S+', ' ', text)
    #removes special chars
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    #removes non ASCII
    text = re.sub('[^\x00-\x7F]+', ' ', text)
    #removes numbers
    text = re.sub('\d+', ' ', text)
    #lemmaziter
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    #removes stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    #removes double spaces
    text = re.sub(' +', ' ', text)
    return text

