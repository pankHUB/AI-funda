import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Remove HTML tags and non-alphanumeric characters
    text = re.sub(r'<[^>]+>','',text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Convert to lowercase and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.lower() not in stop_words]

    # stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text


print(preprocess_text("<a>Get rich quick! Click here to claim your prize.</a>"))