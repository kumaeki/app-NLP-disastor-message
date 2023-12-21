import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def tokenize_kuma(text):
    # normalization
    text = re.sub(r"https?://\S+|[^a-zA-Z0-9]", " ", text)

    # tokenization
    tokens = word_tokenize(text)

    # remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
