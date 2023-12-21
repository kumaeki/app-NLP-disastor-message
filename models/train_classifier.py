# import libraries
# import pickle
# import re
# import ssl
import sys

# import nltk
import pandas as pd
from joblib import dump

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from tokenize_kuma import tokenize_kuma


def load_data(database_filepath, table_name):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name, engine)
    X = df[["message"]]
    y = df.drop(labels=["id", "message", "original", "genre"], axis=1)
    return X, y


# def tokenize(text):
#     # normalization
#     text = re.sub(r"https?://\S+|[^a-zA-Z0-9]", " ", text)

#     # tokenization
#     tokens = word_tokenize(text)

#     # remove stop words
#     tokens = [t for t in tokens if t not in stopwords.words("english")]

#     lemmatizer = WordNetLemmatizer()
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#     return clean_tokens


def build_model():
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize_kuma, token_pattern=None)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    # TAKE TOO LONG!!!!!!
    # parameters = {
    #     "vect__ngram_range": ((1, 1), (1, 2)),
    #     "clf__estimator__n_estimators": [50, 100, 150, 200],
    #     "clf__estimator__min_samples_split": [2, 3, 4],
    # }
    # return GridSearchCV(pipeline, param_grid=parameters)

    parameters = {
        "vect__ngram_range": (1, 1),
        "clf__estimator__n_estimators": 100,
        "clf__estimator__min_samples_split": 2,
    }

    pipeline.set_params(**parameters)
    return pipeline


def train_model(model, X_train, Y_train):
    model.fit(X_train["message"].values, Y_train)


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test["message"].values)
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:\n", accuracy)


def save_model(model, model_filepath):
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 4:
        database_filepath, table_name, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y = load_data(database_filepath, table_name)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        train_model(model, X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
