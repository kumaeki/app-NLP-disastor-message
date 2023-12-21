import json

import pandas as pd
import plotly
from flask import Flask, render_template, request
from joblib import load
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from tokenize_kuma import tokenize_kuma

app = Flask(__name__)


# load data
engine = create_engine("sqlite:///data/KumaDB.db")
df = pd.read_sql_table("KumaTable", engine)

# load model
model = load("models/model.joblib")


@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template("go.html", query=query, classification_result=classification_results)


if __name__ == "__main__":
    app.run()
