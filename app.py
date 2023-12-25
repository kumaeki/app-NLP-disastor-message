import json
import os
import io
import pandas as pd
import plotly
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from flask import Flask, render_template, request
from joblib import load
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine
from tokenize_kuma import tokenize_kuma

app = Flask(__name__)


def load_data():
    """
    load dataframe from db
    the parameters is setted in .env
    :return: the dataframe
    """
    db_name = os.getenv("DB_NAME")
    table_name = os.getenv("TABLE_NAME")
    engine = create_engine(f"sqlite:///{db_name}")
    df = pd.read_sql_table(table_name, engine)
    return df


def load_model():
    """
    load model from Azure storage account blob
    the parameters is setted in .env

    :return: the trained model
    """
    load_dotenv()
    is_load_from_storage = os.getenv("IS_LOAD_FROM_STORAGE")
    if is_load_from_storage == 1:
        connect_str = os.getenv("STORAGE_CONNECTION_STRING")
        container_name = os.getenv("STORAGE_CONTAINER_NAME")
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # Get a client to interact with a specific blob
        blob_name = os.getenv("MODEL_BLOB_NAME")
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download the blob content into a BytesIO buffer
        blob_stream = io.BytesIO()
        blob_client.download_blob().readinto(blob_stream)

        # Seek to the beginning of the stream
        blob_stream.seek(0)

        # Load your model using joblib
        model = load(blob_stream)
    else:
        model_path = os.getenv("MODEL_BLOB_NAME")
        model = load(model_path)

    return model


df = load_data()
model = load_model()


@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    df_category = df.drop(labels=["id", "message", "original", "genre"], axis=1)

    category_counts = df_category.sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    corr_matrix = df_category.corr()
    corr_flat = corr_matrix.stack().reset_index()
    corr_flat.columns = ["Category1", "Category2", "Correlation"]
    corr_flat = corr_flat[corr_flat["Category1"] != corr_flat["Category2"]]
    corr_flat["AbsCorrelation"] = corr_flat["Correlation"].abs()
    top_correlations = corr_flat.sort_values(by="AbsCorrelation", ascending=False)
    top_10_correlations = top_correlations.head(10)
    top_10_correlations["Pair"] = top_10_correlations.apply(
        lambda row: f"{row['Category1']} & {row['Category2']}", axis=1
    )

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=category_names, y=category_counts)],
            "layout": {
                "title": "Distribution of Message Category",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "category"},
            },
        },
        {
            "data": [
                Heatmap(
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.columns.tolist(),
                    z=corr_matrix.values,
                    colorscale="RdBu",
                )
            ],
            "layout": {
                "title": "Correlation Heatmap of Categories",
                "yaxis": {"title": "Category"},
                "xaxis": {"title": "Category"},
            },
        },
        {
            "data": [
                Bar(
                    x=top_10_correlations["Pair"],
                    y=top_10_correlations["Correlation"],
                    marker_color="blue",
                )
            ],
            "layout": {
                "title": "Top 10 Most Correlated Category Pairs",
                "xaxis": {"title": "Category Pairs"},
                "yaxis": {"title": "Correlation Coefficient", "range": [0, 1]},
            },
        },
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
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
