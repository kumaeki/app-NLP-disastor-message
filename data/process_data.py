import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    categories_split = categories["categories"].str.split(";", expand=True)
    row = categories_split.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories_split.columns = category_colnames

    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories_split[column] = pd.to_numeric(categories_split[column])

    categories = pd.concat([categories["id"], categories_split], axis=1)
    df = pd.merge(messages, categories, on="id").drop_duplicates()

    return df
    # pass


def clean_data(df):
    pass


def save_data(df, database_filename, table_name):
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(table_name, engine, index=False)
    pass


def main():
    if len(sys.argv) == 5:
        messages_filepath, categories_filepath, database_filepath, table_name = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        # df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath, table_name)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
