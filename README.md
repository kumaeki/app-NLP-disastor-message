# find the disaster category from a message

This is a NLP portfolio program, based on the message and category message got from <https://www.figure-eight.com/>

It could run in a Azure Web App. you could find my sample App here. <http://app-nlp-disaster.azurewebsites.net/>

## how to run the process_data.py

run the command in the root folder

```zsh
python3 data/process_data.py {your_message_filepath} {your_category_filepath} {your_db_path} {your_table_name}
#python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/KumaDB.db KumaTable
```

## how to run the train_classifier.py

run the command in the root folder

```zsh
python3 data/train_classifier.py {your_db_path} {your_table_name} {your_model_path}
#python3 models/train_classifier.py data/KumaDB.db KumaTable models/model.joblib
```

## how to run the web app in local

create a .env file in root path, set these values as follow and run the command ```flask run``` in root path

```text
IS_LOAD_FROM_STORAGE=0
MODEL_BLOB_NAME={your_model_path} #models/model.joblib
DB_NAME={your_db_path} #data/KumaDB.db
TABLE_NAME={your_db_name} #KumaTable
```

if you have a Azure storage account, you could load the model file from there

```text
IS_LOAD_FROM_STORAGE=1
STORAGE_CONNECTION_STRING={your_storage_connect_str}
STORAGE_CONTAINER_NAME={your_container_name}
MODEL_BLOB_NAME={your_model_path}
DB_NAME={your_db_path}
TABLE_NAME={your_db_name}
```
