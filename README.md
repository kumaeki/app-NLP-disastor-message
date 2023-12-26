# find the disaster category from a message

This is a app could classify message across 36 categories(water, electricity,hospitals ,etc, [here is the detail](#categories)).

When you inputs a message into the app, it will returns classification results for all the 36 categories.

## about the data

This is a NLP portfolio program, based on the message and category message got from <https://www.figure-eight.com/>

## about the code

It could run in a Azure Web App. you could find my sample App here. <http://app-nlp-disaster.azurewebsites.net/>

```text
- .vscode
    - settings.json  (setting file for vscode, if you want to deploy it to azure webapp, you will need it)
- data
    - disaster_categories.csv  (the data file1)
    - disaster_messages.csv  (the data file2)
    - KumaDB.db  (the DB file where save the processed data)
    - process_data.py  (the python file process the data)
- models
    - __init__.py
    - tokenize_kuma.py  (the python file define the tokenize function)
    - train_classifier.py  (the python file train the model)
- static  (static files)
    - bootstrap
        - css
            - ***
        - js
            - ***
- templates
    - go.html  (after you input the message, this page show the categories)
    - master.html  (the main page)
- .development  (setting file for deploying to azure webapp)
- .gitignore
- app.py  (main python file of this app)
- README.md
- requirements.txt
- tokenize_kuma.py  (same as models/tokenize_kuma.py, used by app.py when it load the trained model)
```

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

## categories

- related
- request
- offer
- aid_related
- medical_help
- medical_products
- search_and_rescue
- security
- military
- child_alone
- water
- food
- shelter  
- clothing
- money
- missing_people
- refugees
- death
- other_aid
- infrastructure_related
- transport
- buildings
- electricity
- tools
- hospitals
- shops
- aid_centers
- other_infrastructure
- weather_related  
- floods
- storm
- fire
- earthquake  
- cold
- other_weather
- direct_report