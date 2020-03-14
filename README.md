## Disaster Response Pipeline

We live in a highly connected world, where virtually everyone can reach a myriad of people in a matter of seconds. 
This becomes a challenge for relief organizations when disasters strike, as victims desperately send messages of 
all kinds in overwhelming numbers. 

Some of those messages may carry valuable information if they reach the appropiate responders and in many cases, 
they could even make the difference between life and death. But reading and classifying each and every message in 
order to forward it to the appropriate organization, would be a daunting task.

The purpose of this project is to build a model to classify text messages in a pre-defined set of categories. 
This classification should help in determining the type of information contained in the message and the organization 
that is better qualified to respond.

#### Methodology

With this idea in mind, we built and trained a supervised classification model using 
[Random Forests](https://en.wikipedia.org/wiki/Random_forest). We trained the model on a real dataset of 
categorized texts, provided by [Figure Eight](https://www.figure-eight.com/). After cleaning and normalizing the 
dataset, we used [Count Vectorization](https://en.wikipedia.org/wiki/Bag-of-words_model) and 
[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to transform the messages into numerical features.

We tunned the parameters of the model by using ScikitLearn's 
[GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search). 
This utility performs an exhaustive exploration of a specific set (grid) of parameter values. 
It finds the combination of parameters that yields the best results by applying 
[Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics).

As a final result, we assembled a web application that provides access to the model. This application allows 
users to predict the categories associated with a text message entered by the user. It displays some plots that
give basic overview of the training data.

#### Project structure

The source code is organized in three folders:

- Data: 

Contains the dataset of messages and categories that was used to train and test the model. 
Most importantly, it contains the script process_data.py, which extracts, cleans and merges the messages 
and categories provided in separate CSV files. Then, it loads the results into an SQLite database at 
the specified path. 

This script can be invoked by running the following command in the project's root directory:

```bash
python data/process_data.py MESSAGES_CSV_PATH CATEGORIES_CSV_PATH TARGET_DB_PATH
```

Arguments MESSAGES_CSV_PATH and CATEGORIES_CSV_PATH are the paths of the CSV files containing messages and categories, 
respectively. Argument TARGET_DB_PATH specifies the path where the script will write the resulting database. 
For example, to load the datasets provided in the project run:

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

- Models:






### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Make sure to download:

- nltk.download('stopwords')
- nltk.download('wordnet')
- nltk.download('punkt')

