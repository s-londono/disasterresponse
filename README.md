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

With this idea in mind, we built and trained a supervised classification model using 
[Random Forests](https://en.wikipedia.org/wiki/Random_forest). We trained the model on a real dataset of 
categorized texts, provided by [Figure Eight](https://www.figure-eight.com/). After cleaning and normalizing the 
dataset, we used [Count Vectorization](https://en.wikipedia.org/wiki/Bag-of-words_model) and 
[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to transform the messages into numerical features.


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

