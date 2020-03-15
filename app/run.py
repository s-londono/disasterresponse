import json
import re
import plotly
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from visualization import PlotBuilder

app = Flask(__name__)

# Regex to match URLs
url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

# Lemmatizer based on the WordNet corpus
lemmatzr = WordNetLemmatizer()

stopwords = stopwords.words("english")


def tokenize(text):
    """
    Cleans, normalizes and converts the text into an array of lemmatized tokens
    :text: Piece to be tokenized
    :return: List of tokens corresponding to the lemmatized words of the text
    """
    # Replace URLs with a fixed placeholder
    clean_text = re.sub(url_regex, "urlplaceholder", text)

    # Make the text all lowercase and clean it of any remaining special characters
    clean_text = re.sub(r"[^0-9a-zA-Z]", " ", clean_text.lower())

    # Convert the text into a list of tokens, each corresponding to a word and remove heading and trailing spaces
    word_tokens = [word.strip() for word in word_tokenize(clean_text)]

    return [lemmatzr.lemmatize(word) for word in word_tokens if word not in stopwords]


# Load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# Instantiate a PlotBuilder and set the datasource to generate plots
plot_builder = PlotBuilder(df)

# Load model
model = joblib.load("models/classifier.pkl")


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Create visuals by using the PlotBuilder
    graphs = [
        plot_builder.build_genre_totals_bar(),
        plot_builder.build_category_totals_bar(),
        plot_builder.build_message_length_box()
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
