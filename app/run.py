import joblib
import json
import plotly
import pandas as pd

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram, Pie
from sqlalchemy import create_engine

from models.train_classifier import tokenize


app = Flask(__name__)

# load data
sql_engine = create_engine('sqlite:///../data/etl.db')
df = pd.read_sql_table('data', sql_engine.connect())

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    related_messages = df.groupby('related').count()['message']
    messages_len = df['message'].str.len()
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=True)

    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_counts.index,
                    values=genre_counts,
                    marker={'colors': [
                         'rgb(112, 141, 219)',
                         'rgb(176, 197, 255)',
                         'rgb(42, 85, 201)'
                    ]},
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },
        {
            'data': [
                Pie(
                    labels=["not related", "related"],
                    values=related_messages,
                    marker={'colors': [
                        'rgb(112, 141, 219)',
                        'rgb(176, 197, 255)'
                    ]},
                )
            ],

            'layout': {
                'title': 'Proportion of Related Messages'
            }
        },
        {
            'data': [
                Histogram(
                    x=messages_len,
                    marker={'color': 'rgb(112, 141, 219)'}
                )
            ],

            'layout': {
                'title': 'Messages Length (Histogram)',
                'yaxis': {
                    'title': 'Messages Count'
                },
                'xaxis': {
                    'title': 'Character Length (limited to 500)',
                    'range': [0, 500],
                    'autorange': False
                }
            }
        },
        {
            'data': [
                Bar(
                    y=category_counts.index,
                    x=category_counts.values,
                    orientation='h',
                    marker={'color': 'rgb(112, 141, 219)'}
                )
            ],

            'layout': {
                'title': 'Category Distribution',
                'yaxis': {
                    'title': 'Category Name',
                    'automargin': 'true'
                },
                'xaxis': {
                    'title': 'Messages Count'
                },
                'height': 1000,
                'margin': {
                    'l': 150
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
