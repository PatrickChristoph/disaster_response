import logging
import pickle
import re
import sys
import warnings
from typing import List, Tuple

import nltk
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath: str) -> Tuple[pd.Series, pd.DataFrame, pd.Index]:
    """
    Load data from ETL pipeline and split into feature and target variables.

    :param database_filepath: path of the ETL database file
    :return: ETL pipeline data split into feature and target variables
    """
    sql_engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("data", sql_engine.connect())

    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names

def convert_pos_tags_to_lemmatizer_params(pos_tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Translate the nltk pos tags to the related pos tag that the WordNetLemmatizer needs as input parameter.

    :param pos_tags: tokens with related nltk pos tags
    :return: tokens with converted pos tags for the WordNetLemmatizer
    """
    converted_pos_tags = []

    for pos_tag in pos_tags:
        if pos_tag[1].startswith('J'):
            lemmatizer_param = 'a'
        elif pos_tag[1].startswith('V'):
            lemmatizer_param = 'v'
        elif pos_tag[1].startswith('N'):
            lemmatizer_param = 'n'
        elif pos_tag[1].startswith('R'):
            lemmatizer_param = 'r'
        else:
            lemmatizer_param = None

        converted_pos_tags.append((pos_tag[0], lemmatizer_param))

    return converted_pos_tags

def tokenize(text: str) -> List[str]:
    """
    Split a disaster message into single normalized tokens.

    :param text: disaster message
    :return: built tokens from disaster message
    """
    # remove special characters and lowercase all
    text = re.sub(r'\W', ' ', text.lower())

    # split text into single tokens
    tokens = nltk.tokenize.word_tokenize(text)

    # remove unnecessary whitespaces
    tokens = [token.strip() for token in tokens]

    # lemmatize tokens
    tokens_with_pos_tag = convert_pos_tags_to_lemmatizer_params(nltk.pos_tag(tokens))
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos=pos_tag) if pos_tag is not None else token for token, pos_tag in tokens_with_pos_tag]

    return tokens


def tokenize_without_numbers(text):
    # remove special characters and lowercase all
    text = re.sub(r'\W', ' ', text.lower())

    # replace numbers with placeholder
    text = re.sub(r'\d+', 'num_placeholder', text)

    # split text into single tokens
    tokens = nltk.tokenize.word_tokenize(text)

    # remove unnecessary whitespaces
    tokens = [token.strip() for token in tokens]

    # get pos tags for appropriate lemmatization
    tokens_with_pos_tag = convert_pos_tags_to_lemmatizer_params(nltk.pos_tag(tokens))

    # lemmatize tokens
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos=pos_tag) if pos_tag is not None else token for token, pos_tag in tokens_with_pos_tag]

    return tokens


def build_model() -> GridSearchCV:
    """
    Build the model pipeline using cross-validated model parameters.

    :return: model pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        # best parameter used in the current model: tokenizer=tokenize_without_numbers, ngram_range=(1, 3), max_df=0.9, min_df=0.01
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'vect__max_df': [0.7, 0.8, 0.9, 0.95],
        'vect__min_df': [0.1, 0.05, 0.02, 0.01, 0.0],
        'vect__tokenizer': [tokenize, tokenize_without_numbers]
    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-3)

def evaluate_model(
        model: GridSearchCV,
        X_test: pd.Series,
        Y_test: pd.DataFrame,
        category_names: pd.Index
) -> None:
    """
    Predict target values (categories) with test data and build classification report for each target variable.

    :param model: trained model
    :param X_test: test disaster messages
    :param Y_test: test category values
    :param category_names: names of all target variables
    """
    Y_pred = model.predict(X_test)

    logging.info('Classification reports for each category:')
    for n, category_name in enumerate(category_names):
        print(f'Category: {category_name}')
        print(classification_report(Y_test.iloc[:, n], [pred_value[n] for pred_value in Y_pred]))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    Save trained model as pickle file.

    :param model: trained model
    :param model_filepath: path where the model will be saved as pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)

def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S', level=logging.INFO, stream=sys.stdout)
    warnings.filterwarnings("ignore")

    database_filepath = '../data/etl.db'
    model_filepath = 'model.pkl'

    logging.info('Loading data...')
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    logging.info('Building model...')
    model = build_model()

    logging.info('Training model...')
    model.fit(X_train, Y_train)
    logging.info(f'Best model parameters:\n {model.best_params_}')

    logging.info('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    logging.info('Saving model...')
    save_model(model, model_filepath)

    logging.info('Trained model saved!')


if __name__ == '__main__':
    main()
