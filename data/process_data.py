import logging

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load and merge disaster messages and disaster categories input data.

    :param messages_filepath: path of messages csv file
    :param categories_filepath: path of categories csv file
    :return: loaded disaster input data
    """
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')

    return messages.merge(categories, how='outer', on='id')

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unusable data and transform the data into an appropriate format for the model training.

    :param df: messages with assigned categories
    :return: cleaned data
    """
    # split data into columns with appropriate column names
    categories = df['categories'].str.split(';', expand=True)
    category_columns = categories.iloc[0].apply(lambda x: x[:-2])
    categories.columns = category_columns

    # transform data into numeric values
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates and unusable messages
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[~df["message"].str.len() < 15]
    df.drop(index=df[df["related"] == 2].index, inplace=True)

    # remove categories with less than 100 positive values
    categories_without_100_positive_values = [column for column in category_columns if df[column].sum() < 100]
    if categories_without_100_positive_values:
        logging.warning(f'The column(s) {categories_without_100_positive_values} can not be used for prediction because they have less than 100 positive values.')
        df.drop(columns=categories_without_100_positive_values, inplace=True)

    return df

def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    Save the cleaned data into a SQLite database. A possibly already existing database file will be replaced.

    :param df: cleaned data
    :param database_filename: path of the database file
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('data', engine, index=False, if_exists='replace')

def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S', level=logging.INFO)

    messages_filepath = 'disaster_categories.csv'
    categories_filepath = 'disaster_messages.csv'
    database_filepath = 'etl.db'

    logging.info('Loading data...')
    df = load_data(messages_filepath, categories_filepath)

    logging.info('Cleaning data...')
    df = clean_data(df)

    logging.info('Saving data...')
    save_data(df, database_filepath)

    logging.info(f'Cleaned data saved to database: {database_filepath}')


if __name__ == '__main__':
    main()
