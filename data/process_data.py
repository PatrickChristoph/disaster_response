import sys

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
    Transform data in correct format and remove unusable data.

    :param df: messages with assigned categories
    :return: cleaned data
    """
    # transform data into appropriate columns
    categories = df['categories'].str.split(';', expand=True)
    category_columns = categories.iloc[0].apply(lambda x: x[:-2])
    categories.columns = category_columns

    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates and unusable messages
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[~df["message"].str.len() < 15]

    return df

def save_data(df: pd.DataFrame, database_filename: str) -> None:
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('data', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath = 'disaster_categories.csv'
        categories_filepath = 'disaster_messages.csv'
        database_filepath = 'etl.db'

        print('Loading data...')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()