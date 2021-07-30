'''
PROCESS DATA 

To run this script:
> disaster_response_figure_eight>.\project_env\Scripts\activate
> (project_env) python process_data.py disaster_messages.csv disaster_categories.csv disaster_process_data.db

'''
#---------------------------------------------------------------------------------------------
# Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

#---------------------------------------------------------------------------------------------
# Load data function
def load_data(messages_filepath, categories_filepath):
    '''
    Function for loading and merging the data which is contained in two differents datasets

    Input:
      - messages_filepath -> Path of the 'messages' dataset
      - categories_filepath -> Path of the 'categories' dataset
    Ouput:
      - A unified dataset
    '''
    # Importing datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merging both datasets
    df = messages.merge(categories, how ='left', on =['id'])

    return df

#---------------------------------------------------------------------------------------------
# Clean data function
def clean_data(df):
    '''
    Function for cleaning and transforming the data

    Input:
      - df -> The unified dataset that contains messages and categories

    Ouput:
      - Cleaned dataset
    '''
    # New columns by using the 'categories' variable
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Formating these columns (0 and 1 )
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
    
    categories['related'] = categories['related'].replace(to_replace=2, value=1)

    # Concatenating all categories to the df and removing duplicates
    df = df.drop('categories',axis=1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
 
    return df

#---------------------------------------------------------------------------------------------
# Save data function
def save_data(df, database_filename):
    '''
    Function that store the clean dataset in a sql database

    Input:
      - df -> clean dataset
      - database_filename -> Name (or path) of the database to be store

    Ouput:
      - sql database
    '''
    # Creating the engine and saving the database
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_process_data', engine, index=False, if_exists='replace')   

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

# Putting all together
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
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