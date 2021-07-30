'''
TRAIN CLASSIFIER

To run this script:

> disaster_response_figure_eight>.\project_env\Scripts\activate
> (project_env) cd models > python train_classifier.py ../data/disaster_process_data.db classifier.pkl
'''
#---------------------------------------------------------------------------------------------
# Import libraries
import sys
import pandas as pd
import numpy as np
import pickle 
import re
import nltk
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'wordnet','stopwords', 'averaged_perceptron_tagger'])

#---------------------------------------------------------------------------------------------

# StartingVerbExtractor class (later as a feature for the model)
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Class created.
      Contains a transform function to incorporate in the model as a new feature. 
      It extract the starting verb from a sentence.
    
    '''
    # Function that returs True if the sentence start with a Verb
    def starting_verb(self, text):
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            pos_tags = nltk.pos_tag(tokenize(sent))
            # If the list is empty (pos_tags = []) it returns automaticaly False
            if len(pos_tags):
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
                return False
            else:
                return False

    # Fit model (specially for CountVectorizer())
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        # We noticed that the X_tagged had 2 null values. We solved this by replacing nulls with fillna()
        df = pd.DataFrame(X_tagged).fillna({'message': False})
        return df

#---------------------------------------------------------------------------------------------
# Load data function
def load_data(database_filepath):
    '''
    Function for loading the data

    Input:
      - database_filepath -> The path where the data base is located
    Output
      - X -> Features from the db
      - Y -> Target
      - category_names -> label of each category
    '''
    # Conection and dataframe
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_process_data", con=engine)

    # Features and labels
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names

#---------------------------------------------------------------------------------------------
# Tokenize function
def tokenize(text):
    '''
    Function for preparing the data (tokenization | lemmatization | stopwords)

    Input:
      - text -> raw text
    Output:
      - clean_text -> List of words that are clean and useful (tokens)
    
    '''
    # Detecting urls and replacing them (Although it's unlikely to find urls in disaster messages, it's good criterion to clean it anyways) 
    url_format = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_found = re.findall(url_format, text)
    
    for url in urls_found:
        text = text.replace(url, "urlplaceholder")
    
    # Spliting the sentences into tokens 
    tokens = word_tokenize(text)
    
    # Adding the lemmatizer functions too
    lemmatizer = WordNetLemmatizer()
    
    # Applying the lemmatization function 
    clean_text = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token).lower().strip()
        clean_text.append(lemma)
     
    # Removing stop words
    clean_text = [word for word in clean_text if word not in stopwords.words("english")]
    
    return clean_text

#---------------------------------------------------------------------------------------------
# build_model function
def build_model():
    '''
    Function that applies all techniques from ML pipelines generating the final model. 
    It creates a classifier and then look for the best one by iterating through different parameters
    
    '''
    # Generating a pipeline that contains all functions and features, including the StartingVerbExtractor feature.
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Testing new parameters
    parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    #'features__text_pipeline__vect__max_df': (0.5, 1.0),
    #'clf__estimator__n_estimators' : [50,100]
    }
    
    # Final model having the best parameters tested
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

#---------------------------------------------------------------------------------------------
# Evaluate_model function
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that measures the performance of the model (overall and by each variable)

    Input:
      - model -> The model to be evaluated
      - X_test -> Test dataset for features
      - Y_test -> Test dataser containing the label
      - category_names -> Names of variables or categories
    
    Output
      - Measures of each column and the overall accuracy

    '''
    
    import warnings
    warnings.filterwarnings("ignore")

    y_pred = model.predict(X_test)
    
    # Iterating each column
    for i, col in enumerate(Y_test):
        # Printing the results for each variable
        print(f'\033[1m{col}\033[0m')
        print(classification_report(Y_test[col], y_pred[:,i]))
    
    # Overall accuracy (mean)
    accuracy = (y_pred == Y_test).mean().mean()
    print(f'\033[1mGeneral Accuracy: {accuracy}\033[0m')  

#---------------------------------------------------------------------------------------------
# Save_model function
def save_model(model, model_filepath):
    '''
    Function for saving the model as a pickle file 

    Input:
      - model -> classifier
      - model_filepath -> The path where we want to save the model
    Output:
      - Pickle file which contains the classifier
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

# Putting all together
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()