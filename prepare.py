import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prep_iris(df):
    '''
    prep_iris accepts the iris dataset and returns a transformed iris dataset
    for exploratory analysis.
    type(df) >>> pandas.core.frame.DataFrame
    '''
    # Drop columns of redundant data or 'index-like'/ordinal row.
    df.drop(columns=['species_id', 'measurement_id'], inplace=True)
    
    # Rename species_name to be concise.
    df.rename(columns={'species_name': 'species'}, inplace=True)
    
    # Create dummy variables for our targets - 0 0 represents 'species_setosa'
    encoded_species = pd.get_dummies(df.species, prefix='species', drop_first=True)
    
    # Add the encoded target names as columns to the dataframe.
    df = pd.concat([df, encoded_species], axis=1)
    
    return df


def prep_titanic(df):
    '''
    prep_titanic accepts the titanic dataset and returns a transformed titanic dataset
    for exploratory analysis.
    type(df) >>> pandas.core.frame.DataFrame
    '''
    # Drop missing values in the embarked column.
    # This removes missing values in embark_town as well.
    df.dropna(how='any', subset=['embarked'], inplace=True)
    
    # Throw the deck overboard because there are too many missing values.
    df.drop(columns='deck', inplace=True)
    
    # Create dummy variables for our targets. 0 0 represents 'embarked_c'
    encoded_embarked = pd.get_dummies(df.embarked, prefix='embarked', drop_first=True)
    
    # Scale numerical columns using MinMaxScalar()
    scaler = MinMaxScaler()
    
    # Use `.transform_fit` on the scalar object to fit and transform the data.
    # Assign directly to 'age' and 'fare' columns.
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    
    # Add the encoded target names as columns to the dataframe.
    df = pd.concat([df, encoded_embarked], axis=1)
    
    return df