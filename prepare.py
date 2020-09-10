import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prep_iris(df):
    '''
    prep_iris accepts the iris dataset and returns a transformed iris dataset
    for exploratory analysis.
    type(df) >>> pandas.core.frame.DataFrame
    '''
    df.drop(columns=['species_id', 'measurement_id'], inplace=True)
    df.rename(columns={'species_name': 'species'}, inplace=True)
    
    encoded_species = pd.get_dummies(df.species, prefix='species', drop_first=True)
    
    df = pd.concat([df, encoded_species], axis=1)
    
    return df


def prep_titanic(df):
    '''
    prep_titanic accepts the titanic dataset and returns a transformed titanic dataset
    for exploratory analysis.
    type(df) >>> pandas.core.frame.DataFrame
    '''
    
    df.dropna(how='any', subset=['embarked'], inplace=True)
    df.drop(columns='deck', inplace=True)
    
    encoded_embarked = pd.get_dummies(df.embarked, prefix='embarked', drop_first=True)
    
    scaler = MinMaxScaler()
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    
    df = pd.concat([df, encoded_embarked], axis=1)
    
    return df