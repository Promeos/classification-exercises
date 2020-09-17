import numpy as np
import pandas as pd

model_container = []

def models(df, reset=False):
    
    if reset == False:
        model_container.append([df])
    elif reset == True:
        model_container = []
        model_container.append([df])
    else:
        return None

def outcome_matrix(df, target_column='actual'):
    target = df[target_column]
    model = df.drop(columns=[target_column, 'baseline']).columns.to_list()
    models = pd.concat([df['baseline'], df[model]], axis=1)
    outcome_matrix = pd.crosstab(models, target)
    return outcome_matrix

def evaluation_metrics(matrix, model_number=''):
    '''
    This function accepts a confusion matrix, M
    Returns classification metrics:
    Accuracy, Recall, Precision,Specificity
    '''
     # unravels matrix row wise
    true_positives, false_positives, false_negatives, true_negatives = matrix.values.ravel()
    
    accuracy = (true_positives + true_negatives) / sum([true_positives, false_positives, false_negatives, true_negatives])
    
    recall =true_positives / (true_positives + false_negatives)
    
    precision = true_positives / (true_positives + false_positives)
    
    specificity = true_negatives / (true_negatives + false_positives)
    
    df = pd.DataFrame([accuracy, recall, precision, specificity])
    df = round(df, 2) * 100
    df.columns = [('Model ' + model_number + " Evaluation")]
    df.index = ['Accuracy', 'Recall', 'Precision','Specificity']
    
    
    return df



def summary_metrics(model_metrics):
    return pd.concat(model_metrics)
    