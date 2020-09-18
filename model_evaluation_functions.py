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

    
def accuracy_metric(true_positives, false_positives, false_negatives, true_negatives):
    accuracy = (true_positives + true_negatives) / sum([true_positives, false_positives, false_negatives, true_negatives])
    return accuracy    


def recall_metric(true_positives, false_negatives):
    recall = true_positives / (true_positives + false_negatives)
    return recall


def precision_metric(true_positives, false_positives):
    precision = true_positives / (true_positives + false_positives)
    return precision


def specificity_metric(true_negatives, false_positives):
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity





def outcome_matrix(df, target_column='actual'):
    target = df[target_column]
    model = df.drop(columns=[target_column, 'baseline']).columns.to_list()
    models = pd.concat([df['baseline'], df[model]], axis=1)
    outcome_matrix = pd.crosstab(models, target)
    return outcome_matrix


def evaluation_metrics(matrix=None, model_number=1):
    '''
    This function accepts a confusion matrix, M
    Returns classification metrics:
    Accuracy, Recall, Precision, Specificity
    '''
     # unravels matrix row wise
    true_positives, false_positives, false_negatives, true_negatives = matrix.values.ravel()[::-1]
    
    accuracy = accuracy_metric(true_positives, false_positives, false_negatives, true_negatives)
    
    recall = recall_metric(true_positives, false_negatives)
    
    precision = precision_metric(true_positives, false_positives)
    
    specificity = specificity_metric(true_negatives, false_positives)
    
    tpr = true_positives / (true_positives + false_negatives)
    fnr = false_negatives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)
    tnr = true_negatives / (false_positives + true_negatives)
    f1 = 2*(recall*precision)/(recall+precision)
    
    # Accuracy, true positive rate, false positive rate, true negative rate, false negative rate, precision, recall, f1-score
    
    df = pd.DataFrame([accuracy, tpr, fpr, tnr,
                      fnr, precision, recall, f1])
    df = round(df, 8) * 100
    
    
    df.columns = [('Model ' + str(model_number))]
    df.index = ['Accuracy',
                'True Positive Rate',
                'False Positive Rate',
                'True Negative Rate',
                'False Negative Rate',
                'Precision',
                'Recall',
                'F1 Score']
    
    return df





def summary_metrics(model_metrics):
    return pd.concat(model_metrics)
    