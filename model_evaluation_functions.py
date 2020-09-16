import numpy as np
import pandas as pd

def evaluation_metrics(matrix):
    '''
    This function accepts a confusion matrix, M
    Returns classification metrics:
    Accuracy, Misclassification Rate, Recall, Specificity
    '''
     # unravels matrix row wise
    true_positives, false_positives, false_negatives, true_negatives = matrix.values.ravel()
    
    print("Model Evaluation\n" + ("-" * 16))
    
    accuracy = (true_positives + true_negatives)/sum([true_positives, false_positives, false_negatives, true_negatives])
    print(f"Accuracy {accuracy:20.2%}")
    
    recall = true_positives / (true_positives + false_negatives)
    print(f"Recall {recall:22.2%}")
    
    precision = true_positives / (true_positives + false_positives)
    print(f"Precision {precision:19.2%}")
    
    specificity = true_negatives / (true_negatives + false_positives)
    print(f"Specificity {specificity:17.2%}")
