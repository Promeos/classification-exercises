import numpy as np
import pandas as pd

def evaluation_metrics(matrix):
    '''
    This function accepts a confusion matrix, M
    Returns classification metrics:
    Accuracy, Misclassification Rate, Recall, Specificity
    '''
     # unravels matrix column wise
    tp, fn, fp, tn = matrix.values.ravel()
    
    print("Model Evaluation\n" + ("-" * 16))
    # Use the accuracy formula to above to calculate model accuracy
    accuracy = (tp + tn )/sum([tp, tn, fp, fn])
    print(f"Accuracy {accuracy:20.2%}")
    
    error = (fn + fp)/sum([tp,fp,tn,fn])
    print(f"Misclassification Rate {error:3.2%}")
    
    recall = tp / (tp + tn)
    print(f"Recall {recall:22.2%}")
    
    specificity = tn / (tn + fn)
    print(f"Specificity {specificity:17.2%}")
