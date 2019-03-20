import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score


def auroc(truth, predicted_scores):
    return roc_auc_score(truth, predicted_scores)


def bin_probs(p, threshold=0.5):
    y = np.zeros(p.shape[0])
    y[p > threshold] = 1
    return y


def metric_report(truth, predictions):

    print('Metric report: ')
    print('-----------------------------------------------------------------')

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    sens = 1. * tp / (tp + fn)
    spec = 1. * tn / (tn + fp)
    recall = 1. * tp / (tp + fn)
    precision = 1. * tp / (tp + fp)

    f1 = 2 * (precision * recall) / (precision + recall)

    print('sensitivity: {0} \t specificity: {1}'.format(sens, spec))
    print('precision: {0} \t recall: {1}'.format(precision, recall))

    print('F1 score: {0}'.format(f1))
    print('confusion matrix: \n', confusion_matrix(truth, predictions))
    print('-----------------------------------------------------------------\n')

    return tn, fp, fn, tp, f1


def f1(truth, predictions):
    return f1_score(truth, predictions)

def avgPR(truth, predicted_scores):
    return average_precision_score(truth, predicted_scores)
