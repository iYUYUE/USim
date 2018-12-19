import sys  
import os
import statistics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------     
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def main():  
  filepath1 = sys.argv[1]
  filepath2 = sys.argv[2]

  if not os.path.isfile(filepath1):
    print("File path {} does not exist. Exiting...".format(filepath1))
    sys.exit()

  if not os.path.isfile(filepath2):
    print("File path {} does not exist. Exiting...".format(filepath2))
    sys.exit()


  reference = []
  predict = []

  with open(filepath1) as fp:
    for line in fp:
      reference.append(1)
      predict.append(float(line))

  with open(filepath2) as fp:
    for line in fp:
      reference.append(0)
      predict.append(float(line))

  threshold = Find_Optimal_Cutoff(reference, predict)[0]

  f1 = f1_score(y_true=reference, y_pred=[int(val >= threshold) for val in predict])

  print('Best threshold', threshold)
  print('F1 score', f1)


if __name__ == '__main__':  
  main()
