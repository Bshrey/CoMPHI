
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve,confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def get_aupr(pre,rec):
    pr_value=0.0
    for ii in range(len(rec[:-1])):
        x_r,x_l=rec[ii],rec[ii+1]
        y_t,y_b=pre[ii],pre[ii+1]
        tempo=abs(x_r-x_l)*(y_t+y_b)*0.5
        pr_value+=tempo
    return pr_value

def scores(y_test, y_pred, model, th=0.5):           
    y_predlabel = [(0. if item < th else 1.) for item in y_pred]

    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    #print('tn,fp,fn,tp', tn,fp,fn,tp )
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    
    fpr,tpr,thresholds = roc_curve(y_test, y_pred)  
    df=pd.DataFrame(fpr)
    df.to_csv(model+'_fpr.csv', index=False)
    df1=pd.DataFrame(tpr)
    df1.to_csv(model+'_tpr.csv', index=False)



    sen, spe, pre, f1, mcc, acc, auc, tn, fp, fn, tp = np.array([recall_score(y_test, y_predlabel), SPE, precision_score(y_test, y_predlabel), 
                                                                 f1_score(y_test, y_predlabel), MCC, accuracy_score(y_test, y_predlabel), 
                                                                 roc_auc_score(y_test, y_pred), tn, fp, fn, tp])
    precision,recall,thresholds1 =precision_recall_curve(y_test, y_pred)

    aupr=get_aupr(precision,recall)
    return [aupr, auc, f1, acc, sen, spe, pre]  