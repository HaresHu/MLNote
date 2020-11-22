#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
   FileName:  ml_tools.py

Description:  

    Version:  1.0
    Created:  2020/11/23 00:52:05
   Revision:  2020/11/23 00:52:05

     Author:  HY(Hares)    hares.hu.cn@outlook.com
    Company:  D
"""


from sklearn.metrics import roc_curve, auc

# 画roc曲线
def cal_roc(y_test, y_score, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test[:,:].ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr,tpr,roc_auc
