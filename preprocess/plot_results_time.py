#!/usr/bin/env python3

import sys
import os
import pickle
from typing import List, Dict
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import seaborn as sns

# General settings for the plot
feature_set_colors = {'color': ['#B34439', '#6A7F99', '#869926']}
roc_cls_mark_types = ['o', '^', 's', 'D', 'v', '<', '>', '8', 'p', '*', 'h', 'H', 'd', 'P', 'X']
bar_width = 0.15
bar_spacer = 0.025
dpi = 96
marker_size = 80
fontsize = 20

# Load data
working_directory = 'D:/iPScnn/data/EMG/results/'
output = {}
for k in range(1,9,1):
    output[k]: Dict[str, any] = pickle.load(
        open(os.path.join(working_directory, "results_tremorclass" + str(k)  +"secs.bin"), "rb"))

# Re-arrange data into df
df = pds.DataFrame(columns=["accuracy", "duration", "clf", "feature_set"])
for k in range(1,9,1):
    for i, feature_set in enumerate(output[1]["feature_sets"].keys()):
        for i_clf, clf in enumerate(output[1]["classifiers"].keys()):
            data = list(filter(lambda r: r["clf"] == clf and r["feature_set"] == feature_set, output[k]["results"]))
            y_true = [r["y_true"] for r in data]
            y_pred = [r["y_pred"] for r in data]

            accuracy = [accuracy_score(t, p, normalize=True) for p, t in zip(y_pred, y_true)]
            #cm_sum[k] = sum([r["cm"] for r in data])

            dftemp = pds.DataFrame(list(zip(accuracy,
                                          np.repeat(str(k)+" s.", len(accuracy)),
                                          np.repeat(clf, len(accuracy)),
                                          np.repeat(feature_set, len(accuracy)))),
                      columns=["accuracy", "duration", "clf", "feature_set"])
            df = df.append(dftemp)

# Open the figures for filling with data later
df2 = df.loc[(df["clf"]=="SVM") | (df["clf"]=="kNN")]

#fig = plt.subplots(num="Classifiers accuracies over time", figsize=(1800 / dpi, 450 / dpi), dpi=dpi)
sns.set(font_scale=1.5, style='whitegrid', context='paper')
sns.set_palette(sns.color_palette(feature_set_colors["color"]))

g = sns.catplot(x="duration", y="accuracy", hue="feature_set", col="clf", data=df2, legend=False,
                        linewidth=3.5, kind= "point", height=6, aspect=.95, scale=.65, markers=["^", "o", "*"], s=12)
(g.set_axis_labels("Duration", "Accuracy")
    .set_titles("Classifier = {col_name}", fontsize=24)
    .set(ylim=(0.55, 1))
    .despine(left=True))


new_labels = list(output[1]["feature_sets"].keys())
plt.legend(bbox_to_anchor=(1.02, 0.3), title='Feature set', loc='lower right')
plt.show()
print()



