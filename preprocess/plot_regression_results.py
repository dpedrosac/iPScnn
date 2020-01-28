#!/usr/bin/env python3

import sys
import os, math, pickle, re
from typing import List, Dict
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.metrics import mean_squared_error, r2_score

# General settings for the plot
feature_set_colors = {'#B34439', '#6A7F99', '#869926'}
roc_cls_mark_types = ['o', '^', 's', 'D', 'v', '<', '>', '8', 'p', '*', 'h', 'H', 'd', 'P', 'X']
bar_width = 0.15
bar_spacer = 0.025
dpi = 96
marker_size = 80
fontsize = 20

# Load data
working_directory = 'D:/iPScnn/data/EMG/results/'
output = pickle.load(
    open(os.path.join(working_directory, "results_shallow_regression.bin"), "rb"))

df2 = pds.DataFrame(columns=["MSE", "R2", "clf", "feature"])
dfcorr = {}
data_temp1 = []
data_temp2 = []
data_temp3 = []
data_temp4 = []
data_temp5 = []
data_temp6 = []
data_temp7 = []
data_temp8 = []

for i, feature_set in enumerate(output["feature_sets"].keys()):
    for i_clf, clf in enumerate(output["classifiers"].keys()):
        if (clf == 'LRS') or (clf == 'xLassoRegression'):
            continue
        else:
            data = list(filter(lambda r: r["clf"] == clf and r["feature_set"] == feature_set, output["results"]))

            y_true_temp = [r["y_true"] for r in data]
            y_pred_temp = [r["y_pred"] for r in data]

            ytrain_true_temp = [r["ytrain_true"] for r in data]
            ytrain_pred_temp = [r["ytrain_pred"] for r in data]

            mse = [math.sqrt(mean_squared_error(y_pred_temp[i], y_true_temp[i])) for i in range(len(y_pred_temp))]
            r2 = [r2_score(y_pred_temp[i], y_true_temp[i]) for i in range(len(y_pred_temp))]
            new_entry = {"MSE": mse, "R2": r2, "clf": [clf] * len(mse), "feature": [feature_set] * len(mse)}
            df2 = df2.append(pds.DataFrame(new_entry))

            data_temp1 = data_temp1 + list(np.mean(np.array(y_true_temp), axis=0))
            data_temp2 = data_temp2 + list(np.mean(np.array(y_pred_temp), axis=0))
            data_temp3 = data_temp3 + [clf] * len(np.mean(np.array(y_true_temp), axis=0))
            data_temp4 = data_temp4 + [feature_set] * len(np.mean(np.array(y_true_temp), axis=0))

            data_temp5 = data_temp5 + list(np.mean(np.array(ytrain_true_temp), axis=0))
            data_temp6 = data_temp6 + list(np.mean(np.array(ytrain_pred_temp), axis=0))
            data_temp7 = data_temp7 + [clf] * len(np.mean(np.array(ytrain_pred_temp), axis=0))
            data_temp8 = data_temp8 + [feature_set] * len(np.mean(np.array(ytrain_pred_temp), axis=0))

        dfcorr[clf] = [r["trainX"] for r in data]

df = pds.DataFrame(data={'y_true': data_temp1,
                         'y_pred': data_temp2,
                         'clf': data_temp3,
                         'feature': data_temp4})

dftrain = pds.DataFrame(data={'ytrain_true': data_temp5,
                              'ytrain_pred': data_temp6,
                              'clf': data_temp7,
                              'feature': data_temp8})

# -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- #
# Plot the correlation matrix
df_corrDU = dfcorr['LR']
CorrDUall = [df_corrDU[i].corr() for i in range(len(df_corrDU))]
ftr = [re.sub('input_{:d}_'.format(i), "", c) for i, c in enumerate(df_corrDU[0].columns)]

dat_corr = np.empty((CorrDUall[1].shape[1], CorrDUall[1].shape[1], 10))
for n in range(len(CorrDUall)):
    dat_corr[:,:,n] = CorrDUall[n].values

plt.figure(1)
corr = pds.DataFrame(data=np.mean(dat_corr, axis=2), columns=ftr)
sns.set(style="whitegrid", context="paper", font_scale=1.5, font="Cambria")
# TO display diagonal matrix instead of full matrix.
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap.
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio.
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, annot=True, fmt='.2f', square=True,
                linewidths=1.2, cbar_kws={"shrink": .5}, xticklabels=ftr, yticklabels=ftr)

plt.title("Correlation HeatMap for all features used", fontsize=18)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!

# -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- #
plt.figure(2, dpi=300, figsize=(8,3))
sns.set(style='whitegrid', rc={'grid.linestyle': 'dashdot', 'grid.linewidth' : .01, 'lines.linewidth' : .2}, font="Cambria")
sns.set_context(context="paper", font_scale=.8)
sns.set_palette(sns.color_palette(feature_set_colors))
sns.despine(offset=10, trim=True)
dattemp = df2[df2.feature != "all"]

g = sns.boxplot(x="clf", y="MSE",
                hue="feature",
                data=dattemp, orient="v",
                linewidth=.8, fliersize=2)

for p, artist in enumerate(g.artists):
    artist.set_edgecolor('black')
    for q in range(p*6, p*6+6):
        line = g.lines[q]
        line.set_color('black')

    g.lines[p*6+2].set_xdata(g.lines[p*6+1].get_xdata())
    g.lines[p*6+3].set_xdata(g.lines[p*6+1].get_xdata())

handles, _ = g.get_legend_handles_labels()
g.legend(handles, ["RMS", "Hudgins", "Du"], bbox_to_anchor=(1.03, 0.99))
g.set_xlabel('')

g.set_ylabel("Root Mean Squared Error")
labels: Dict = {}
labels = ["Linear \nregression",
          "Lasso \nregression",
          "SVR \n(Gaussian RBF)",
          "SVR \n(Polynomial)",
          "k-nearest neighbor \nregression (kNN)"]
plt.setp(g, xticklabels=labels)
plt.show()

# -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- #
plt.figure(3, dpi=300, figsize=(8,3))
sns.set(style='whitegrid', rc={'grid.linestyle': 'dashdot', 'grid.linewidth' : .01, 'lines.linewidth' : .2}, font="Cambria")
sns.set_context(context="paper", font_scale=1.5)
sns.set_palette(sns.color_palette(feature_set_colors))
sns.despine()

# Remove data from less efficient regression techniques
dftemp = pds.DataFrame()
dattemp = df[df.feature != "all"]
dftemp = dftemp.append(dattemp[dattemp.clf == "SVR_poly"])
dftemp = dftemp.append(dattemp[dattemp.clf == "SVR_rbf"])
dftemp = dftemp.append(dattemp[dattemp.clf == "kNNRegression"])
dfreg = dftemp

g = sns.lmplot(data=dfreg, x="y_true", y="y_pred", hue="clf", col="feature",
               markers=["o", "x", "*"], scatter_kws={'alpha':0.4})
g.set(ylim=(0, 66), xlim=(0, 66))
for n in range(0, 3):
    lims = [0, 66]
    g.axes[0][n].plot(lims, lims, ':k')
g.set_axis_labels("True changes in\nUPDRS after Levodopa intake", "Predicted changes in\nUPDRS after Levodopa intake")

# title
g._legend.set_title("")
g._legend._loc = 4
g._legend.set_bbox_to_anchor([1.09, 0.1])

# replace labels
new_labels = ['SVR \n(kernel: Gaussian RBF)', 'SVR \n(kernel: Gaussian Polynomial)', 'k-nearest neighbour \nregression (kNN)']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
plt.setp(g._legend.get_texts(), fontsize='18')
plt.show()

# -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- #

df_residtest = df[df.clf == "kNNRegression"]
df_residtest = df_residtest[df_residtest.feature == "Du"]

df_residtrain = dftrain[dftrain.clf == "kNNRegression"]
df_residtrain = df_residtrain[df_residtrain.feature == "Du"]

plt.figure(5)
sns.despine(offset=10, trim=True)
sns.set(style="whitegrid", context="paper", font_scale=1.5)
g = sns.residplot(x="y_true", y="y_pred", data=df_residtest,
                  scatter_kws={'alpha': 0.7}, color=list(feature_set_colors)[2], label='Test data')
g = sns.residplot(x="ytrain_true", y="ytrain_pred", data=df_residtrain,
                  scatter_kws={'alpha': 0.25}, color="black", label='Training data')
g.set_xlabel('k-Nearest Neighbour Regression \nstatistics'); g.set_ylabel('UPDRS change Residual $(y-\hat{y})$');
plt.legend(fontsize=24)
plt.show()
# -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- #

plt.figure(4)
sns.set(style="ticks")
sns.set_palette(sns.color_palette(feature_set_colors))
# Load the example tips dataset
# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="clf", y="R2",
            hue="feature",
            data=df2, orient="v")
sns.despine(offset=10, trim=True)
plt.show()

print()