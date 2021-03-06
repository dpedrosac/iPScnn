#!/usr/bin/env python3

import sys
import os
import pickle
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

# Load data
working_directory = 'D:/iPScnn/data/EMG/results/'
output: Dict[str, any] = pickle.load(
    open(os.path.join(working_directory, "results_tremorclass.bin"), "rb"))

# General settings for the plot
feature_set_colors = {'RMS': '#B34439', 'Hudgins': '#6A7F99', 'Du': '#869926'}
roc_cls_mark_types = ['o', '^', 's', 'D', 'v', '<', '>', '8', 'p', '*', 'h', 'H', 'd', 'P', 'X']
index = np.arange(len(output["classifiers"]))
bar_width = 0.15
bar_spacer = 0.025
dpi = 96
marker_size = 80
fontsize = 20

# Open the figures for filling with data later
fig_precision, ax_precision = plt.subplots(num="Classifiers precisions", figsize=(1800 / dpi, 450 / dpi), dpi=dpi)
fig_recall, ax_recall = plt.subplots(num="Classifiers recalls", figsize=(1800 / dpi, 450 / dpi), dpi=dpi)
fig_accuracy, ax_accuracy = plt.subplots(num="Classifiers accuracies", figsize=(1800 / dpi, 450 / dpi), dpi=dpi)
fig_roc, ax_roc = plt.subplots(num="precision_vs_recall", figsize=(1800 / dpi, 800 / dpi), dpi=dpi)

figs = [fig_precision, fig_recall, fig_accuracy]
axes = [ax_precision, ax_recall, ax_accuracy]

roc_legend_str = []

for i, feature_set in enumerate(output["feature_sets"].keys()):

    precision_mean = []
    precision_std = []
    precision_median = []
    precision_25percentile = []
    precision_75percentile = []

    recall_mean = []
    recall_std = []
    recall_median = []
    recall_25percentile = []
    recall_75percentile = []

    accuracy_mean = []
    accuracy_std = []
    accuracy_median = []
    accuracy_25percentile = []
    accuracy_75percentile = []

    for i_clf, clf in enumerate(output["classifiers"].keys()):
        data = list(filter(lambda r: r["clf"] == clf and r["feature_set"] == feature_set, output["results"]))

        y_true = [r["y_true"] for r in data]
        y_pred = [r["y_pred"] for r in data]

        precision = [precision_score(t, p, average="macro", labels=np.unique(p))
                     for p, t in zip(y_pred, y_true)]

        recall = [recall_score(t, p, average="macro", labels=np.unique(t))
                  for p, t in zip(y_pred, y_true)]

        accuracy = [accuracy_score(t, p, normalize=True) for p, t in zip(y_pred, y_true)]

        cm_sum = sum([r["cm"] for r in data])

        precision_std.append(np.std(precision))
        recall_std.append(np.std(recall))
        accuracy_std.append(np.std(accuracy))

        precision_mean.append(np.mean(precision))
        recall_mean.append(np.mean(recall))
        accuracy_mean.append(np.mean(accuracy))

        precision_median.append(np.median(precision))
        recall_median.append(np.median(recall))
        accuracy_median.append(np.median(accuracy))

        precision_25percentile.append(np.percentile(precision, 25))
        recall_25percentile.append(np.percentile(recall, 25))
        accuracy_25percentile.append(np.percentile(accuracy, 25))

        precision_75percentile.append(np.percentile(precision, 75))
        recall_75percentile.append(np.percentile(recall, 75))
        accuracy_75percentile.append(np.percentile(accuracy, 75))

        ax_roc.scatter(recall_mean[-1], precision_mean[-1],  # median -> mean !!!
                       c=feature_set_colors[feature_set], marker=roc_cls_mark_types[i_clf],
                       s=[marker_size], zorder=3)

        ax_roc.errorbar(recall_mean[-1], precision_mean[-1],
                        fmt='none', ecolor=feature_set_colors[feature_set], lw=2, capsize=5, capthick=2,
                        yerr=[[precision_mean[-1] - precision_25percentile[-1]],
                              [precision_75percentile[-1] - precision_mean[-1]]],
                        xerr=[[recall_mean[-1] - recall_25percentile[-1]],
                              [recall_75percentile[-1] - recall_mean[-1]]],
                        zorder=2)
        roc_legend_str.append(clf + '/' + feature_set)

    ax_precision.bar(index + (bar_width + bar_spacer) * i, precision_mean, bar_width, label=feature_set,
                     color=feature_set_colors[feature_set])
    ax_precision.errorbar(index + (bar_width + bar_spacer) * i, precision_median,
                          fmt='ko', ecolor="#383634", lw=2, capsize=6,
                          yerr=[np.array(precision_median) - np.array(precision_25percentile),
                                np.array(precision_75percentile) - np.array(precision_median)])

    ax_recall.bar(index + (bar_width + bar_spacer) * i, recall_mean, bar_width, label=feature_set,
                  color=feature_set_colors[feature_set])
    ax_recall.errorbar(index + (bar_width + bar_spacer) * i, recall_median,
                       fmt='ko', ecolor="#383634", lw=2, capsize=6,
                       yerr=[np.array(recall_median) - np.array(recall_25percentile),
                             np.array(recall_75percentile) - np.array(recall_median)])

    ax_accuracy.bar(index + (bar_width + bar_spacer) * i, accuracy_mean, bar_width, label=feature_set,
                    color=feature_set_colors[feature_set])
    ax_accuracy.errorbar(index + (bar_width + bar_spacer) * i, accuracy_median,
                         fmt='ko', ecolor="#383634", lw=2, capsize=6,
                         yerr=[np.array(accuracy_median) - np.array(accuracy_25percentile),
                               np.array(accuracy_75percentile) - np.array(accuracy_median)])

ax_precision.set_ylabel('Precision')
ax_recall.set_ylabel('Recall')
ax_accuracy.set_ylabel('Accuracy')

maj_ticks = np.arange(0, 1100, step=100)
maj_ticks = maj_ticks / 1000

for f, a in zip(figs, axes):
    a.yaxis.label.set_size(25)

    a.set_ylim([0, 1.01])
    a.set_yticks(maj_ticks)
    a.set_xticks(index + ((bar_width + bar_spacer) * (len(output["feature_sets"]) - 1)) / 2)
    a.set_xticklabels(output["classifiers"].keys())
    a.tick_params(axis='x', which='major', labelsize=25)
    a.tick_params(axis='y', which='major', labelsize=15)

    a.yaxis.grid(b=True, which='major', linestyle='-')
    a.set_axisbelow(True)

    a.legend(fontsize=fontsize, loc=[0.18, 0.11])

    f.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.09)

ax_roc.legend(roc_legend_str, ncol=3, fontsize=fontsize, loc='lower right')

maj_ticks = np.arange(400, 1000, step=100)
min_ticks = np.arange(375, 950, step=25)
min_ticks = np.setdiff1d(min_ticks, maj_ticks)
maj_ticks = maj_ticks / 1000
min_ticks = min_ticks / 1000

ax_roc.set_xlim([0.35, 0.95])
ax_roc.set_xticks(min_ticks, minor=True)
ax_roc.set_xticks(maj_ticks)
ax_roc.set_yticks(min_ticks, minor=True)
ax_roc.set_yticks(maj_ticks)
ax_roc.set_ylim([0.42, 0.95])

ax_roc.tick_params(axis='both', which='major', labelsize=15)

ax_roc.yaxis.grid(b=True, which='minor', color='lightgray', linestyle='--')
ax_roc.yaxis.grid(b=True, which='major', linestyle='-')
ax_roc.xaxis.grid(b=True, which='minor', color='lightgray', linestyle='--')
ax_roc.xaxis.grid(b=True, which='major', linestyle='-')

ax_roc.set_axisbelow(True)

ax_roc.set_ylabel('Precision')
ax_roc.set_xlabel('Recall')
ax_roc.yaxis.label.set_size(25)
ax_roc.xaxis.label.set_size(25)

fig_roc.subplots_adjust(left=0.05, right=0.99, top=0.985, bottom=0.09)

plt.show()
