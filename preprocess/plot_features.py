import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import seaborn as sns
import math


class PlotRoutines:
    def __init__(self, MOI, feat1, feat2, lbl, idx_lbl, details, type_plot, type_test):
        self.MOI = MOI
        self.dims = math.ceil(np.sqrt(len(MOI)))
        self.gs = gridspec.GridSpec(self.dims, self.dims, wspace=.5, hspace=.5)
        self.lbl = lbl
        self.idx_lbl = idx_lbl
        if type_plot == "bp":
            self.boxplot(feat1, feat2, type_test)
        else:
            self.correlation(feat1, feat2, details)

    def boxplot(self, feat1, feat2, type_test):
        fig = plt.figure(figsize=(9, 6))
        iter = -1
        for i in range(self.dims):
            for j in range(self.dims):
                iter = iter + 1

                if i == self.dims - 1:
                    lbl = self.lbl
                else:
                    lbl = []

                try:
                    mtr = self.MOI[iter]
                    dataplot = pds.DataFrame([feat1[mtr].values, feat2[mtr].values], index=self.idx_lbl).transpose()

                    sns.set(style="white", context="paper", font_scale=.84)
                    sns.despine()
                    ax = plt.subplot(self.gs[i * self.dims + j])
                    sns.boxplot(data=dataplot,
                                width=0.4,
                                notch=False,
                                palette="Blues",
                                linewidth=1.2,
                                fliersize=3.0,
                                medianprops=dict(color="firebrick"),
                                orient='v').set(ylabel=mtr)
                    ax.set_xticklabels(lbl)

                    sns.set(style="white", context="paper", font_scale=.84)
                    sns.despine()
                    sns.stripplot(data=dataplot,
                                  color='k',  # Make points black
                                  size=3.0,
                                  alpha=0.6)  # and slightly transparent
                    ax.set_xticklabels(lbl)

                    if type_test == "indep":
                        [h, pnpar] = stats.mannwhitneyu(dataplot[self.idx_lbl[0]].values,
                                                        dataplot[self.idx_lbl[1]].values)
                    elif type_test == "dep":
                        [h, pnpar] = stats.wilcoxon(dataplot[self.idx_lbl[0]].values,
                                                    dataplot[self.idx_lbl[1]].values)
                    else:
                        pnpar = 1
                        print("please enter either 'dep' or 'indep' for tyoe_test!")

                    if pnpar < .05:
                        print("U={:.1f}, p = {:.3f}".format(h, pnpar))
                        if pnpar < .001:
                            form_pstring = "p < .001"
                        else:
                            form_pstring = "p = {:.3f}".format(pnpar)

                        style = dict(size=6, color='gray')
                        ax.text(.5, max(dataplot.values.flatten()) * 1.12, form_pstring, ha='center', **style)
                        ax.axhline(y=max(dataplot.values.flatten()) * 1.1, xmin=.35, xmax=.65,
                                   color='k', linewidth=.5)
                    print(mtr, ":", pnpar)

                except:
                    continue
                    # print("Continuing")

    def correlation(self, feat1, feat2, details):
        fig = plt.figure(figsize=(9, 6))
        iter = -1
        for i in range(self.dims):
            for j in range(self.dims):
                iter = iter + 1

                if i == self.dims - 1:
                    lbl = "Change in UPDRS"
                else:
                    lbl = []

                try:
                    mtr = self.MOI[iter]
                    feat_temp = feat2[mtr].values - feat1[mtr].values

                    updrs_temp = 100 * np.divide(details["updrsOFF"] - details["updrsON"],
                                                 details["updrsOFF"])  # details["updrsDiff"]

                    mtr = self.MOI[iter]
                    sns.set(style="white", context="paper", font_scale=.84)
                    sns.despine()
                    ax = plt.subplot(self.gs[i * self.dims + j])

                    lbl_y = "Change in " + mtr
                    dataplot = pds.DataFrame({lbl_y: feat_temp.transpose(), "UPDRSchange": updrs_temp})
                    value = (details['type'] < 1) | (details['type'] > 1)
                    dataplot["color"] = np.where(value == True, "#561ddf", "#3498db")
                    sns.regplot(data=dataplot, x="UPDRSchange", y=lbl_y, scatter_kws={'facecolors': dataplot['color']},
                                ci=95, n_boot=5000,
                                marker="+")
                    ax.set_xlabel(lbl)
                    x_axis = ax.xaxis
                    if i == self.dims - 1:
                        x_axis.label.set_visible(True)
                    else:
                        x_axis.label.set_visible(False)

                    ax.set_ylabel(lbl_y)

                    slope, intercept, r_value, p_value, std_err = stats.linregress(dataplot[lbl_y],
                                                                                   dataplot['UPDRSchange'])

                    print("for metric:", mtr, ",r-squared:", r_value ** 2, ",p =", p_value)

                except:
                    continue
                    # print("Continuing!")
