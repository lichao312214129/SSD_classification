# -*- coding: utf-8 -*-
"""This script is used to visualization for figure 3.
"""

#%%
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
sys.path.append(r'D:\My_Codes\easylearn-fmri\eslearn\statistical_analysis')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import pickle
import seaborn as sns

from lc_binomialtest import lc_binomialtest
from eslearn.statistical_analysis.lc_anova import oneway_anova
from eslearn.statistical_analysis.lc_chisqure import lc_chisqure
from eslearn.statistical_analysis.lc_ttest2 import ttest2
from eslearn.visualization.el_violine import ViolinPlotMatplotlib
from eslearn.utils.lc_evaluation_model_performances import eval_performance

#%% Inputs
scale_550_file = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
classification_results_figure3_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_leave_one_site_cv.npy'
is_plot = 1
is_savefig = 1

#%% Load and proprocess
scale_550 = pd.read_excel(scale_550_file)
scale_550['folder'] = np.int32(scale_550['folder'])

results_figure3_leave_one_site_cv = np.load(classification_results_figure3_leave_one_site_cv_file, allow_pickle=True)
results_special_figure3 = results_figure3_leave_one_site_cv['special_result']
results_special_figure3 = pd.DataFrame(results_special_figure3)
results_special_figure3.iloc[:, 0] = np.int32(results_special_figure3.iloc[:, 0])


# Filter subjects that have .mat files
scale_550_selected_figure3 = pd.merge(results_special_figure3, scale_550, left_on=0, right_on='folder', how='inner')
#%% Calculate performance for Schizophrenia Spectrum subgroups
duration = 18  # Upper limit of first episode: 
# Frist episode unmedicated; first episode medicated; chronic medicated
# figure3_svc
data_chronic_medicated_SSD_550_18_figure3 = scale_550_selected_figure3[
    (scale_550_selected_figure3['诊断']==3) & 
    (scale_550_selected_figure3['病程月'] > duration) & 
    (scale_550_selected_figure3['用药'] == 1)
]
data_firstepisode_medicated_SSD_550_18_figure3 = scale_550_selected_figure3[
    (scale_550_selected_figure3['诊断']==3) & 
    (scale_550_selected_figure3['首发'] == 1) &
    (scale_550_selected_figure3['病程月'] <= duration) & 
    (scale_550_selected_figure3['用药'] == 1)
]
data_firstepisode_unmedicated_SSD_550_18_figure3 = scale_550_selected_figure3[
    (scale_550_selected_figure3['诊断']==3) & 
    (scale_550_selected_figure3['首发'] == 1) &
    (scale_550_selected_figure3['病程月'] <= duration) & 
    (scale_550_selected_figure3['用药'] == 0)
]

#%% Calculating Accuracy
# figure3
acc_chronic_medicated_SSD_550_18_figure3 = np.sum(data_chronic_medicated_SSD_550_18_figure3[1]-data_chronic_medicated_SSD_550_18_figure3[3]==0) / len(data_chronic_medicated_SSD_550_18_figure3)
acc_firstepisode_medicated_SSD_550_18_figure3 = np.sum(data_firstepisode_medicated_SSD_550_18_figure3[1]-data_firstepisode_medicated_SSD_550_18_figure3[3]==0) / len(data_firstepisode_medicated_SSD_550_18_figure3)
acc_first_episode_unmedicated_SSD_550_18_figure3 = np.sum(data_firstepisode_unmedicated_SSD_550_18_figure3[1]-data_firstepisode_unmedicated_SSD_550_18_figure3[3]==0) / len(data_firstepisode_unmedicated_SSD_550_18_figure3)
accuracy_figure3, sensitivity_figure3, specificity_figure3, auc_figure3 = eval_performance(scale_550_selected_figure3[1].values, scale_550_selected_figure3[3].values, scale_550_selected_figure3[2].values,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=False, legend1='HC', legend2='Patients', is_savefig=False, out_name=None)

#%% Statistics
# figure3
n = len(data_chronic_medicated_SSD_550_18_figure3)
acc = acc_chronic_medicated_SSD_550_18_figure3
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_medicated_SSD_550_18_figure3)
acc = acc_firstepisode_medicated_SSD_550_18_figure3
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_unmedicated_SSD_550_18_figure3)
acc = acc_first_episode_unmedicated_SSD_550_18_figure3
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)

#%% Plot
plt.figure(figsize=(5,8))

plt.bar([0,1,2,3,4,5],
    [accuracy_figure3, sensitivity_figure3, specificity_figure3,
    acc_chronic_medicated_SSD_550_18_figure3, 
    acc_firstepisode_medicated_SSD_550_18_figure3, 
    acc_first_episode_unmedicated_SSD_550_18_figure3], 
    alpha=0.7,
    facecolor='w',
    edgecolor=['teal', 'teal', 'teal', 'teal', 'teal', 'deeppink'],
    linewidth=3,
)
plt.yticks(fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5], ['Accuracy', 'Sensitivity','Specificity', 'Sensitivity of chronic medicated SSD', 'Sensitivity of first episode medicated SSD', 'Sensitivity of first episode unmedicated SSD'], rotation=45, ha="right")  
plt.grid(axis='y')
plt.title('Classification performances', fontsize=12, fontweight='bold')

plt.subplots_adjust(wspace = 0.5, hspace =1)
plt.tight_layout()
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\performances_fig3.pdf')
pdf.savefig()
pdf.close()
plt.show()
print('-'*50)


