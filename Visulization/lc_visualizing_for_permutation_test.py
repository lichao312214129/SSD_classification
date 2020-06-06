import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages

permutation_pooledcv = np.load(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\performances_excluded_greater_fd_and_regressed_out_age_sex_motion_separately.npy', allow_pickle=True)
permutation_feucv = np.load( r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\performances_feu_excluded_greater_fd_and_regressed_out_age_sex_motion_separately.npy', allow_pickle=True)

#%% Pooled
rand_pooledcv = [np.mean(perf, axis=1) for perf in permutation_pooledcv[:-1]]
rand_pooledcv_acc = [perf[0] for perf in rand_pooledcv]
rand_pooledcv_sens = [perf[1] for perf in rand_pooledcv]
rand_pooledcv_spec = [perf[2] for perf in rand_pooledcv]

plt.figure(figsize=(12,7))
plt.subplot(3,3,1)
plt.hist(rand_pooledcv_acc, color='darkturquoise', alpha=0.9)
plt.plot(permutation_pooledcv[-1][0].mean(),7, '*', markersize=15, color='orange')
pvalue = (np.sum(rand_pooledcv_acc > permutation_pooledcv[-1][0].mean())+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Accuracy\np={pvalue:.3f}')

plt.subplot(3,3,4)
plt.hist(rand_pooledcv_sens, color='darkturquoise', alpha=0.9)
plt.plot(permutation_pooledcv[-1][1].mean() ,7, '*', markersize=15, color='orange')
pvalue = (np.sum(rand_pooledcv_sens > permutation_pooledcv[-1][1].mean())+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Sensitivity\np={pvalue:.3f}')

plt.subplot(3,3,7)
plt.hist(rand_pooledcv_spec, color='darkturquoise', alpha=0.9)
plt.plot(permutation_pooledcv[-1][2].mean() ,7, '*', markersize=15, color='orange')
pvalue = (np.sum(rand_pooledcv_spec > permutation_pooledcv[-1][2].mean())+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Specificity\np={pvalue:.3f}')


#%% FEU
rand_feucv = [np.mean(perf, axis=1) for perf in permutation_feucv[:-1]]
rand_feucv_acc = [perf[0] for perf in rand_feucv]
rand_feucv_sens = [perf[1] for perf in rand_feucv]
rand_feucv_spec = [perf[2] for perf in rand_feucv]

plt.subplot(3,3,2)
plt.hist(rand_feucv_acc, color='paleturquoise', alpha=0.9)
plt.plot(permutation_feucv[-1][0].mean(),7, '*', markersize=15, color='orange')
pvalue = (np.sum(rand_feucv_acc > permutation_feucv[-1][0].mean())+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Accuracy\np={pvalue:.3f}')

plt.subplot(3,3,5)
plt.hist(rand_feucv_sens, color='paleturquoise', alpha=0.9)
plt.plot(permutation_feucv[-1][1].mean() ,7, '*', markersize=15, color='orange')
pvalue = (np.sum(rand_feucv_sens > permutation_feucv[-1][1].mean())+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Sensitivity\np={pvalue:.3f}')

plt.subplot(3,3,8)
plt.hist(rand_feucv_spec, color='paleturquoise', alpha=0.9)
plt.plot(permutation_feucv[-1][2].mean() ,7, '*', markersize=15, color='orange')
pvalue = (np.sum(rand_feucv_spec > permutation_feucv[-1][2].mean())+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Specificity\np={pvalue:.3f}')

#%% Real performances
accuracy_pooling, sensitivity_pooling, specificity_pooling = permutation_pooledcv[-1][0],  permutation_pooledcv[-1][1], permutation_pooledcv[-1][2]
performances_pooling = [accuracy_pooling, sensitivity_pooling, specificity_pooling]
performances_pooling = pd.DataFrame(performances_pooling)

accuracy_feu, sensitivity_feu, specificity_feu = permutation_feucv[-1][0],  permutation_feucv[-1][1], permutation_feucv[-1][2]
performances_feu = [accuracy_feu, sensitivity_feu, specificity_feu]
performances_feu = pd.DataFrame(performances_feu)

# Bar: performances in the whole Dataset.
plt.subplot(1,3,3)
all_mean = pd.concat([np.mean(performances_pooling, axis=1), np.mean(performances_feu, axis=1)])
error = pd.concat([np.std(performances_pooling, axis=1), np.std(performances_feu, axis=1)])

color = ['darkturquoise'] * 3 +  ['paleturquoise'] * 3
plt.bar(np.arange(0,len(all_mean)), all_mean, yerr = error, 
        capsize=5, linewidth=2, color=color)
# plt.tick_params(labelsize=10)
plt.xticks(np.arange(0,len(all_mean)), ['Accuracy', 'Sensitivity', 'Specificity'] * 3, fontsize=10, rotation=45, ha='right')
plt.title('Classification performances')
y_major_locator=MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.grid(axis='y')

    
plt.subplots_adjust(wspace=0.4, hspace=0.5)
plt.tight_layout()
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\permutation_test.pdf')
pdf.savefig()
pdf.close()
plt.show()
print('-'*50)
