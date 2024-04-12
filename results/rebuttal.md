## Additional Response to Reviewer oEFN

**Cons 2: Dataset**

(1) The 4 datasets that we adopted are widely used by existing works, sourced from diverse real-world domains, covering internet data (SMD), server operational data (PSM), infrastructure system (SWaT), and monitoring events (MSL).

(2) We have added an experiment on a new UCR Dataset (KDD21 cup) with 250 types of time series from various domains. The main results below regarding AUC indicate that PeFAD outperforms baselines by at least 22.8%.

| AT | FPT | TimsNet | PeFAD |
| --- | --- | --- | --- |
| 61.7 | 63.6 | 64.6 | 87.4 |

**Minor Details**

**M1: reference**

We update the reference for GPT2: 
>Radford A, Wu J, Child R, et al. Language models are unsupervised multitask learners[J]. OpenAI blog, 2019, 1(8): 9.


**M2: autoML**

We adopt FL to address privacy issues where local data is not allowed to be shared, while AutoML typically doesn't tackle this issue.


**M3: Fig 3**

We change the high-contrast colors in Fig 3, as shown below.

<div style="display: flex;">
  <img src="https://github.com/xu737/PeFAD/blob/main/results/fscore_change.png" alt="F1-Score" width="30%">
  <img src="https://github.com/xu737/PeFAD/blob/main/results/ROC_AUC_change.png" alt="AUC-ROC" width="30%">
</div>


## Other Additional Experiments
The result on the UCR dataset is shown as follows.

|         | AUC   | F1-Score |
| ------- | ----- | -------- |
| AT      | 61.75 | 43.70    |
| FPT     | 63.64 | 50.12    |
| TimsNet | 64.69 | 52.42    |
| PeFAD   | 87.45 | 79.07    |


Moreover, we conduct experiments on the hyperparameter sensitivity of PPDS, the results on SMD dataset are as follows.


| PPDS ($\alpha_1,\alpha_2$) | AUC-ROC |
| --- | --- |
| (0.7, 0.3) | 97.22 |
| (0.6, 0.4) | 97.22 |
| (0.5, 0.5) | 97.16 |
| (0.4, 0.6) | 96.89 |
| (0.3, 0.7) | 96.55 |
| w/o_PPDS | 93.97 |
