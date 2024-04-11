## Additional Response to Reviewer oEFN

**Cons 2: Dataset**

(1) The 4 datasets that we adopted are widely used by existing works, sourced from diverse real-world domains, covering internet data (SMD), server operational data (PSM), infrastructure system (SWaT), and monitoring events (MSL).

(2) We have added an experiment on a new UCR Dataset (KDD21 cup) with 250 types of time series from various domains. The main results below regarding F1-Score indicate that PeFAD outperforms baselines by at least 22.8%.

| AT | FPT | TimsNet | PeFAD |
| --- | --- | --- | --- |
| 61.7 | 63.6 | 64.6 | 87.4 |

**Minor Details**
**M2: autoML**

We adopt FL to address privacy issues where local data is not allowed to be shared, while AutoML typically doesn't tackle this issue.

**M1 & M3: GPT2 reference and Fig 3**

We will update the reference for GPT2 and change the high-contrast colors in Fig 3.

## Other Additional Experiments
The result on the UCR dataset is shown as follows.

|         | AUC   | F1-Score |
| ------- | ----- | -------- |
| AT      | 61.75 | 43.70    |
| FPT     | 63.64 | 50.12    |
| TimsNet | 64.69 | 52.42    |
| PeFAD   | 87.45 | 79.07    |


Moreover, we conduct experiments on the hyperparameter sensitivity of ADMS and PPDS, the results on SMD dataset are as follows.

| ADMS ($\beta$) | AUC-ROC |
| --- | --- |
| 0.2 | 97.22 |
| 0.4 | 96.41 |
| 0.6 | 97.19 |
| 0.8 | 97.19 |

| PPDS ($\alpha_1,\alpha_2$) | AUC-ROC |
| --- | --- |
| (0.7, 0.3) | 97.22 |
| (0.6, 0.4) | 97.22 |
| (0.5, 0.5) | 97.16 |
| (0.4, 0.6) | 96.89 |
| (0.3, 0.7) | 96.55 |
| w/o_PPDS | 93.97 |
