# PeFAD: A Parameter-Efficient Federated Framework for Time Series Anomaly Detection (under review)
# Get Start
### Environment
- **Python Version**: 3.8
- **PyTorch Version**: 1.7.1
```shell
pip install -r requirements.txt
```
### Datasets
#### SMD
- Server Machine Dataset (SMD) is a 5-week-long dataset collected from a large Internet company with a time granularity of 1 minute. You can learn about it from 
[Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf).
#### PSM
- Pooled Server Metrics (PSM) is a public dateset collected internally from multiple application server nodes at eBay. You can learn about it from 
[Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization](https://dl.acm.org/doi/abs/10.1145/3447548.3467174).
#### SWaT
- Secure water treatment (SWaT) offers a condensed depiction of an actual industrial water treatment plant specializing in filtered water production. You can learn about it from [SWaT: a water treatment testbed for research and training on ICS security](https://ieeexplore.ieee.org/abstract/document/7469060).
#### MSL
- NASA's Mars Science Laboratory (MSL) dataset, collected during the spacecraft's journey to Mars, is a valuable resource accessible through NASA-designated data centers. You can learn about it from [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/pdf/1802.04431.pdf).


### Train and evaluate
You can reproduce the experiment results as follows:
```
bash ./scripts/SMD.sh
bash ./scripts/PSM.sh
bash ./scripts/SWAT.sh
bash ./scripts/MSL.sh
```
