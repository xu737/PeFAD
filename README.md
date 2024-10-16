# PeFAD: A Parameter-Efficient Federated Framework for Time Series Anomaly Detection

### Train and Evaluate
You can reproduce the experiment results by running each script in ```./scripts/```:
```
bash ./scripts/SMD.sh
bash ./scripts/PSM.sh
bash ./scripts/SWAT.sh
bash ./scripts/MSL.sh
```

### Citation
Please cite the following paper if this paper/repository is useful for your research.
```
@inproceedings{xu2024pefad,
  title={PeFAD: A Parameter-Efficient Federated Framework for Time Series Anomaly Detection},
  author={Xu, Ronghui and Miao, Hao and Wang, Senzhang and Yu, Philip S and Wang, Jianxin},
  booktitle={SIGKDD},
  pages={3621--3632},
  year={2024}
}
```

### Datasets

_**SMD**_, _**PSM**_, _**SWaT**_, and _**MSL**_ can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm).

#### Biref Dataset Details
- Server Machine Dataset (SMD) is a 5-week-long dataset collected from a large Internet company with a time granularity of 1 minute. Please refer to 
[Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf).
- Pooled Server Metrics (PSM) is a public dateset collected internally from multiple application server nodes at eBay. Please refer to 
[Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization](https://dl.acm.org/doi/abs/10.1145/3447548.3467174).
- Secure Water Treatment (SWaT) is obtained from 51 sensors of the critical infrastructure system under continuous operations. Please refer to [SWaT: a water treatment testbed for research and training on ICS security](https://ieeexplore.ieee.org/abstract/document/7469060).
- NASA's Mars Science Laboratory (MSL) dataset, collected during the spacecraft's journey to Mars, is a valuable resource accessible through NASA-designated data centers. Please refer to [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/pdf/1802.04431.pdf).


### Environment
- Python Version: 3.8
- PyTorch Version: 1.7.1
- Run the following script for environment configuration.
  ```shell
  pip install -r requirements.txt
  ```
