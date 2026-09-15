<div align="center">

# 🌟 awesome-NILM-with-code

**A curated list of Non-Intrusive Load Monitoring (NILM) papers — every entry ships with code.**

> 😸 Welcome valuable opinions from researchers in the same direction

[![Stars](https://img.shields.io/github/stars/zhgqcn/awesome-NILM-with-code.svg?style=flat-square&logo=github)](https://github.com/zhgqcn/awesome-NILM-with-code/stargazers)
[![Forks](https://img.shields.io/github/forks/zhgqcn/awesome-NILM-with-code.svg?style=flat-square&logo=github)](https://github.com/zhgqcn/awesome-NILM-with-code/network/members)
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg?style=flat-square)](./LICENSE)
[![Papers](https://img.shields.io/badge/papers-30%2B-green.svg?style=flat-square)](#-papers)
[![Updated](https://img.shields.io/badge/updated-2026--09-brightgreen.svg?style=flat-square)](#-recent-updates)

</div>

---

## 📑 Contents

| | |
| :--- | :--- |
| 🧰 [Toolkits](#-toolkits) | 🏫 [Conferences](#-conferences) |
| 📊 [Datasets](#-datasets) | 📄 [Papers](#-papers) |
| 🆕 [2025 – 2026](#-2025--2026) | 📆 [2023 – 2024](#-2023--2024) |
| 📂 [2020 – 2022](#-2020--2022) | 🏛️ [Classics · 2015 – 2019](#-classics--2015--2019) |
| 📚 [Reviews](#-reviews) | 🚀 [Deployment](#-deployment) |
| 🔥 [Recent Updates](#-recent-updates) | ⭐ [Star History](#-star-history) |

> 💡 **Reading guide** — every paper is a *card*: a title, one summary paragraph, and badge links to the **paper** and the **code**. Older work is folded into collapsible year blocks so the front page stays scannable — click a year to expand it.

---

## 🔥 Recent Updates

| Date | Change |
| :--- | :--- |
| **2026-09** | 📦 Redesigned README into **modular, year-grouped cards** with badge links; older entries folded into collapsible year blocks. <br> 🆕 Added **MSDCANet** (DSP 2026), **DualNILM** (PV-injection aware), **NILMFormer** (KDD 2025), **training-free LLM NILM**, **diffusion-based augmentation** (Energy 2025), **MATNilm**, **Tsetlin Machine NILM on MCUs** (2026). <br> 🐛 Fixed broken links: MSDC (Google redirect ⇒ AAAI official), GRAD-NILM PDF, attention-NILM code, HawkDATA, nilmworkshop. <br> 🧹 De-duplicated the federated-learning survey entry and added the CC BY 4.0 license. |

---

# 🧰 Toolkits

| Toolkit | Summary | Stack | Links |
| :--- | :--- | :--- | :--- |
| **NILMTK** | The de-facto standard NILM toolkit: dataset parsers, benchmarking API and statistics | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/1404.3878v1.pdf) [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/nilmtk/nilmtk) |
| **NILMTK-Contrib** | Community extensions and reference implementations for NILMTK | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | [![Paper](https://img.shields.io/badge/Paper-ACM-01689E?style=flat-square)](https://dl.acm.org/doi/10.1145/3360322.3360844) [![Model](https://img.shields.io/badge/Model-BuildSys-2d8cf0?style=flat-square)](https://nipunbatra.github.io/papers/2021/buildsys.pdf) [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/nilmtk/nilmtk-contrib) |
| **NILM-Eval** | Reproducible evaluation framework and baseline algorithms | ![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=flat-square&logo=mathworks&logoColor=white) | [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/beckel/nilm-eval) |
| **Torch-NILM** | Benchmarking suite for deep learning models in energy disaggregation | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | [![Paper](https://img.shields.io/badge/Paper-MDPI-00A0DD?style=flat-square)](https://www.mdpi.com/1996-1073/15/7/2647) [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/Virtsionis/torch-nilm) |
| **Deep-NILMtk** | Modular deep learning NILM toolbox (PyTorch + TensorFlow models) | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | [![Paper](https://img.shields.io/badge/Paper-NILM%20Workshop-6f42c1?style=flat-square)](http://nilmworkshop.org/2022/proceedings/nilm22-final4.pdf) [![PyTorch](https://img.shields.io/badge/PyTorch-models-EE4C2C?style=flat-square)](https://github.com/BHafsa/deep-nilmtk-v1/tree/master/deep_nilmtk/models/pytorch) [![TensorFlow](https://img.shields.io/badge/TensorFlow-models-FF6F00?style=flat-square)](https://github.com/BHafsa/deep-nilmtk-v1/tree/master/deep_nilmtk/models/tensorflow) |
| **nilmtk-ukdale** | Exploratory data analysis pipeline for UK-DALE | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/kehkok/nilmtk-ukdale) |
| **NeuralNILM_Pytorch** | PyTorch re-implementation of *Neural NILM* | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/Ming-er/NeuralNILM_Pytorch) |
| **nilm_analyzer** | Disaggregation result inspection and error-analysis utilities | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/mahnoor-shahid/nilm_analyzer) |

---

# 🏫 Conferences

| Venue | Link |
| :--- | :--- |
| 🗞️ **Energy Informatics** | [![Site](https://img.shields.io/badge/Springer-Energy%20Informatics-2d8cf0?style=flat-square)](https://energyinformatics.springeropen.com/) |
| 🗞️ **NILM Workshop** (annual) | [![Site](https://img.shields.io/badge/nilmworkshop.org-homepage-6f42c1?style=flat-square)](https://nilmworkshop.org/) [![Repo](https://img.shields.io/badge/GitHub-site%20source-181717?style=flat-square&logo=github)](https://github.com/smakonin/nilmworkshop.org) |
| 🗞️ **ICPERE 2022** | [![Site](https://img.shields.io/badge/icpere2022-official-2d8cf0?style=flat-square)](http://icpere2022.com/) |

---

# 📊 Datasets

<details>
<summary><b>🏠 Residential datasets (17)</b></summary>
<br>

[UK-DALE](https://www.nature.com/articles/sdata20157) · [REDD](https://energy.duke.edu/content/reference-energy-disaggregation-data-set-redd) · [REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned) · [AMPds/2](http://ampds.org/) · [Dataport](https://ieee-dataport.org/keywords/nilm) · [ECO](http://www.vs.inf.ethz.ch/res/show.html?what=eco-data) · [ENERTALK](https://www.nature.com/articles/s41597-019-0212-5) · [iAWE](https://iawe.github.io/) · [BLUED](http://portoalegre.andrew.cmu.edu:88/BLUED/) · [PLAID](https://www.nature.com/articles/s41597-020-0389-7) · [DRED](https://www.st.ewi.tudelft.nl/~akshay/dred/) · [Georges Hebrail (UCI)](https://archive.ics.uci.edu/ml/datasets/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption) · [GREEND](https://sourceforge.net/projects/greend/) · [HES](https://randd.defra.gov.uk/ProjectDetails?ProjectID=17359&FromSearch=Y&Publisher=1&SearchText=EV0702&SortString=ProjectCode&SortOrder=Asc&Paging=10#Description) · [TraceBase](https://github.com/areinhardt/tracebase) · [IDEAL](https://www.nature.com/articles/s41597-021-00921-y) · [HawkDATA](https://github.com/WZiJ/SenSys24-Hawk)

</details>

<details>
<summary><b>🏢 Commercial buildings datasets (2)</b></summary>
<br>

[COMBED](https://combed.github.io/) · [BLOND](https://www.nature.com/articles/sdata201848)

</details>

<details>
<summary><b>🏭 Industrial datasets (3)</b></summary>
<br>

[Industrial Machines Dataset](https://ieee-dataport.org/open-access/industrial-machines-dataset-electrical-load-disaggregation) · [Aachen Smart Factory](http://www.finesce.eu/Trial_Site_Aachen.html) · [HIPE](https://www.energystatusdata.kit.edu/hipe.php)

</details>

<details>
<summary><b>🧪 Synthetic data generators (5)</b></summary>
<br>

[SynD](https://github.com/klemenjak/SynD/) · [COLD](https://github.com/arx7ti/cold-nilm) · [FIRED](https://github.com/voelkerb/FIRED_dataset_helper) · [SHED](https://nilm.telecom-paristech.fr/shed/) · [smartsim](https://github.com/sustainablecomputinglab/smartsim)

</details>

---

# 📄 Papers

## 🆕 2025 – 2026

> 🔥 The newest wave: **non-stationarity-aware Transformers**, **LLMs with zero training**, **diffusion augmentation**, **multi-scale attention** and **PV injection-aware disaggregation**.

### 📌 MSDCANet: A Multi-Scale Dual-Channel Convolutional Attention Network for Non-Intrusive Load Disaggregation

> Most deep NILM models rely on single-scale features and ignore the multi-scale variations caused by appliance mode transitions, which leads to overfitting and to poor separation of similar appliances. MSDCANet integrates multi-scale feature extraction, adaptive normalization and a multi-scale attention mechanism. On UK-DALE and REDD it beats SotA on MAE / SAE / F1 for several high-energy appliances, under both origin-household and cross-household evaluation.

![Venue](https://img.shields.io/badge/Digital%20Signal%20Processing-2026-blue?style=flat-square)
![Task](https://img.shields.io/badge/Task-regression%20%2B%20state-4caf50?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-DOI-orange?style=flat-square)](https://doi.org/10.1016/j.dsp.2025.105605)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/linfengYang/MSDCANet)

---

### 📌 DualNILM: Energy Injection Identification Enabled Disaggregation with Deep Multi-Task Learning

> Behind-the-meter PV generation breaks NILM's non-negativity assumption — once injection exceeds consumption, disaggregation becomes a severely under-constrained inverse problem. DualNILM is a Transformer-based multi-task framework that unifies sequence-to-point state detection with sequence-to-sequence energy-injection estimation over multi-channel signatures. The authors also release **PV-augmented REDD and UK-DALE** with realistic weather data.

![Venue](https://img.shields.io/badge/arXiv-2025-b31b1b?style=flat-square)
![Topic](https://img.shields.io/badge/PV%20%2F%20prosumer-ready-ffca28?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/abs/2508.14600)
[![Datasets + Code](https://img.shields.io/badge/Datasets%20%2B%20Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/MathAdventurer/PV-Augmented-NILM-Datasets)

---

### 📌 NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-Stationarity

> SotA methods slice household consumption into subsequences, but real smart-meter data is **non-stationary** — distribution drift inside each window wrecks model performance. NILMFormer is a seq2seq Transformer with a subsequence stationarization / de-stationarization scheme plus positional encoding that relies only on timestamp information. It was validated on four real-world datasets and has been deployed as the backbone of **EDF's (Électricité de France) consumption monitoring service**, serving millions of customers.

![Venue](https://img.shields.io/badge/KDD-2025-blueviolet?style=flat-square)
![Deployed](https://img.shields.io/badge/industrial%20deployment-EDF-brightgreen?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/abs/2506.05880)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/adrienpetralia/NILMFormer)

---

### 📌 Prompting Large Language Models for Training-Free Non-Intrusive Load Monitoring

> The first prompt-based NILM framework: it drives LLMs purely through in-context learning, injecting appliance features, timestamps, contextual hints and representative time-series examples into the prompt. Guided by prompts alone, LLMs reach a competitive 0.676 average F1 on unseen REDD households, generalise across houses and even regions without fine-tuning, and emit human-readable explanations for the states they infer.

![Venue](https://img.shields.io/badge/arXiv-2025-b31b1b?style=flat-square)
![Topic](https://img.shields.io/badge/LLM%20%2F%20zero--shot-00b8a9?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/abs/2505.06330)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/SusCom-Lab/llm-for-nilm)

---

### 📌 A Diffusion Model-Based Framework to Enhance the Robustness of Non-Intrusive Load Disaggregation

> Existing NILM models need a lot of labels, yet public datasets are noisy. This work augments training data with a diffusion model tuned to generate multi-state, low-noise load signatures; mixing synthetic with real data improves disaggregation while shrinking the required training set. A tailored loss function and post-processing step further raise noise resistance, improving MAE / SAE / F1 in both intra- and cross-household scenarios.

![Venue](https://img.shields.io/badge/Energy-2025-blue?style=flat-square)
![Topic](https://img.shields.io/badge/data%20augmentation-purple?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-orange?style=flat-square)](https://www.sciencedirect.com/science/article/pii/S0360544225010655)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/linfengYang/DiffusionModel_NILM)

---

<details>
<summary><b>📆 2023 – 2024</b></summary>
<br>

### 📌 Hawk: An Efficient NALM System for Accurate Low-Power Appliance Recognition

> Non-intrusive Appliance Load Monitoring recognises individual appliance usage from the main meter with no indoor sensors, but existing systems trade dataset-construction cost against recognition accuracy — especially for low-power appliances. Hawk runs in two stages: it builds **HawkDATA**, a balanced and diverse dataset collected via balanced Gray code and auto-annotated through a *shared perceptible time* synchronisation strategy (1/71.5 of the collection time, 6.34× more state combinations than the baseline), then recognises events with steady-state differential pre-processing plus voting-based post-processing. Average F1: **93.94% state / 97.07% event** recognition (+47.98% / +11.57% over SotA), and deployment in two real-world scenarios with unseen background appliances reached 96.02% and 94.76% event F1.

![Venue](https://img.shields.io/badge/SenSys-2024-blueviolet?style=flat-square)
![Award](https://img.shields.io/badge/Best%20AE%20Award-gold?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/pdf/10.1145/3666025.3699359)
[![Data + Code](https://img.shields.io/badge/HawkDATA-GitHub-181717?style=flat-square&logo=github)](https://github.com/WZiJ/SenSys24-Hawk)

<p align="center"><img src="./img/20241110-nilm-Hawk.png" alt="Hawk" width="800"></p>

---

### 📌 MATNilm: Multi-Appliance-Task Non-Intrusive Load Monitoring with Limited Labeled Data

> Instead of training one model per appliance on huge labelled corpora, MATNilm proposes a multi-appliance-task framework with a training-efficient **sample augmentation (SA)** scheme: a shared-hierarchical split structure handles regression and classification for each appliance, and a two-dimensional attention mechanism captures spatio-temporal correlations across all of them. With only **one day** of training data, test performance becomes comparable to training on the full dataset, and relative errors drop by more than 50% on average.

![Venue](https://img.shields.io/badge/arXiv-2023-b31b1b?style=flat-square)
![Topic](https://img.shields.io/badge/limited%20labels-ffca28?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/abs/2307.14778)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/jxiong22/MATNilm)

---

### 📌 Graph-Based Dependency-Aware Non-Intrusive Load Monitoring (GRAD-NILM)

> Neural NILM models ignore — or only implicitly characterise — the dependencies between simultaneously running appliances. This work builds a weighted adjacency matrix from prior temporal knowledge between working appliances (adding per-appliance hard dependencies to avoid sparsity), learns non-sequential dependencies through a **graph attention network**, and uses a dilated-convolution encoder–decoder to estimate power and detect states at the same time. Evaluated on UK-DALE with clear gains over the SotA models of the time.

![Venue](https://img.shields.io/badge/PRCV-2023-blueviolet?style=flat-square)
![Topic](https://img.shields.io/badge/graph%20attention-4caf50?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-Springer-01689E?style=flat-square)](https://link.springer.com/chapter/10.1007/978-981-99-8549-4_8)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/zhgqcn/GRAD-NILM)

<p align="center"><img src="./img/GRAD-NILM.png" alt="GRAD-NILM" width="800"></p>

---

### 📌 MSDC: Exploiting Multi-State Power Consumption in Non-Intrusive Load Monitoring Based on a Dual-CNN Model

> MSDC explicitly models an appliance's multiple states and state transitions instead of regressing power alone: one CNN outputs state distributions while the other predicts the power of each state, and conditional random fields (CRF) capture state transitions. On REDD and UK-DALE it significantly outperforms previous SotA models.

![Venue](https://img.shields.io/badge/AAAI-2023-blueviolet?style=flat-square)
![Topic](https://img.shields.io/badge/multi--state%20CRF-4caf50?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-AAAI%20Press-01689E?style=flat-square)](https://ojs.aaai.org/index.php/AAAI/article/view/25636/25408)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/sub-paper/MSDC-NILM)

<p align="center"><img src="./img/MSDC-NILM_2023-11-27_11-33-35.jpg" alt="MSDC" width="800"></p>

</details>

<details>
<summary><b>📂 2020 – 2022</b></summary>
<br>

### 📌 "I do not know": Quantifying Uncertainty in Neural Network Based Approaches for Non-Intrusive Load Monitoring

> Can models tell when they are unsure? The authors evaluate 14 neural variants with uncertainty estimation on REDD and show that uncertainty can be estimated accurately without sacrificing traditional metrics, that different appliances and states differ in how well calibrated they are, and that *recalibration* methods further improve the estimates.

![Venue](https://img.shields.io/badge/BuildSys-2022-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/abs/10.1145/3563357.3564063)
[![Code](https://img.shields.io/badge/Code-Jax-009688?style=flat-square)](https://github.com/VibhutiBansal-11/NILM_Uncertainty/tree/master)

<p align="center"><img src="./img/NILM_Uncertain_2023-11-27_11-42-34.jpg" alt="Uncertainty" width="800"></p>

---

### 📌 Fed-GBM: A Cost-Effective Federated Gradient Boosting Tree for Non-Intrusive Load Monitoring

> A collaborative learning framework combining two-stage voting with node-level parallelism, making co-modelling for NILM both privacy-preserving and cost-effective.

![Venue](https://img.shields.io/badge/CIKM-2022-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/10.1145/3538637.3538840)
[![Code](https://img.shields.io/badge/Code-scikit--learn-F89939?style=flat-square)](https://github.com/FedGBM/FedGBM-NILM)

<p align="center"><img src="./img/fedgbm.png" alt="FedGBM" width="800"></p>

---

### 📌 DeepDFML-NILM: A New CNN-Based Architecture for Detection, Feature Extraction and Multi-Label Classification in NILM Signals

> High-frequency approaches rarely cover the full pipeline. This work delivers an integrated detector → feature extractor → multi-label classifier for high-frequency NILM signals, evaluated on the public LIT-Dataset.

![Venue](https://img.shields.io/badge/IEEE-2022-00629B?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-IEEE-00629B?style=flat-square)](https://ieeexplore.ieee.org/abstract/document/9611234)
[![Code](https://img.shields.io/badge/Code-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/LucasNolasco/DeepDFML-NILM)

<p align="center"><img src="./img/DeepDFML.png" alt="DeepDFML" width="800"></p>

---

### 📌 Thresholding Methods in Non-Intrusive Load Monitoring to Estimate Appliance Status

> Compares three thresholding strategies for turning continuous estimates into on/off statuses, and discusses how they differ across appliances from UK-DALE.

![Venue](https://img.shields.io/badge/Research%20Square-2022-2d8cf0?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-PDF-orange?style=flat-square)](https://www.researchsquare.com/article/rs-1923023/v1)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/UCA-Datalab/nilm-thresholding)

<p align="center"><img src="./img/nilm-threshold.png" alt="Thresholding" width="800"></p>

---

### 📌 Multi-Label Appliance Classification with Weakly Labeled Data for Non-Intrusive Load Monitoring

> An appliance classifier built on a convolutional recurrent neural network trained under weak supervision, cutting the annotation burden for multi-label NILM.

![Venue](https://img.shields.io/badge/IEEE-2022-00629B?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-IEEE-00629B?style=flat-square)](https://ieeexplore.ieee.org/document/9831435)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/GiuTan/Weak-NILM)

<p align="center"><img src="./img/Weak-NILM.png" alt="Weak-NILM" width="800"></p>

---

### 📌 ELECTRIcity: An Efficient Transformer for Non-Intrusive Load Monitoring

> Uses Transformer layers to estimate appliance-level power, relying entirely on attention mechanisms to capture global dependencies between the aggregate signal and each domestic appliance.

![Venue](https://img.shields.io/badge/Sensors-2022-00A0DD?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-Sensors-00A0DD?style=flat-square)](https://www.mdpi.com/1424-8220/22/8/2926)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/ssykiotis/ELECTRIcity_NILM)

<p align="center"><img src="./img/ELECTRIcity.png" alt="ELECTRIcity" width="600"></p>

---

### 📌 Learning to Learn Neural Networks for Energy Disaggregation

> Applies *learning to learn* (meta-learning of the optimiser itself) to energy disaggregation, improving both performance and transferability across datasets.

![Note](https://img.shields.io/badge/MSc%20thesis-2022-lightgrey?style=flat-square)

[![Report](https://img.shields.io/badge/Report-PDF-orange?style=flat-square)](https://github.com/jsobbe/meta_nilm/blob/thesis/thesis_jsobbe.zip)
[![Code](https://img.shields.io/badge/Code-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/jsobbe/meta_nilm/tree/main)

<p align="center"><img src="./img/L2L.png" alt="L2L" width="600"></p>

---

### 📌 Deep Learning-Based Non-Intrusive Commercial Load Monitoring

> Introduces **TTRNet**, a multi-label classification network that learns inter-load correlations through its own structure, and **MLFL**, a loss function purpose-built for multi-label imbalance — together improving accuracy on the hardest commercial loads.

![Venue](https://img.shields.io/badge/ResearchGate-2022-00CCBB?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ResearchGate-00CCBB?style=flat-square)](https://www.researchgate.net/publication/361988541_Deep_Learning-Based_Non-Intrusive_Commercial_Load_Monitoring/figures?lo=1)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/shaoshuai6666/TTRNet)

<p align="center"><img src="./img/TTRNet.png" alt="TTRNet" width="800"></p>

---

### 📌 Improving Non-Intrusive Load Disaggregation through an Attention-Based Deep Neural Network

> Improves the generalisation capability of the architecture by inserting an encoder–decoder with a tailored temporal attention mechanism into the regression subnetwork.

![Venue](https://img.shields.io/badge/Energies-2021-00A0DD?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-Energies-00A0DD?style=flat-square)](https://www.mdpi.com/1996-1073/14/4/847)
[![Code](https://img.shields.io/badge/Code-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/antoniosudoso/attention-nilm)

<p align="center"><img src="./img/attention-NILM.png" alt="attention-NILM" width="800"></p>

---

### 📌 Energy Disaggregation using Variational Autoencoders

> Casts disaggregation into the variational autoencoder framework, where the probabilistic encoder efficiently encodes the information needed to reconstruct the target appliance's consumption.

![Venue](https://img.shields.io/badge/arXiv-2021-b31b1b?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/2103.12177.pdf)
[![Code](https://img.shields.io/badge/Code-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/ETSSmartRes/VAE-NILM)

<p align="center"><img src="./img/VAE-NILM.png" alt="VAE-NILM" width="800"></p>

---

### 📌 Sequence to Point Learning Based on Bidirectional Dilated Residual Network for Non-Intrusive Load Monitoring (BitcnNILM)

> Sequence-to-point learning with bidirectional dilated convolutions for low-frequency NILM; on REDD and UK-DALE it improves both load disaggregation and on/off identification.

![Venue](https://img.shields.io/badge/Electrical%20Power%20%26%20Energy%20Systems-2021-blue?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-orange?style=flat-square)](https://www.sciencedirect.com/science/article/pii/S0142061521000776)
[![Code](https://img.shields.io/badge/Code-Keras-D00000?style=flat-square)](https://github.com/linfengYang/BitcnNILM)

<p align="center"><img src="./img/BitcnNILM_2023-11-27_14-55-50.jpg" alt="BitcnNILM" width="800"></p>

---

### 📌 BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring

> Adapts BERT-style bidirectional Transformers to energy disaggregation in a sequence-to-sequence fashion, with an improved objective function designed specifically for NILM.

![Venue](https://img.shields.io/badge/BuildSys-2020-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/10.1145/3427771.3429390)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/Yueeeeeeee/BERT4NILM)

<p align="center"><img src="./img/BERT4NILM.png" alt="BERT4NILM" width="800"></p>

---

### 📌 Generative Adversarial Networks and Transfer Learning for Non-Intrusive Load Monitoring in Smart Grids

> Two GAN-based approaches for high-accuracy disaggregation that also tackle generalisability — via parameter-sharing transfer learning and via compact shared representations between source and target domains — with a quantitative study of how domain similarity affects the transfer payoff.

![Venue](https://img.shields.io/badge/IEEE-2020-00629B?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-IEEE-00629B?style=flat-square)](https://ieeexplore.ieee.org/document/9302933)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/Awadelrahman/GAN-NILM)

<p align="center"><img src="./img/GAN-NILM.png" alt="GAN-NILM" width="800"></p>

---

### 📌 Exploring Time Series Imaging for Load Disaggregation

> A comparative study of three time-series imaging techniques for NILM — Gramian Angular Fields, Markov Transition Fields and Recurrence Plots.

![Venue](https://img.shields.io/badge/BuildSys-2020-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/10.1145/3408308.3427975)
[![Code](https://img.shields.io/badge/Code-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/BHafsa/image-nilm)

<p align="center"><img src="./img/image-nilm.png" alt="image-nilm"></p>

---

### 📌 On Time Series Representations for Multi-Label NILM

> Leverages **Signal2Vec** dimensionality reduction inside a multi-label NILM system, outperforming another state-of-the-art multi-label baseline on two popular public datasets.

![Venue](https://img.shields.io/badge/Neural%20Computing%20%26%20Applications-2020-01689E?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-Springer-01689E?style=flat-square)](https://link.springer.com/epdf/10.1007/s00521-020-04916-5)
[![Code](https://img.shields.io/badge/Code-scikit--learn-F89939?style=flat-square)](https://github.com/ChristoferNal/multi-nilm)

<p align="center"><img src="./img/online-multi-nilm.png" alt="multi-nilm"></p>

---

### 📌 Improved Appliance Classification in NILM Using Weighted Recurrence Graph and Convolutional Neural Networks

> Introduces the **weighted recurrence graph (WRG)** generated from one cycle of current and voltage — a richer, non-binary image-like representation — and feeds it to a CNN for appliance recognition.

![Venue](https://img.shields.io/badge/Energies-2020-00A0DD?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-Energies-00A0DD?style=flat-square)](https://www.mdpi.com/1996-1073/13/13/3374/htm)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/sambaiga/WRG-NILM)

<p align="center"><img src="./img/WRG-nilm.png" alt="WRG-NILM"></p>

---

### 📌 UNet-NILM: A Deep Neural Network for Multi-Tasks Appliances State Detection and Power Estimation in NILM

> A U-Net variant performing state detection and power estimation jointly, using multi-label learning and multi-target quantile regression.

![Venue](https://img.shields.io/badge/BuildSys-2020-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/10.1145/3427771.3427859)
[![Official](https://img.shields.io/badge/Official-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/sambaiga/UNETNiLM)
[![Reimplementation](https://img.shields.io/badge/Reimpl-PyTorch-lightgrey?style=flat-square)](https://github.com/jonasbuchberger/energy_disaggregation)

<p align="center"><img src="./img/Unet-NILM.png" alt="UNet-NILM"></p>

---

### 📌 Non-Intrusive Load Disaggregation by Convolutional Neural Network and Multilabel Classification

> Recognises appliance activation states with a fully convolutional network, borrowing techniques from semantic image segmentation and multi-label classification.

![Venue](https://img.shields.io/badge/Applied%20Sciences-2020-00A0DD?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-Applied%20Sciences-00A0DD?style=flat-square)](https://www.mdpi.com/2076-3417/10/4/1454)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/lmssdd/TPNILM)

<p align="center"><img src="./img/TP-NILM.png" alt="TPNILM" width="700"></p>

---

### 📌 Multi-Label Learning for Appliances Recognition in NILM using Fryze-Current Decomposition and Convolutional Neural Network

> Applies Fryze power theory to split the current into active and non-active components, converts the decomposition into an image-like representation via Euclidean similarity, and classifies it with a CNN under multi-label learning.

![Venue](https://img.shields.io/badge/Energies-2020-00A0DD?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-Energies-00A0DD?style=flat-square)](https://www.mdpi.com/1996-1073/13/16/4154)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/sambaiga/MLCFCD)

<p align="center"><img src="./img/Fryze-Current.png" alt="Fryze-Current" width="700"></p>

---

### 📌 EdgeNILM: Towards NILM on Edge Devices

> Studies neural network compression schemes against a SotA NILM model and proposes a multi-task learning architecture that compresses the models further — targeting deployment on resource-constrained edge hardware.

![Venue](https://img.shields.io/badge/BuildSys-2020-blueviolet?style=flat-square)
![Topic](https://img.shields.io/badge/edge%20%2F%20compression-4caf50?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/pdf/10.1145/3408308.3427977)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/EdgeNILM/EdgeNILM)

<p align="center"><img src="./img/Edge-NILM-1.png" alt="EdgeNILM" width="700"></p>
<p align="center"><img src="./img/Edge-NILM-2.png" alt="EdgeNILM" width="800"></p>

---

### 📌 eeRIS-NILM: An Open Source, Unsupervised Baseline for Real-Time Feedback Through NILM

> An ideal NILM algorithm should be unsupervised and give real-time feedback, yet such solutions are under-studied. eeRIS pairs a lightweight *Live* algorithm (running continuously) with a heavier, more robust NILM procedure, synchronising their appliance models periodically.

![Venue](https://img.shields.io/badge/IEEE-2020-00629B?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-IEEE-00629B?style=flat-square)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9209866)
[![Code](https://img.shields.io/badge/Code-scikit--learn-F89939?style=flat-square)](https://github.com/eeris-nilm/eeris_nilm)

<p align="center"><img src="./img/eeRIS-NILM.png" alt="eeRIS-NILM" width="600"></p>

</details>

<details>
<summary><b>🏛️ Classics · 2015 – 2019</b></summary>
<br>

### 📌 Deep Learning Based Energy Disaggregation and On/Off Detection of Household Appliances

> Investigates **WaveNet** architectures for energy disaggregation with fast sequence-to-point learning.

![Venue](https://img.shields.io/badge/arXiv-2019-b31b1b?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/abs/1908.00941)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/jiejiang-jojo/fast-seq2point)

<p align="center"><img src="./img/fast-seq2point.png" alt="fast-seq2point" width="800"></p>

---

### 📌 WaveNILM: A Causal Neural Network for Power Disaggregation from the Complex Power Signal

> A causal 1-D WaveNet-style CNN for low-frequency NILM, plus a study showing that all four complex-power components (current, active, reactive and apparent power) are worth using.

![Venue](https://img.shields.io/badge/ICASSP-2019-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/1902.08736.pdf)
[![Code](https://img.shields.io/badge/Code-Keras-D00000?style=flat-square)](https://github.com/picagrad/WaveNILM)

<p align="center"><img src="./img/WaveNILM.png" alt="WaveNILM"></p>

---

### 📌 Transfer Learning for Non-Intrusive Load Monitoring

> The foundational transfer-learning study for NILM, covering both appliance transfer learning (ATL) and cross-domain transfer learning (CTL).

![Venue](https://img.shields.io/badge/arXiv-2019-b31b1b?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/1902.08835.pdf)
[![Code](https://img.shields.io/badge/Code-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/MingjunZhong/transferNILM)

<p align="center"><img src="./img/TransferNILM.png" alt="TransferNILM" width="700"></p>

---

### 📌 Sliding Window Approach for Online Energy Disaggregation Using Artificial Neural Networks

> Two recurrent network architectures with a sliding window, designed for real-time energy disaggregation.

![Venue](https://img.shields.io/badge/SETN-2018-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-01689E?style=flat-square)](https://dl.acm.org/doi/pdf/10.1145/3200947.3201011)
[![Code](https://img.shields.io/badge/Code-Keras-D00000?style=flat-square)](https://github.com/OdysseasKr/online-nilm)

<p align="center"><img src="./img/Short-Seq2Point.png" alt="online-nilm"></p>

---

### 📌 Subtask Gated Networks for Non-Intrusive Load Monitoring

> Combines the main regression network with an on/off classification subtask network through a gating mechanism.

![Venue](https://img.shields.io/badge/AAAI-2019-blueviolet?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/1811.06692.pdf)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/inesylla/energy-disaggregation-DL)

<p align="center"><img src="./img/Subtask-NILM.png" alt="Subtask-NILM"></p>

---

### 📌 Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring

> ⭐ *The canonical seq2point baseline.* The input is a window of the mains aggregate; the output is a single point of the target appliance.

![Venue](https://img.shields.io/badge/AAAI-2017-blueviolet?style=flat-square)
![Classic](https://img.shields.io/badge/most--cited%20baseline-gold?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/1612.09106.pdf)
[![Original](https://img.shields.io/badge/Original-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/MingjunZhong/seq2point-nilm)
[![Reimplementation](https://img.shields.io/badge/Reimpl-PyTorch-EE4C2C?style=flat-square)](https://github.com/mahnoor-shahid/seq2point)

<p align="center"><img src="./img/Seq2Point.png" alt="Seq2Point"></p>

---

### 📌 Neural NILM: Deep Neural Networks Applied to Energy Disaggregation

> 🏛️ *The paper that started deep NILM.* It adapts three DNN architectures to energy disaggregation: a form of RNN called LSTM, denoising autoencoders, and a network that regresses the start time, end time and average power demand of each appliance activation.

![Venue](https://img.shields.io/badge/BuildSys-2015-blueviolet?style=flat-square)
![Classic](https://img.shields.io/badge/foundational%20work-gold?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-ResearchGate-00CCBB?style=flat-square)](https://www.researchgate.net/publication/280329746_Neural_NILM_Deep_Neural_Networks_Applied_to_Energy_Disaggregation)
[![Code](https://img.shields.io/badge/Code-Theano-lightgrey?style=flat-square)](https://github.com/JackKelly/neuralnilm_prototype)

<p align="center"><img src="./img/neural-nilm.png" alt="Neural NILM"></p>

</details>

---

## 📚 Reviews

| 📖 Survey | Links |
| :--- | :--- |
| **Neural Load Disaggregation: Meta-Analysis, Federated Learning and Beyond** — emphasises federated neural NILM, where models are trained locally so sensitive data never leaves the home. <br><img src="./img/FL-NILIM-survey.png" alt="FL-NILM" width="420"> | [![Paper](https://img.shields.io/badge/Paper-Energies-00A0DD?style=flat-square)](https://www.mdpi.com/1996-1073/16/2/991) [![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://github.com/BHafsa/FL-NILM) ![2023](https://img.shields.io/badge/2023-lightgrey?style=flat-square) |
| **Non-Intrusive Load Monitoring: A Review** — broad review of the field shipped with baseline code. | [![Paper](https://img.shields.io/badge/Paper-IEEE-00629B?style=flat-square)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9820770) [![Code](https://img.shields.io/badge/Code-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/pascme05/BaseNILM) ![2022](https://img.shields.io/badge/2022-lightgrey?style=flat-square) |
| **NILM Applications: Literature Review of Learning Approaches, Recent Developments and Challenges** — maps learning-based approaches, recent progress and open challenges. | [![Paper](https://img.shields.io/badge/Paper-ScienceDirect-orange?style=flat-square)](https://www.sciencedirect.com/science/article/abs/pii/S0378778822001220) ![2022](https://img.shields.io/badge/2022-lightgrey?style=flat-square) |
| **Review on Deep Neural Networks Applied to Low-Frequency NILM** — a focused review of deep architectures for low-frequency smart-meter data. | [![Paper](https://img.shields.io/badge/Paper-Energies-00A0DD?style=flat-square)](https://www.mdpi.com/1996-1073/14/9/2390) ![2021](https://img.shields.io/badge/2021-lightgrey?style=flat-square) |

---

## 🚀 Deployment

> Systems that put NILM onto real hardware — from servers down to 18 KB MCUs.

### 📌 A Real-Time Tsetlin Machine-Based Non-Intrusive Load Monitoring System on MCUs

> Reformulates NILM as a Boolean classification task solved by a **Tsetlin Machine**, reaching 90% precision / 96% recall for two-appliance classification on REDD while using only **18 KB of flash** and **0.43 ms** inference latency on an ESP32 — enabling privacy-preserving, fully on-device NILM.

![Venue](https://img.shields.io/badge/arXiv-2026-b31b1b?style=flat-square)
![Device](https://img.shields.io/badge/MCU%20%2F%20TinyML-18%20KB%20flash-brightgreen?style=flat-square)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/abs/2608.18780)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/wuhanstudio/nilm)

---

| Project | Stack | Link |
| :--- | :--- | :--- |
| **Energy Management Using Real-Time NILM** | Arduino + Raspberry Pi | [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/goruck/nilm) |
| **flask-NILM-app-v1** | Flask web app | [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/Selim321/flask-NILM-app-v1) |
| **Inverse Decomposition of Energy Consumption (IDEC)** | C++ | [![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/mieskolainen/IDEC) |

---

# ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zhgqcn/awesome-NILM-with-code&type=Date)](https://star-history.com/#zhgqcn/awesome-NILM-with-code&Date)

---

<div align="center">

**Contributions welcome!** Open an issue or a PR to add your NILM work *with code*.

Made with 💛 for the energy disaggregation community · Licensed under [CC BY 4.0](./LICENSE)

</div>
