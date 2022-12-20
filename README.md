# awesome-NILM-with-code ![Awesome](https://camo.githubusercontent.com/64f8905651212a80869afbecbf0a9c52a5d1e70beab750dea40a994fa9a9f3c6/68747470733a2f2f617765736f6d652e72652f62616467652e737667)

> A repository of awesome Non-Intrusive Load Monitoring(NILM) with code



# Toolkits

- NILMTK: Non-Intrusive Load Monitoring Toolkit [[PDF](https://arxiv.org/pdf/1404.3878v1.pdf)] [[CODE](https://github.com/nilmtk/nilmtk)]
- NILMTK-Contrib [[PDF](https://dl.acm.org/doi/10.1145/3360322.3360844)] [[Model](https://nipunbatra.github.io/papers/2021/buildsys.pdf)] [[Tensorflow](https://github.com/nilmtk/nilmtk-contrib)]
- NILM-Eval [[Matlab](https://github.com/beckel/nilm-eval)]
- Torch-NILM [[PDF](https://www.mdpi.com/1996-1073/15/7/2647)] [[Pytorch](https://github.com/Virtsionis/torch-nilm)]
- Deep-NILMtk [[PDF](http://nilmworkshop.org/2022/proceedings/nilm22-final4.pdf)] [[Pytorch](https://github.com/BHafsa/deep-nilmtk-v1/tree/master/deep_nilmtk/models/pytorch)] [[Tensorflow](https://github.com/BHafsa/deep-nilmtk-v1/tree/master/deep_nilmtk/models/tensorflow)]
- nilmtk-ukdale: Exploratory Data Analysis [[Pytorch](https://github.com/kehkok/nilmtk-ukdale)]

- NeuralNILM_Pytorch [[Pytorch](https://github.com/Ming-er/NeuralNILM_Pytorch)] 



# Datasets

|               Type                |                           Dataset                            |
| :-------------------------------: | :----------------------------------------------------------: |
|     **Residential datasets**      | [[UK-DALE](https://www.nature.com/articles/sdata20157)] [[REDD](https://energy.duke.edu/content/reference-energy-disaggregation-data-set-redd)] [[REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)] [[AMpds/2](http://ampds.org/)] [[Dataport](https://ieee-dataport.org/keywords/nilm)] [[ECO](http://www.vs.inf.ethz.ch/res/show.html?what=eco-data)] [[ENERTALK](https://www.nature.com/articles/s41597-019-0212-5)] [[iAWE](https://iawe.github.io/)] [[BLUED](http://portoalegre.andrew.cmu.edu:88/BLUED/)] [[PLAID](https://www.nature.com/articles/s41597-020-0389-7)] [[DRED](https://www.st.ewi.tudelft.nl/~akshay/dred/)] [[Georges Hebrail UCI](https://archive.ics.uci.edu/ml/datasets/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption)] [[GREEND](https://sourceforge.net/projects/greend/)] [[HES](https://randd.defra.gov.uk/ProjectDetails?ProjectID=17359&FromSearch=Y&Publisher=1&SearchText=EV0702&SortString=ProjectCode&SortOrder=Asc&Paging=10#Description)] [[TraceBase](https://github.com/areinhardt/tracebase)] [[IDEAL](https://www.nature.com/articles/s41597-021-00921-y)] |
| **Commercial buildings datasets** |            [[COMBED](https://combed.github.io/)]             |
|      **Industrial datasets**      | [[Industrial Machines Dataset](https://ieee-dataport.org/open-access/industrial-machines-dataset-electrical-load-disaggregation)] [[Aachen Smart Factory](http://www.finesce.eu/Trial_Site_Aachen.html)] [[HIPE](https://www.energystatusdata.kit.edu/hipe.php)] |
|        **Synthetic Data**         | [[SynD](https://github.com/klemenjak/SynD/)] [[COLD](https://github.com/arx7ti/cold-nilm)] [[FIRED](https://github.com/voelkerb/FIRED_dataset_helper)] [[SHED](https://nilm.telecom-paristech.fr/shed/)] [[smartsim](https://github.com/sustainablecomputinglab/smartsim)] |

# Papers

## Reviews

- NILM applications: Literature review of learning approaches, recent developments and challenges [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0378778822001220)] [2022]

- Review on Deep Neural Networks Applied to Low-Frequency NILM [[PDF](https://www.mdpi.com/1996-1073/14/9/2390)] [2021]

## Methods

### Thresholding Methods in Non-Intrusive Load Monitoring to Estimate Appliance Status

> We treat three different thresholding methods to perform this task, discussing their differences on various devices from the UK-DALE dataset. [[PDF](https://www.researchsquare.com/article/rs-1923023/v1)] [[Pytorch](https://github.com/UCA-Datalab/nilm-thresholding)] [2022]

<p align="center">
    <img title="" src="./img/nilm-threshold.png" alt="" data-align="center">
</p>




### Multi-Label Appliance Classification with Weakly Labeled Data for Non-Intrusive Load Monitoring

> This paper presents an appliance classification method based on a Convolutional Recurrent Neural Network trained with weak supervision.  [[PDF](https://ieeexplore.ieee.org/document/9831435)] [[Pytorch](https://github.com/GiuTan/Weak-NILM)] [2022]

<p align='center'>
    <img title="" src="./img/Weak-NILM.png" alt="" width="800" data-align="center">
</p>




### ELECTRIcity: An Efficient Transformer for Non-Intrusive Load Monitoring

> Utilizing transformer layers to accurately estimate the power signal of domestic appliances by relying entirely on attention mechanisms to extract global dependencies between the aggregate and the domestic appliance signals. [[PDF](https://www.mdpi.com/1424-8220/22/8/2926)] [[Pytorch](https://github.com/ssykiotis/ELECTRIcity_NILM)] [2022] 

<p align='center'>
    <img title="" src="./img/ELECTRIcity.png" alt="" width="600" data-align="center">
</p>




### Improving Non-Intrusive Load Disaggregation through an Attention-Based Deep Neural Network

> We improve the generalization capability of the overall architecture by including an encoder–decoder with a tailored attention mechanism in the regression subnetwork. The attention mechanism is inspired by the temporal attention. [[PDF](https://www.mdpi.com/1996-1073/14/4/847)] [[Tensorflow](tensorflow)] [2021]

<p align='center'>
    <img title="" src="./img/attention-NILM.png" alt="" width="800" data-align="center">
</p>




### Energy Disaggregation using Variational Autoencoders

> In this paper we propose an energy disaggregation approach based on the variational autoencoders framework. The probabilistic encoder makes this approach an efficient model for encoding information relevant to the reconstruction of the target appliance consumption. [[PDF](https://arxiv.org/pdf/2103.12177.pdf)] [[Tensorflow](https://github.com/ETSSmartRes/VAE-NILM)] [2021]

<p align='center'>
    <img title="" src="./img/VAE-NILM.png" alt="" width="800" data-align="center">
</p>




### BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring

> We propose BERT4NILM, an architecture based on bidirectional encoder representations from transformers (BERT) and an improved objective function designed specifically for NILM learning. We adapt the bidirectional transformer architecture to the field of energy disaggregation and follow the pattern of sequence-to-sequence learning. [[PDF](https://dl.acm.org/doi/10.1145/3427771.3429390)] [[Pytorch](https://github.com/Yueeeeeeee/BERT4NILM)] [2020]

<p align='center'>
    <img title="" src="./img/BERT4NILM.png" alt="" width="800" data-align="center">
</p>




### Exploring Time Series Imaging for Load Disaggregation

> The main contribution presented in this paper is a comparison study between three common imaging techniques: Gramian Angular Fields, Markov Transition Fields, and Recurrence Plots. [[PDF](https://dl.acm.org/doi/10.1145/3408308.3427975)] [[Tensorflow](https://github.com/BHafsa/image-nilm)] [2020]

<p align='center'>
    <img title="" src="./img/image-nilm.png" alt="" data-align="center">
</p>




### On time series representations for multi-label NILM

>The proposed system leverages dimensionality reduction using Signal2Vec, is evaluated on two popular public datasets and outperforms another state-of-the-art multi-label NILM system. [[PDF](https://link.springer.com/epdf/10.1007/s00521-020-04916-5?sharing_token=bTZg6CBADDbWx7UVvztexPe4RwlQNchNByi7wbcMAY4YyOCPZ8jI-u3LyC4lDtEOZIQACACm_MVY_633J4jzg0CtjGEkhvPkzOs5Z-2UGgB1P_m1_4nDnPxtIplmNRaDx7TM52V6MVQYVJPSqJEKpxv1n3RqXoEm1ZpW5amjaaA%3D)] [[Scikit-learn](https://github.com/ChristoferNal/multi-nilm)] [2020]

<p align='center'>
    <img title="" src="./img/online-multi-nilm.png" alt="" data-align="center">
</p>




### Improved Appliance Classification in Non-Intrusive Load Monitoring Using Weighted Recurrence Graph and Convolutional Neural Networks

> We propose an appliance recognition method utilizing the recurrence graph (RG) technique and CNNs. We introduce the weighted recurrent graph (WRG) generation that, given one cycle current and voltage, produces an image-like representation with more values than the binary output created by RG. [[PDF](https://www.mdpi.com/1996-1073/13/13/3374/htm)] [[Pytorch](https://github.com/sambaiga/WRG-NILM)] [2020]

<p align='center'>
    <img title="" src="./img/WRG-nilm.png" alt="" data-align="center">
</p>




### UNet-NILM: A Deep Neural Network for Multi-tasks Appliances State Detection and Power Estimation in NILM

>We propose UNet-NILM for multi-task appliances' state detection and power estimation, applying a multi-label learning strategy and multi-target quantile regression.  [[PDF](https://dl.acm.org/doi/10.1145/3427771.3427859)] [[Official | Pytorch](https://github.com/sambaiga/UNETNiLM)] [[Reimplement | Pytorch](https://github.com/jonasbuchberger/energy_disaggregation)] [2020] 

<p align='center'>
    <img title="" src="./img/Unet-NILM.png" alt=""  data-align="center">
</p>




### Non-Intrusive Load Disaggregation by Convolutional Neural Network and Multilabel Classification

>We address the problem through the recognition of the state of activation of the appliances using a fully convolutional deep neural network, borrowing some techniques used in the semantic segmentation of images and multilabel classification. [[PDF](https://www.mdpi.com/2076-3417/10/4/1454)] [[Pytorch](https://github.com/lmssdd/TPNILM)] [2020]

<p align='center'>
    <img title="" src="./img/TP-NILM.png" alt="" width="700" data-align="center">
</p>




### EdgeNILM: Towards NILM on Edge Devices

> We study different neural network compression schemes and their efficacy on the state-of-the-art neural network NILM method. We additionally propose a multi-task learning-based architecture to compress models further. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3408308.3427977)] [[Pytorch](https://github.com/EdgeNILM/EdgeNILM)] [2020]

<p align='center'>
    <img title="" src="./img/Edge-NILM-1.png" alt="" width='700' data-align="center">
</p>


<p align='center'>
    <img title="" src="./img/Edge-NILM-2.png" alt="" width='800' data-align="center">
</p>




### Deep Learning Based Energy Disaggregation and On/Off Detection of Household Appliances

> We investigate the application of the recently developed WaveNet models for the task of energy disaggregation. [[PDF](https://arxiv.org/abs/1908.00941)] [[Pytorch](https://github.com/jiejiang-jojo/fast-seq2point)] [2019] 

<p align='center'>
    <img title="" src="./img/fast-seq2point.png" alt="" width='800' data-align="center">
</p>




### Wavenilm: A causal neural network for power disaggregation from the complex power signal

> We present a causal 1-D convolutional neural network inspired by WaveNet for NILM on low-frequency data. We also study using various components of the complex power signal for NILM, and demonstrate that using all four components available in a popular NILM dataset (current, active power, reactive power, and apparent power). [[PDF](https://arxiv.org/pdf/1902.08736.pdf)] [[Keras](https://github.com/picagrad/WaveNILM)] [2019]

<p align='center'>
    <img title="" src="./img/WaveNILM.png" alt="" data-align="center">
</p>




### Transfer Learning for Non-Intrusive Load Monitoring

> Appliance transfer learning (ATL) and cross-domain transfer learning (CTL). [[PDF](https://arxiv.org/pdf/1902.08835.pdf)] [[Tensorflow](https://github.com/MingjunZhong/transferNILM)] [2019]

<p align='center'>
    <img title="" src="./img/TransferNILM.png" alt="" width='700' data-align="center">
</p>




### Sliding Window Approach for Online Energy Disaggregation Using Artificial Neural Networks 

> We propose two recurrent network architectures that use sliding window for real-time energy disaggregation.  [[PDF](https://dl.acm.org/doi/pdf/10.1145/3200947.3201011)] [[Keras](https://github.com/OdysseasKr/online-nilm)] [2018]

<p align='center'>
    <img title="" src="./img/Short-Seq2Point.png" alt="" data-align="center">
</p>




### Subtask Gated Networks for Non-Intrusive Load Monitoring

> We propose a subtask gated network that combines the main regression network with an on/off classification subtask network. [[PDF](https://arxiv.org/pdf/1811.06692.pdf)] [[Pytorch](https://github.com/inesylla/energy-disaggregation-DL)] [2018]

<p align='center'>
    <img title="" src="./img/Subtask-NILM.png" alt="" data-align="center">
</p>




### Sequence-to-point learning with neural networks for non-intrusive load monitoring

> We propose sequence-to-point learning, where the input is a window of the mains and the output is a single point of the target appliance. [[PDF](https://arxiv.org/pdf/1612.09106.pdf)] [[Tensorflow](https://github.com/MingjunZhong/seq2point-nilm)] [2017]

<p align='center'>
    <img title="" src="./img/Seq2Point.png" alt="" data-align="center">
</p>




### Neural NILM: Deep Neural Networks Applied to Energy Disaggregation

> We adapt three DNN architectures to energy disaggregation: 1) a form of RNN called LSTM; 2) denoising autoencoders; and 3) a network which regresses the start time, end time and average power demand of each appliance activation. [[PDF](https://www.researchgate.net/publication/280329746_Neural_NILM_Deep_Neural_Networks_Applied_to_Energy_Disaggregation)] [[Theano](https://github.com/JackKelly/neuralnilm_prototype)] [2015]



