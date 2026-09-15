

# Awesome-TISR

> Awesome-Thermal-Image-Super-Restoration

# 😸Dataset

## Thermal Image Super-resolution: A Novel Architecture and Dataset

> - This dataset is commonly used in the PBSV competition, providing LR, MR, and HR formats to achieve single-image super-resolution and cross-domain super-resolution.
> - Challenges addressed: improving super-resolution quality, cross-domain image registration.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/TISR-A-novel-dataset.jpg" alt="" width="1000">
</div>




# 🐼Paper

## Thermal Image Enhancement using Convolutional Neural Network

> - https://blog.csdn.net/z243624/article/details/120328561 
> - https://github.dev/ninadakolekar/Thermal-Image-Enhancement 
> - Guided
>
> The choice of thermal cameras provides a rich source of temperature information, less affected by illumination changes or background clutter. However, existing thermal imagers have relatively lower resolution compared to RGB cameras, making it difficult to fully utilize information in recognition tasks. To alleviate this, our goal is to enhance low-resolution thermal images based on a comprehensive analysis of existing methods. To this end, we introduce **Thermal Image Enhancement using Convolutional Neural Networks**, termed TEN, which **directly learns an end-to-end mapping from a single low-resolution image to the desired high-resolution image**. Furthermore, we examine various image domains to identify the best representatives for thermal enhancement. Overall, we present the **first CNN-based thermal image enhancement method utilizing RGB data**. We provide extensive experiments aimed at evaluating image quality and the performance of several object recognition tasks, such as pedestrian detection, visual odometry, and image registration.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/TEN.jpg" alt="" width="800">
</div>

## Infrared image super-resolution using auxiliary convolutional neural

> - https://blog.csdn.net/weixin_42180950/article/details/86661352
> - Guided
>
> Convolutional [neural networks](https://so.csdn.net/so/search?q=神经网络&spm=1001.2101.3001.7020) have been successfully applied to visible image super-resolution methods. In this paper, we propose a CNN-based super-resolution algorithm that utilizes corresponding visible-light images and extends the method to near-infrared images under low-light conditions. Our algorithm first extracts high-frequency components from the extended low-resolution near-infrared images, which are then used as multi-inputs for the CNN. Next, the CNN outputs the high-resolution high-frequency components of the near-infrared input images. Finally, a high-resolution near-infrared image is synthesized by combining the high-resolution high-frequency components with the low-resolution near-infrared image. Simulation results show that the proposed method outperforms state-of-the-art methods in both qualitative and quantitative aspects.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/tisr_acn_1.jpg" alt="" width="1200">
</div>

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/tisr_acn_2.jpg" alt="" width="1200">
</div>

## Cascaded Deep Networks With Multiple Receptive Fields for Infrared Image Super-Resolution

> Instead of using a single complex deep network to reconstruct high-resolution images from low-resolution versions, our method establishes an intermediate step between scale×1 and ×8 (scale ×2), allowing the lost information to be split into two components. The lost information in each component contains similar patterns, so it can be recovered more accurately even using simpler deep networks.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/cascaded_tisr.jpg" alt="" width="1200">
</div>

## Research of infrared image super-resolution reconstruction based on improved FSRCNN

> We propose a multi-channel infrared image super-resolution reconstruction algorithm based on a Fast Super-Resolution Convolutional Neural Network (FSRCNN). The improvements include two aspects: first, multi-scale feature extraction channels are designed according to the characteristics of infrared images to enrich the detailed information of image reconstruction, and residual channels are introduced to improve learning efficiency; second, the activation function is improved to be more representative in the negative region, enhancing network performance.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/improved_FSRCNN.jpg" alt="" width="1200">
</div>

## Deep Networks With Detail Enhancement for Infrared Image Super-Resolution

> A new convolutional network is proposed to improve the spatial resolution of infrared images. Our network recovers fine details by decomposing the input image into low-frequency and high-frequency domains. In the low-frequency domain, we reconstruct the image structure using a deep network. In the high-frequency domain, we recover infrared image details. Additionally, we propose another network to eliminate artifacts. Furthermore, we introduce a novel loss function that leverages visible-light images to enhance the details of infrared images. During the training phase, visible-light images are used to guide the recovery of infrared images; in the testing phase, only infrared images need to be input to obtain super-resolved infrared images. We optimize our deep network using an objective function that penalizes images at different semantic levels with corresponding terms. Moreover, we established a dataset containing paired LR-VIS images of the same scene captured by cameras with both infrared and visible-light sensors sharing the same optical axis.
>
> - Guided
> - Evaluation metrics are not PSNR and SSIM

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/DE_TISR.jpg" alt="" width="1200">
</div>

## DDcGAN: A Dual-Discriminator Conditional Generative Adversarial Network for Multi-Resolution Image Fusion

> This paper proposes a novel end-to-end model called Dual-Discriminator Conditional Generative Adversarial Network (DDcGAN) for fusing infrared and visible-light images of different resolutions. Our method establishes an adversarial game between a generator and two discriminators. The generator aims to generate realistic-looking fused images based on a specifically designed content loss to deceive both discriminators, while the discriminators aim to distinguish the structural differences between the fused image and the two source images, in addition to the content loss. Consequently, the fused image is forced to simultaneously preserve thermal radiation from the infrared image and texture details from the visible-light image. Furthermore, to fuse source images of different resolutions, such as low-resolution infrared images and high-resolution visible-light images, our DDcGAN constrains the downsampled fused image to have attributes similar to those of the infrared image. This avoids blurring of thermal radiation information or loss of visible texture details, which typically occurs in traditional methods. Additionally, we also apply our DDcGAN to fuse multi-modal medical images of different resolutions, such as low-resolution Positron Emission Tomography (PET) images and high-resolution Magnetic Resonance (MR) images.
>
> - https://github.com/jiayi-ma/DDcGAN
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/DDcGAN.jpg" alt="" width="1200">
</div>

## TherISuRNet: A Computationally Efficient Thermal Image Super-Resolution Network

> We propose a super-resolution (SR) architecture for thermal images using a deep neural network, which we call TherISuRNet. We employ a progressive upscaling strategy with asymmetric residual learning in the network, which is computationally efficient for different upscaling factors (e.g., ×2, ×3, and ×4). The proposed architecture includes different modules for low-frequency and high-frequency feature extraction, as well as upscaling blocks.
>
> - https://github.dev/Vishal2188/TherISuRNet---A-Computationally-Efficient-Thermal-Image-Super-Resolution-Network

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/TherIsuRNet.jpg" alt="" width="1300">
</div>

## Pyramidal Edge-maps and Attention-based Guided Thermal Super-resolution

> Guided Super-Resolution (GSR) of thermal images using visible-range images is challenging due to the different spectral ranges across images. This, in turn, implies that texture mismatches between images are significant, manifesting as blurring and ghosting artifacts in super-resolved thermal images. To address this, we propose a GSR algorithm based on pyramidal edge maps extracted from visible images. Our proposed network consists of two subnetworks. The first subnetwork super-resolves the low-resolution thermal image, while the second subnetwork extracts edge maps from the visible image at continuously increasing perceptual scales and integrates them into the super-resolution subnetwork with the help of attention-based fusion. The extraction and integration of multi-level edges allow the super-resolution network to process information from texture to object level step by step, enabling it to more directly identify overlapping edges between input images.
>
> - https://github.com/honeygupta/PAGSR
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/PAGSR.jpg" alt="" width="1200">
</div>

## **Infrared Image Super-Resolution via Heterogeneous Convolutional WGAN**

> Infrared images typically have low resolution. In recent years, deep learning methods have dominated image super-resolution, achieving significant performance on visible images; however, infrared images have received less attention. Infrared images exhibit fewer patterns, making it difficult for deep neural networks to learn diverse features from them. In this paper, we propose a framework incorporating heterogeneous convolution and adversarial training, namely Heterogeneous Kernel-based Super-Resolution Wasserstein GAN (HetSRWGAN), for infrared image super-resolution. The HetSRWGAN algorithm is a lightweight GAN architecture that employs plug-and-play heterogeneous kernel residual blocks. Additionally, a novel loss function utilizing image gradients is adopted, which can be applied to any model.
>
> - ❓ The infrared images in the dataset are RGB images, which are three-channel images 

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/WGAN.jpg" alt="" width="1200">
</div>

## Super-resolution reconstruction of infrared images based on a convolutional neural network with skip connections

> We propose a method for infrared image super-resolution reconstruction based on a convolutional neural network with skip connections. The introduction of global residual learning and local residual learning reduces computational complexity and accelerates network convergence. Multiple convolutional and deconvolutional layers are used to extract and reconstruct infrared image features, respectively. Skip connections and channel fusion are introduced into the network to increase the number of feature maps and facilitate the deconvolutional layers in recovering image details. Compared to other infrared information recovery methods, this approach has a significant advantage in acquiring high-resolution details.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/skip_conn.jpg" alt="" width="1200">
</div>

## Multi-Scale Ensemble Learning for Thermal Image Enhancement

> In this study, we propose a multi-scale ensemble learning method based on convolutional neural networks for thermal image enhancement under different image scale conditions. Integrating thermal images of multiple scales has always been a tricky task, which is why methods have been trained and evaluated for each scale separately. However, this limits the network's proper operation on specific scales. To address this, a novel parallel architecture utilizing multi-scale confidence maps is introduced to train a network that performs well under varying scale conditions.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/multi-scale-ensem.jpg" alt="" width="1200">
</div>

## Joint image fusion and super-resolution for enhanced visualization via semi-coupled discriminative dictionary learning and advantage embedding

> Research combining image fusion and super-resolution is scarce, and the performance of existing methods is far from that of simple image fusion. To address this issue, we propose a joint fusion and super-resolution framework based on discriminative dictionary learning. Specifically, we first jointly learn two pairs of low-rank sparse dictionaries (LRSD) and one transform dictionary. One pair is used to represent the low-rank sparse components of the low-resolution input images, while the other pair is used to reconstruct the high-resolution fusion results; the transform dictionary is used to establish the relationship between the low-resolution and high-resolution image encoding coefficients. To compensate for the loss of details, a Structure Information Compensation Dictionary (SICD) is also learned and used to compensate for the lost information, thereby enhancing the visualization of the final result. To incorporate the advantages of excellent image fusion methods into the fused reconstruction results, a deconvolution-based advantage embedding scheme is proposed.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/joint-image-fusion.jpg" alt="" width="1200">
</div>

## Infrared Image Super-Resolution via Transfer Learning and PSRGAN

> Recent advances in Single Image Super-Resolution (SISR) demonstrate the power of deep learning in achieving better performance. Since re-collecting training data and retraining models for infrared image super-resolution is costly, recovering infrared images with only a few samples remains a significant challenge in the SISR field. To address this, we first propose the Progressive Super-Resolution Generative Adversarial Network (PSRGAN), which consists of a main path and a branch path. A Deep Residual Block (DWRB) is used to represent the features of the infrared image on the main path. Then, a novel Shallow Component Distillation Residual Block (SLDRB) is utilized to extract visible-light image features on the other path; additionally, inspired by transfer learning, we propose a multi-stage transfer learning strategy to bridge the gap between different high-dimensional feature spaces, thereby improving the performance of PSRGAN.
>
> - https://github.com/yongsongH/Infrared_Image_SR_PSRGAN
> - Guided, Unpaired
> - Infrared training → Visible training → Infrared training

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/PSRGAN.jpg" alt="" width="1200">
</div>

## Channel Split Convolutional Neural Network (ChaSNet) for Thermal Image Super-Resolution

> This paper introduces a channel-split convolutional neural network (ChaSNet) for thermal image SR to eliminate redundant features in the network. Utilizing channel splitting to extract common features from low-resolution (LR) thermal images helps preserve high-frequency details in the SR images. We demonstrate the applicability of the proposed SR task network in two different scenarios organized by the PBVS-2021 Thermal SR Challenge, including noise removal (Track-1) and domain transfer (Track-2).
>
> - https://github.com/kalpeshjp89/ChasNet
> - Guided, Cross-domain, both within the infrared domain

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/ChasNet.jpg" alt="" width="1200">
</div>

## Real-World Thermal Image Super-Resolution

> Real-World Super-Resolution (RWSR) is a topic that can be used to address this issue; it uses image processing techniques to enhance the quality of real-world images by reconstructing lost high-frequency information. This work adopts an existing RWSR framework aimed at super-resolving real-world RGB images. The framework estimates the degradation parameters required to generate realistic low-resolution (LR) and high-resolution (HR) image pairs, after which the SR model uses the constructed images to learn the mapping between the LR and HR domains and applies this mapping to new LR thermal images.

## Toward Unaligned Guided Thermal Super-Resolution

> Many thermal imagers are equipped with a high-resolution visible-range camera, which can serve as a guide for super-resolving low-resolution thermal images. However, thermal images and visible-light images form stereoscopic pairs, and the difference in their spectral ranges makes pixel-level alignment of the two images highly challenging. Existing Guided Super-Resolution (GSR) methods are based on aligned image pairs and are therefore unsuitable for this task. In this paper, we attempt to eliminate the necessity of pixel-to-pixel alignment in GSR by proposing two models: the first model employs a correlation-based feature alignment loss to reduce misalignment within the feature space itself, and the second model incorporates a misalignment map estimation block as part of an end-to-end framework, which fully aligns the input images to perform guided super-resolution.
>
> - https://github.com/honeygupta/UGSR
> - Guided, Infrared-Visible

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/UGSR_1.jpg" alt="" width="1200">
</div>

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/UGSR_2.jpg" alt="" width="1200">
</div>

## Heterogeneous Knowledge Distillation for Simultaneous Infrared-Visible Image Fusion and Super-Resolution

> Some methods can simultaneously achieve the fusion and super-resolution of low-resolution images, but due to the lack of guidance from high-resolution fusion results, the improvement in fusion performance is limited. To address this issue, we propose a Heterogeneous Knowledge Distillation Network (HKDnet) with multi-layer attention embedding to achieve the fusion and super-resolution of infrared and visible-light images. Specifically, the proposed method consists of a high-resolution image fusion network (teacher network) and a low-resolution image fusion and super-resolution network (student network). The teacher network mainly fuses high-resolution input images, guiding the student network to acquire the capability for joint fusion and super-resolution. To enable the student network to focus more on the texture details of visible input images, we design a corner embedding attention mechanism. This mechanism integrates channel attention, positional attention, and corner attention to highlight the edges, textures, and structures of visible images. For input infrared images, a dual-frequency attention is constructed by mining the relationships between inter-layer features to emphasize the role of salient targets in the infrared images within the fusion results.
>
> - https://github.com/firewaterfire/HKDnet 
> - https://blog.csdn.net/weixin_43690932/article/details/127947851
> - Guided, Multi-task

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/HKDnet-1.jpg" alt="" width="1200">
</div>

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/HKDnet-2.jpg" alt="" width="1200">
</div>

## Super-resolution reconstruction of thermal imaging of power equipment based on Generative Adversarial Network with Channel Filtering

> This paper achieves enhancement of thermal imaging of power equipment by constructing a Generative Adversarial Network with channel filtering. The network embeds a threshold filtering module in the generator part of the SRGAN (Super-Resolution GAN), utilizing channel information for autonomous threshold learning. The filtering improves training stability on the basis of reducing image noise; meanwhile, by applying edge extraction techniques to strengthen the peak information of images, the network's ability to recover image edges is enhanced.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/CFM-GAN.jpg" alt="" width="1200">
</div>

## A Novel Domain Transfer-Based Approach for Unsupervised Thermal Image Super-Resolution

> This paper proposes a domain transfer strategy to address the limitations of low-resolution thermal sensors and generate higher-resolution images of reasonable quality. The proposed technique adopts a CycleGAN architecture and uses ResNet as the encoder in the generator, along with an attention module and a novel loss function. The network is trained on a multi-resolution thermal image dataset obtained from three different thermal sensors. Results show that at the 2nd CVPR-PBVS-2021 Thermal Imaging Super-Resolution Challenge, the performance benchmark results outperformed state-of-the-art methods.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/CycleGAN.jpg" alt="" width="1200">
</div>

## Single Infrared Image Super-Resolution with Lightweight Self-corrected Attention Network

> We propose a deep learning-based Single Image Super-Resolution (SISR) method for infrared imaging systems. We construct a self-corrected attention network (SCANet) to reconstruct high-resolution (HR) infrared images of targets from low-resolution (LR) images. Specifically, we design a self-corrected attention block (SCAB) that combines upsampling and downsampling operations with the attention module in a recursive and feedback manner. Through SCAB, we train an end-to-end network with infrared images, achieving a reduction in parameters and computational load. Numerous experiments verify the effectiveness of the method. Results indicate that SCANet can achieve single infrared image super-resolution using multiple scaling factors (e.g., x2, x3, and x4).

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/light-self-corr.jpg" alt="" width="1200">
</div>

## Infrared Image Super-Resolution via Generative Adversarial Network with Gradient Penalty Loss

> Infrared thermal imaging technology has gradually developed and is widely used in the fields of measurement and non-destructive testing. However, low-contrast blurry details and expensive acquisition equipment remain obstacles to its further practical application and widespread adoption. This paper proposes a new framework incorporating deep learning techniques, providing a relatively competitive and compatible solution for infrared image super-resolution. First, by leveraging the Wasserstein distance, a Generative Adversarial Network (GAN) detects the radiation information of low-resolution images and automatically converts it to high-resolution images. Second, the discriminator utilizes a gradient penalty loss function to guide the generator to achieve reasonable and acceptable convergence. Evaluated on three widely used infrared datasets, the proposed method demonstrates performance superior to existing methods, achieving more accurate Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/penalty-loss.jpg" alt="" width="1200">
</div>

## CIPPSRNet: A Camera Internal Parameters Perception Network Based Contrastive Learning for Thermal Image Super-Resolution

> Current research lacks effective solutions for training on multi-sensor data, likely driven by pixel misalignment and simple degradation settings. We propose a Camera Internal Parameters Perception Network for infrared thermal image enhancement. Camera Internal Parameters (CIP) are explicitly modeled as feature representations, and LR features are transformed into an intermediate domain containing internal parameter information through the perception of the CIP representation. The mapping between the intermediate domain and the spatial domain of HR features is learned via CIPPSRNet. Furthermore, we introduce **contrastive learning** to optimize the pre-trained camera internal parameter representation network and feature encoder. The proposed network enables a more effective transformation from the LR to the HR domain. Additionally, using contrastive learning improves the network's adaptability and robustness to misaligned data with insufficient pixel matching. Experiments on the PBVS2022 TISR dataset demonstrate that our network achieves state-of-the-art performance on the thermal random resonance task.
>
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/CIPPSRNet-1.jpg" alt="" width="1200">
</div>

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/CIPPSRNet-2.jpg" alt="" width="1200">
</div>

## Multimodal super-resolution reconstruction of infrared and visible images via deep learning

> The image fusion task is transformed into a problem of preserving the structural and intensity ratios of infrared-visible images. A corresponding loss function is designed to widen the weight difference between thermal targets and the background. Additionally, addressing the issue that traditional network mapping functions are unsuitable for natural scenes, single-image super-resolution reconstruction based on a regression network is introduced. Forward generation and backward regression models are considered, employing dual mapping constraints to reduce irrelevant function mapping spaces and approximate ideal scene data.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/Multimodal-tisr.jpg" alt="" width="1200">
</div>

## Edge-Focus Thermal Image Super-Resolution using Generative Adversarial Network

> A method utilizing edge features from high-resolution visible-light images to improve thermal image resolution is proposed. Canny edge detection and thin-line downscaling algorithms are used to generate edge maps from high-resolution visible-light images to assist the super-resolution network. The proposed super-resolution model is designed based on a Generative Adversarial Network architecture for ×2, ×3, and ×4 upsampling. The KAIST dataset is used to train and test the model. Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) are used to evaluate the quality of super-resolved images. After the training process, to demonstrate the effectiveness of edge features, we compared the quality of super-resolved images generated by our proposed method with those from other methods.
>
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/edge-focus.jpg" alt="" width="1200">
</div>

## Meta transfer learning-based super-resolution infrared imaging

> We propose an infrared image super-resolution method based on meta-transfer learning and a lightweight network. We design a lightweight network to learn the mapping between low-resolution and high-resolution infrared images. The network is trained on an external dataset and applies meta-transfer learning on an internal dataset to guide the network toward a sensitive and transferable point. We established an infrared imaging system equipped with an infrared module. The designed network is implemented on a personal computer, and SR images are reconstructed through the trained network. The main contribution of this paper lies in adopting a lightweight network and meta-transfer learning method to obtain infrared super-resolved images with better visual effects. Numerical and experimental results demonstrate that the method achieves infrared image super-resolution with performance superior to four existing image super-resolution methods. This method has practical application value for image super-resolution of mobile infrared devices.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/meta-transfer.jpg" alt="" width="1200">
</div>

## Thermal Image Super‐Resolution Methods Using Neural Networks

> Thermal imaging technology has become widespread in many fields. Since the human eye cannot see the thermal spectrum and light levels are often weak, thermal image analysis has become an indispensable part of medicine, manufacturing, construction, and other industries. Most thermal imagers produce low-resolution images when analyzing object temperatures, complicating the process of analyzing raw thermal spectral maps. Therefore, the problem of improving thermal image quality is highly relevant today. With the development of artificial intelligence and deep learning technologies, new super-resolution methods continue to emerge. These methods are also applicable to thermal imaging processing. This work examines the performance of modern super-resolution methods in the field of thermal vision.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/tisr-NN.jpg" alt="" width="1200">
</div>

## Thermal UAV Image Super-Resolution Guided by Multiple Visible Cues

> In this paper, we propose a novel multi-condition guided network (MGNet) to effectively mine visible-light image information for thermal UAV image SR. High-resolution visible-light UAV images typically contain salient appearance, semantic, and edge information, which play a key role in improving the performance of thermal UAV image SR. Therefore, we design an efficient multi-condition guided module (MGM) to leverage the appearance, edge, and semantic cues from visible-light images to guide thermal UAV image SR. Furthermore, we establish the first benchmark dataset for visible-light guided thermal UAV image SR. It is collected by a multimodal UAV platform and consists of 1025 manually aligned visible-light and thermal image pairs. Extensive experiments on the established dataset show that our MGNet can effectively utilize useful information from visible-light images to improve the SR performance of thermal UAV images, performing well compared to several state-of-the-art methods.
>
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/thermal-UAV.jpg" alt="" width="1200">
</div>

## Infrared image super-resolution method based on dual-branch deep neural network

> Infrared images have lower resolution, lower contrast, and fewer details compared to visible-light images, making their super-resolution processing more difficult. We propose a method based on a deep neural network that includes an image super-resolution branch and a gradient super-resolution branch for reconstructing high-quality super-resolved images from single-frame infrared images. The image SR branch uses a basic structure similar to Enhanced Super-Resolution GAN (ESRGAN) to reconstruct SR images from the initial low-resolution infrared images. The gradient SR branch removes blur, extracts gradient maps, and reconstructs SR gradient maps. To obtain more natural super-resolved images, attention-mechanism-based fusion blocks are employed between these branches. To maintain geometric structures, gradient L1 loss and gradient GAN loss are defined and incorporated.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/Haze_tisr.jpg" alt="" width="1200">
</div>

## Improved Thermal Infrared Image Super-Resolution Reconstruction Method Base on Multimodal Sensor Fusion

> We propose a thermal infrared image super-resolution reconstruction method based on multi-modal sensor fusion, aiming to improve the resolution of thermal infrared images by relying on multi-modal sensor information to reconstruct high-frequency details in the images, thereby overcoming the limitations of imaging mechanisms. First, we design a novel super-resolution reconstruction network consisting of a main feature encoding subnetwork, a super-resolution reconstruction subnetwork, and a high-frequency detail fusion subnetwork to enhance the resolution of thermal infrared images. We designed hierarchical dilated distillation modules and cross-attention transformation modules to extract and transmit image features, enhancing the network's ability to express complex patterns. Then, we propose a hybrid loss function to guide the network in extracting salient features from thermal infrared images and reference images while maintaining accurate thermal information. Finally, we propose a **learning strategy** to ensure the network's high-quality super-resolution reconstruction performance, even in the absence of reference images.
>
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/multimodal-sensor-fusion.jpg" alt="" width="1200">
</div>

## CoReFusion: Contrastive Regularized Fusion for Guided Thermal Super-Resolution

> A novel data fusion framework and regularization technique are proposed for guided thermal image super-resolution. The proposed architecture is computationally inexpensive and lightweight; it maintains performance even when one modality (i.e., high-resolution RGB image or lower-resolution thermal image) is lost, and is designed to be robust in the presence of missing data.
>
> - https://github.com/Kasliwal17/CoReFusion
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/CoReFusion.jpg" alt="" width="1200">
</div>

## MULTI-SPECTRAL SUPER-RESOLUTION OF THERMAL INFRARED DATA PRODUCTS FOR URBAN HEAT APPLICATIONS

> In the context of urban heat analysis, we evaluate recent advances in deep learning-based Single-Image Super-Resolution (SISR) on two multi-spectral datasets. The targets of the datasets are respectively Land Surface Temperature (LST) products and Atmosphere Top-Of-Atmosphere (TOA) LWIR radiation. In this process, we demonstrate the potential of generative modeling methods, particularly Super-Resolution GANs (SRGAN), to improve the spatial resolution of thermal data products. We extend the original SRGAN model with additional bands from the visible spectrum to increase spatial resolution by four times, and estimate the model's **prediction uncertainty**. Compared to bilinear upsampling, this Multi-Spectral Super-Resolution (MSSR) method improves the Peak Signal-to-Noise Ratio (PSNR) by 3dB to 6dB.
>
> - Guided

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/urban-heat.jpg" alt="" width="1200">
</div>

## Thermal image super-resolution via multi-path residual attention network

> The performance of existing deep SR methods is limited by the narrow receptive fields of single small convolutional kernels (e.g., 3×3). This paper proposes a thermal imaging SISR deep network, MPRANet, combining multi-path residual and attention blocks. Specifically, the innovatively designed multi-path residual block consists of parallel depth-wise separable convolution paths formed by convolutional kernels of different sizes, used to extract local fine-grained and global large-scale features, effectively enhancing the capacity of MPRANet. Meanwhile, the attention block is formed by cascading channel attention and spatial attention modules to sequentially rescale features in channel and spatial dimensions. A mixed data augmentation (MoDA) strategy is proposed to improve MPRANet's performance without increasing computational overhead. MoDA fully utilizes various pixel-domain data augmentation methods to enhance the generalization capability of MPRANet.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/MODA.jpg" alt="" width="1200">
</div>

## SRDRN-IR: A Super Resolution Deep Residual Neural Network for IR Images

> To address the super-resolution problem of infrared images, we propose a deep neural network structure called SRDRN. SRDRN utilizes a channel-splitting concept with residual learning to achieve computationally efficient super-resolution. The feasibility of the proposed design is verified through analysis using available thermal image datasets.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/SRDRN_IR.jpg" alt="" width="1200">
</div>

## Super-Resolution Infrared Imaging via Degraded Information Distillation Network

> We propose an unsupervised super-resolution infrared imaging method using a degraded information extraction network. We design a network model that progressively extracts degraded information to learn more degraded information with discriminative features. We use dual-attention convolution to achieve adaptive features in channels and spatial dimensions. We use sub-pixel convolution to reconstruct infrared images. We train our model using infrared images and systematically evaluate the proposed method.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/DIDSRN.jpg" alt="" width="1200">
</div>

## Real-infraredSR: real-world infrared image super-resolution via thermal imager

> Most existing super-resolution methods do not achieve satisfactory reconstruction performance in real-world scenes when using synthetic data obtained via bilinear interpolation. To address this issue, this paper innovatively proposes a real-world infrared dataset with different resolutions based on cooled thermal detectors and infrared zoom lenses, enabling the network to acquire more realistic details. By adjusting the infrared zoom lens to capture images under different fields of view, scale and brightness alignment between high-resolution (HR) and low-resolution (LR) images is achieved. This dataset can be used for infrared image super-resolution with an upsampling scale of 2. To effectively learn the complex features of infrared images, an asymmetric residual block structure is proposed, which effectively reduces the number of parameters and improves network performance. Finally, to address slight misalignment issues during the preprocessing stage, context loss and perceptual loss are introduced to enhance visual performance.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/Real-infraredSR.jpg" alt="" width="1200">
</div>

## CMRFusion: A cross-domain multi-resolution fusion method for infrared and visible image fusion

> We propose CMRFusion, a cross-domain multi-resolution fusion method for infrared and visible image fusion based on an autoencoding network and a cross-domain attention fusion strategy. An autoencoding network is employed, where the encoder network extracts deep multi-scale features and the decoder network reconstructs the images. A cross-domain attention fusion strategy is adopted to promote the preservation of texture details from one of the source images. The method first upscales the low-resolution infrared image using a simple bilinear strategy to match the resolution of the source images. Then, the encoder network extracts features from both infrared and visible images. Based on the extracted infrared image features, a cross-domain attention fusion strategy is used to supplement details from the extracted visible image features to obtain fused features, which are then reconstructed into high-resolution infrared images using the first decoder network. Finally, the encoder network extracts features from the visible and reconstructed infrared images. Based on the extracted visible image features, a cross-domain attention fusion strategy is used to supplement details from the extracted features of the reconstructed high-resolution infrared images to obtain fused features, which are reconstructed into the final fusion results using the second decoder network.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/CMRFusion.jpg" alt="" width="1200">
</div>

## Infrared and Visible Image Fusion via Test-Time Training

> Infrared and Visible Image Fusion (IVIF) is a technology widely used in instrument-related fields. Its purpose is to extract contrast information from infrared images, texture details from visible-light images, and combine these two types of information into a single image. Most autoencoder-based methods train networks on natural images (such as MS-COCO) and then test the models on IVIF datasets. Such methods suffer from domain shift problems and struggle to generalize well in real-world scenarios. To this end, we propose a self-supervised Test-Time Training (TTT) method to achieve better fusion effects during testing. Specifically, we develop a novel self-supervised loss function to evaluate the quality of the fusion results. This loss function guides the network to improve fusion quality by optimizing model parameters through a few iterations during testing. Furthermore, instead of manually designing fusion strategies, we utilize a fusion adapter to automatically learn fusion rules. Experimental comparisons on two public IVIF datasets validate that the proposed method outperforms existing methods both subjectively and objectively.

<div style="display: flex; justify-content: center;">
    <img title="" src="./img/IVIF-TTT.png" alt="" width="1200">
</div>
