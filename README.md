# TSMMF-ESWA
This is the official implementation of ESWA - A bidirectional cross-modal transformer representation learning model for EEG-fNIRS multimodal affective BCI

# Abstract
By recognizing or regulating human emotions, the affective brain-computer interfaces (BCIs) could improve human-computer interactions. However, human emotion involves complex temporal-spatial brain networks. Therefore, unimodal brain imaging methods have difficulty to decode complex human emotions. Multimodal brain imaging methods, which capture temporal-spatial multi-dimensional brain signals, have been successfully employed in non-affective BCIs, showing extreme potential to improve the affective BCIs. In order to explore a multimodal fusion model with interpretability and improve emotion recognition performance for multimodal affective BCIs. In this study, we propose a Temporal-Spatial Multimodal Fusion (TSMMF) model, which leverages the bidirectional Cross-Modal Transformer (BCMT) to fuse electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS) multimodal brain signals. Firstly, intra-modal feature extractors and the Self-Attention Transformer were employed to construct joint EEG-fNIRS multimodal representations, reducing inter-modal differences. Secondly, the BCMT was adopted to achieve temporal-spatial multimodal fusion, followed by attention fusion to adaptively adjust the weights of the temporal-spatial multimodal features. Thirdly, modality-specific branches were introduced to preserve the unique features of each modality, then the outputs of all branches were weighted sum for emotion recognition. Furthermore, the model learned the weights of emotion-related brain regions for different modalities. Results showed that: (1) We proposed the first affective BCI based on multimodal brain imaging methods and the emotion recognition outperformed the state-of-the-art methods. (2) An accuracy of 76.15\% was achieved for cross-subject emotion decoding, representing improvements of 6.06\% and 12.44\% compared to EEG and fNIRS unimodal approaches, respectively. (3) The spatial interpretability indicated that: compared to modality-specific branches focusing on common brain regions, whereas the multimodal fusion branch emphasizes differential brain regions related to different emotions. Collectively, our method, inspired by neuroscience, could enhance the development of BCI and multimodal brain signals decoding.

![image](https://github.com/user-attachments/assets/9ca816f6-3e56-41c2-99a6-d485cf1c65eb)

# data
The data is waiting to be open sourced. If you need it, please contact tjzhangshuai@tju.edu.cn. We will provide sample data of 3 subjects.

# contact
If you have any bugs or questions please contact tjzhangshuai@tju.edu.cn
