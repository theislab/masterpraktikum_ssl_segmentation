# 2024 Masterpraktikum
Repository accompanying the 2024 Masterpraktikum Project: Integrative Analysis of Histological Images and Gene Expression Data for Pathological Region Classification

### Introduction
Self-supervised learning is a subset of machine learning where a model learns to understand and represent data by teaching itself. This approach is different from supervised learning, where models are trained with labeled data, and unsupervised learning, where models identify patterns without any labels. In self-supervised learning, the system generates its own labels from the input data, usually through a pretext task, and then uses these labels to learn a representation of the data. Examples of pretext tasks include predicting a missing part of an image or predicting the next word in a sentence. The features learned in the pretext task can be used for downstream tasks, like classification or segmentation, often with minimal additional training.

Pathological image segmentation, such as segmenting cells or tissues in medical scans or histological images, is a challenging task, often complicated by the scarcity of labeled data. Labeling medical images requires expert knowledge and is time-consuming and expensive. This is where self-supervised learning can be particularly beneficial. 
This project aims to develop a model that integrates histological images and gene expression data obtained through spatial omics technologies like Visium. The primary goal is to assign each spot into specific pathological regions in order to study spatial patterns leading to disease development. These regions can further be examined for identification of new biomarkers or for drug target discovery. This project will leverage self-supervised learning to effectively utilize both labeled and unlabeled data, addressing the common challenge of limited labeled data sets in medical image research.

### Supervisors
#### PhD students
- Merel Kuijs merelsentina.kuijs@helmholtz-munich.de
- Soroor Hediyeh-zadeh soroor.hediyehzadeh@helmholtz-munich.de
- Rushin Gindra rushin.gindra@helmholtz-munich.de
- Till Richter till.richter@helmholtz-munich.de
- Korbinian Träuble korbinian.traeuble@helmholtz-munich.de
- Justus Wettich justus.wettich@tum.de
#### PIs
Please don't reach out to the PIs directly. If you have questions about the project, ask the PhD students first.
- Matthias Heinig matthias.heinig@helmholtz-muenchen.de
- Fabian Theis fabian.theis@helmholtz-munich.de

### Proposed period and modality 
#### Lecture period (kickoff around March-April):
- Hybrid or online meetings possible
- Once a week on Tuesdays or Thursdays (flexible)
- 3-4 students working together

#### Block part:
- 3 weeks
- July or late August/early September (after the exam period)

### Methods that the students will apply in the project
- Preprocessing spatial transcriptomics datasets
- Self-supervised learning for histological images and gene expression data
- Classifier development
- Ablation studies

### Milestones
1. Develop self-supervised learning (SSL) models for both histological images and gene expression data to learn robust features from unlabeled datasets
    1. SSL for histological images: Develop pretext tasks such as predicting a segment of an image or colorizing grayscale images.
    2. SSL for gene expression data: Implement pretext tasks like predicting missing gene expression levels.
2. Feature integration and classifier development: Combine learned features from both domains, for example using concatenation or more advanced fusion methods, to build a downstream classifier.
3. Fine-tuning on labeled data: Use the limited labeled data to fine-tune the classifier for accurate pathological region classification. Validate the model's performance using metrics like accuracy and Rand index.
4. Apply ablation studies to assess the importance of histological and gene expression features in the classification of pathological regions.
5. Expected outcomes include:
    1. A robust model capable of accurately classifying pathological regions.
    2. An understanding of the contribution of histological and gene expression features to the overall model performance.
    3. A framework that can be adapted to other similar integrative analysis tasks in computational biology research.


### Number of students that can be supervised
Teams of up to 4 students are welcome to join the project. Ideal pre-requisite skills are familiarity with machine learning and deep learning. Experience with the PyTorch package in the Python ecosystem is a useful asset but not a requirement. 


