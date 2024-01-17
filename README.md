# masterpraktikum_ssl_segmentation
Repository accompanying the 2024 Masterpraktikum Project: Integrative Analysis of Histological Images and Gene Expression Data for Pathological Region Classification

### Introduction
Self-supervised learning is a subset of machine learning where a model learns to understand and represent data by teaching itself. This approach is different from supervised learning, where models are trained with labeled data, and unsupervised learning, where models identify patterns without any labels. In self-supervised learning, the system generates its own labels from the input data, usually through a pretext task, and then uses these labels to learn a representation of the data. Examples of pretext tasks include predicting a missing part of an image or predicting the next word in a sentence. The features learned in the pretext task can be used for downstream tasks, like classification or segmentation, often with minimal additional training.

Pathological image segmentation, such as segmenting cells or tissues in medical scans or histological images, is a challenging task, often complicated by the scarcity of labeled data. Labeling medical images requires expert knowledge and is time-consuming and expensive. This is where self-supervised learning can be particularly beneficial. 
This project aims to develop a model that integrates histological images and gene expression data obtained through spatial omics technologies like Visium. The primary goal is to assign each spot into specific pathological regions in order to study spatial patterns leading to disease development. These regions can further be examined for identification of new biomarkers or for drug target discovery. This project will leverage self-supervised learning to effectively utilize both labeled and unlabeled data, addressing the common challenge of limited labeled data sets in medical image research.
Names of the supervisors

### Supervisors
- Soroor Zadeh soroor.hediyehzadeh@helmholtz-munich.de
- Merel Kuijs merelsentina.kuijs@helmholtz-munich.de
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

### Number of students that can be supervised
Teams of up to 4 students are welcome to join the project. Ideal pre-requisite skills are familiarity with machine learning and deep learning. Experience with the PyTorch package in the Python ecosystem is a useful asset but not a requirement. 


