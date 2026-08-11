# Brain-Tumour-Classification-using-Vision-Transformer-and-Explainable-AI-Techniques

Brain Tumor Multi-Class Classification using Vision Transformer (ViT), Explainable AI (XAI), and K-Fold Cross Validation for interpretable and robust MRI diagnosis. This project was done during my research internship in NIT Silchar
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  Dataset is large

Developed a ViT-B/16 brain tumor classifier on 7,023 MRI images (4 classes: glioma, meningioma, pituitary, no tumor), achieving 99.94% training accuracy and 98.51% test accuracy with 5-fold cross-validation.
•  Outperformed ResNet-50 (94.25%), AlexNet (92.18%), and DenseNet-201 (95.84%); achieved 99.00% specificity and 98.00% sensitivity, confirming superior diagnostic reliability.
•  Integrated Grad-CAM and LIME to highlight clinically relevant tumor regions, improving model transparency and reducing the black-box nature for clinical decision-making.
•  Trained with Adamw optimizer \& GELU activation function for 120 epochs (batch 64, LR 0.001) with flipping, ±10° rotation, and brightness augmentation for robust generalization across diverse MRI samples.
. ablation study was done

