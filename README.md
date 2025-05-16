# Lettuce NPK Multi-Task Model – INNOFarms.AI Assignment

##  Problem Statement
Build a multi-headed deep learning model using computer vision to:
1. **Classify** lettuce leaf images into nutrient deficiency categories (Nitrogen, Phosphorus, Potassium, or Healthy).
2. **Regress** the **mean RGB intensity** values of the input image.

##  Dataset Overview
The **Lettuce NPK dataset** consists of images split into four classes:
| Class | Count |
|-------|-------|
| FN (Fully Nutritious) | 12 |
| -N (Nitrogen Deficiency) | 58 |
| -P (Phosphorus Deficiency) | 66 |
| -K (Potassium Deficiency) | 72 |

To address class imbalance, extensive augmentation was performed to create up to **750 images per class**.

---

##  Model Architecture

A **dual-headed ResNet50** was used:

- **Backbone:** `ResNet50` pretrained on ImageNet
- **Classification Head:** Fully connected layer for 3 or 4-class classification
- **Regression Head:** Two linear layers to regress RGB values from extracted features

Loss:
```python
Total Loss = λ_cls * CrossEntropyLoss + λ_reg * SmoothL1Loss
