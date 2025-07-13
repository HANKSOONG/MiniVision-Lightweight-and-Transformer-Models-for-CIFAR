# MiniVision: Benchmarking CNNs and Vision Transformers on CIFAR-10/100

**MiniVision** benchmarks three computer vision architectures — **ResNet-18**, **EfficientNet-B0**, and **DINOv2 (ViT-B/14)** — using a unified PyTorch pipeline for training, evaluation, and feature visualization.

I first trained all models on **CIFAR-10** to establish strong base performance, and then applied **transfer learning** to adapt each model to the more fine-grained **CIFAR-100** dataset. The goal was to assess both raw accuracy and the models' ability to generalize across domains.

---

##  Highlights

* **Achieved** **98.7%** test accuracy on CIFAR-10 and **91.5%** on CIFAR-100 with DINOv2
* **Applied** transfer learning techniques to adapt models from CIFAR-10 to CIFAR-100
* **Built** a modular pipeline for training, evaluation, and visualization using PyTorch
* **Integrated** early stopping, learning rate scheduling, and model checkpointing
* **Visualized** model embeddings with UMAP; analyzed per-class accuracy and confusion matrices
* **Supported** single-image predictions with dynamic model switching

---

##  Results Overview

| Model           | CIFAR-10 | CIFAR-100 |
| --------------- | -------- | --------- |
| ResNet-18       | 84.5%    | 58.4%     |
| EfficientNet-B0 | 87.3%    | 61.0%     |
| DINOv2-B/14     | 98.7%    | 91.5%     |

> All results are based on the CIFAR-10 and CIFAR-100 datasets from [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html). Pretrained weights were used and fine-tuned where applicable.

---

##  Technical Approach

**Training Workflow:**

* Trained all models on CIFAR-10 first using custom augmentations and early stopping
* Transferred learned weights to CIFAR-100 for fine-tuning
* For DINOv2, froze the first 9 transformer layers to preserve pretrained features

**Optimization:**

* Data Augmentations: random crop, horizontal flip, color jitter
* Optimizers: AdamW for Vision Transformers, SGD for CNNs
* Regularization: weight decay, ReduceLROnPlateau scheduling, early stopping after 5 stagnant epochs
* Mixed precision training enabled for faster GPU performance

**Evaluation:**

* Confusion matrices and per-class accuracy metrics
* UMAP projections of learned feature embeddings
* Cross-dataset comparison to analyze generalization

---

##  Project Structure

```
MiniVision/
├── notebooks/                  # Training notebooks
├── pipeline/                   # Inference notebooks
├── figures/                    # Confusion matrices, UMAPs, prediction samples
├── LICENSE   
├── requirements.txt
└── README.md
```

---

## 📦 Try It Yourself

```bash
git clone https://github.com/HANKSOONG/MiniVision-Lightweight-and-Transformer-Models-for-CIFAR.git
cd MiniVision-Lightweight-and-Transformer-Models-for-CIFAR
pip install -r requirements.txt
```

Then open `pipeline.ipynb` to:

* Load any model (ResNet-18 / EfficientNet-B0 / DINOv2)
* Run inference on CIFAR-10 or CIFAR-100
* Visualize accuracy, confusion matrix, and feature space clustering

---

##  Sample Output

**DINOv2 predictions on CIFAR-10:**

![DINOv2 Predictions](figures/prediction_for_dinov2_cifar10.png)

> DINOv2-B/14 correctly predicted 29 out of 30 samples on CIFAR-10. The misclassification involved visual similarity between frog and cat classes.

**DINOv2 predictions on CIFAR-100:**

![DINOv2 Predictions](figures/prediction_for_dinov2_cifar100.png)

> DINOv2-B/14 correctly predicted 49 out of 50 samples on CIFAR-100. The misclassified image involved a visual overlap between the categories "boy" and "baby," highlighting challenges in fine-grained classification.

**UMAP Embeddings for CIFAR-100:**

* DINOv2-B/14:
  ![UMAP DINO](figures/umap_embeddings/umap_dino_cifar100.png)
* ResNet-18:
  ![UMAP ResNet](figures/umap_embeddings/umap_res_cifar100.png)
* EfficientNet-B0:
  ![UMAP EfficientNet](figures/umap_embeddings/umap_eff_cifar100.png)

> The UMAP visualizations show that DINOv2 learns more compact and well-separated feature clusters, while ResNet-18 and EfficientNet-B0 display significantly less distinct grouping, aligning with their lower performance on CIFAR-100.

---

##  Key Takeaways

* DINOv2 significantly outperforms CNN baselines in both accuracy and feature clarity
* EfficientNet-B0 offers a strong balance between performance and efficiency
* Transfer learning from CIFAR-10 to CIFAR-100 reveals generalization gaps in CNNs

---

##  Dataset and Weights

* CIFAR datasets provided by [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
* Pretrained weights for this project:

  * [CIFAR-10 models](https://drive.google.com/file/d/1--vYxuc0fRE7539StX1Ts9RkAw00_XiZ/view?usp=drive_link)
  * [CIFAR-100 models](https://drive.google.com/file/d/1Qp063eb6V9tSmYsfnJOtNH_fMHCQ_I7M/view?usp=drive_link)
