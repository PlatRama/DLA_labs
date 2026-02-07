# Deep Learning Applications - Lab 4: OOD Detection and Adversarial Learning

Implementation of Lab4 exercises on Out-of-Distribution (OOD) detection and adversarial robustness. The lab covers OOD scoring methods, adversarial attacks, robust training, and autoencoder-based detection.

## Project Structure

```
lab4_rewrite/
├── configs/
│   └── config.py              # Configuration classes
├── models/
│   └── ood_models.py          # CNN classifier and autoencoder
├── utils/
│   ├── data_utils.py          # Dataset loading (ID/OOD)
│   ├── adversarial_utils.py   # Adversarial attacks
│   ├── ood_metrics.py         # OOD detection metrics
│   └── trainer.py             # Training loops
├── experiments/
│   ├── train_classifier.py            # Train baseline classifier
│   ├── experiment_ood_detection.py    # Exercise 1: OOD detection
│   ├── experiment_adversarial.py      # Exercise 2: Adversarial attacks
│   ├── train_robust.py                # Exercise 3: Robust training
│   └── train_autoencoder.py           # Autoencoder for OOD
├── plots/                     # Visualization plots
└── README.md
```

## Exercises

### Preliminary: Train Baseline Classifier

Train a CNN classifier on CIFAR-10 for subsequent experiments:

```bash
python experiments/train_classifier.py --epochs 100
```

**Test accuracy**: 83.93%

This creates `checkpoints/classifier_baseline/best.pt` used in all experiments.

---

### Exercise 1: OOD Detection

Exercise 1 implements **Out-of-Distribution (OOD) detection**: identifying samples that don't belong to the training distribution. We compare three scoring methods on two OOD datasets.

**Goal**: Detect out-of-distribution samples using different scoring methods.

### OOD Detection Results Comparison

### OOD Dataset: FakeData (Random Noise)
| Method      | AUROC  | FPR@95 | AUPR   | Accuracy |
|:------------|:-------|:-------|:-------|:---------|
| max_softmax | 0.6779 | 0.9989 | 0.7875 | 72.80%   |
| energy      | 0.7038 | 0.9999 | 0.8117 | 75.58%   |
| max_logit   | 0.6990 | 0.9999 | 0.8078 | 75.22%   |

### OOD Dataset: CIFAR-100 Subset (People Classes)
| Method      | AUROC  | FPR@95 | AUPR   | Accuracy |
|:------------|:-------|:-------|:-------|:---------|
| max_softmax | 0.7646 | 0.8520 | 0.9849 | 95.24%   |
| max_logit   | 0.7418 | 0.8580 | 0.9828 | 95.24%   |
| energy      | 0.7356 | 0.8640 | 0.9824 | 95.24%   |

---

## Visualization



 **Fake Dataset**
<div>
  <img src="plots/lab4/score_dist_fakedata_max_softmax.png" alt="Img1" width="450">
  <img src="plots/lab4/roc_pr_fakedata_max_softmax.png" alt="Img1" width="450">
</div>

**CIFAR-100 Subset**
<div>
  <img src="plots/lab4/score_dist_cifar100_subset_max_softmax.png" alt="Img1" width="450">
  <img src="plots/lab4/roc_pr_cifar100_subset_max_softmax.png" alt="Img1" width="450">
</div>

Each experiment generates plots in `./plots/lab4`:

### Key Findings

### 1. CIFAR-100 Subset Performance

**Best Method**: Max Softmax (AUROC: 0.7646)

| Metric       | max_softmax | max_logit | energy |
|--------------|-------------|-----------|--------|
| **AUROC**    | **0.7646**  | 0.7418    | 0.7356 |
| **FPR@95**   | **0.8520**  | 0.8580    | 0.8640 |
| **AUPR**     | **0.9849**  | 0.9828    | 0.9824 |
| **Accuracy** | 95.24%      | 95.24%    | 95.24% |

**Analysis**:
- All methods achieve excellent **95.24% accuracy** with optimal threshold
- Max softmax slightly outperforms others (AUROC: 0.76 vs 0.74)
- High AUPR (0.98+) indicates good precision-recall trade-off
- **Moderate FPR@95**: ~85% false positives at 95% TPR

---

### 2. FakeData Performance 📉

**Best Method**: Energy (AUROC: 0.7038) - but all methods struggle!

| Metric       | max_softmax | max_logit | energy     |
|--------------|-------------|-----------|------------|
| **AUROC**    | 0.6779      | 0.6990    | **0.7038** |
| **FPR@95**   | 0.9989      | 0.9999    | 0.9999     |
| **AUPR**     | 0.7875      | 0.8078    | **0.8117** |
| **Accuracy** | 72.80%      | 75.22%    | **75.58%** |

**Analysis**:
- **Poor performance across all methods** (AUROC: 0.68-0.70)
- Nearly **100% FPR@95**: Model assigns high confidence to random noise!
- Energy marginally better (AUROC: 0.70 vs 0.68)
- Only **75% accuracy** (vs 95% on CIFAR-100 subset)

---

#### OOD Detection Pipeline

```bash
# Using max softmax score
python experiments/experiment_ood_detection.py \
    --ood-dataset cifar100_subset \
    --score-method max_softmax

# Using max logit score
python experiments/experiment_ood_detection.py \
    --ood-dataset cifar100_subset \
    --score-method max_logit

# Using energy score
python experiments/experiment_ood_detection.py \
    --ood-dataset fakedata \
    --score-method energy
```

#### OOD Metrics

The script automatically computes and displays:
- **AUROC**: Area Under ROC Curve
- **FPR@95**: False Positive Rate at 95% True Positive Rate
- **AUPR**: Area Under Precision-Recall Curve
- **Detection Error**: Minimum (FPR + FNR) / 2

**Compare all methods:**
```bash
python experiments/experiment_ood_detection.py --compare
```

#### OOD Datasets Available

1. **CIFAR-100 Subset**: People classes from CIFAR-100
2. **FakeData**: Random noise images
3. **SVHN**: Street View House Numbers

---

### Exercise 2: Adversarial Attacks
Exercise 2 focuses on **adversarial attacks**: generating imperceptible perturbations that fool neural networks. This experiment compares three attack methods (FGSM, BIM, PGD) across different perturbation budgets.

**Goal**: Generate adversarial examples and evaluate model robustness.

#### Supported Attacks

1. **FGSM** (Fast Gradient Sign Method) - One-step attack
2. **PGD** (Projected Gradient Descent) - Iterative attack
3. **BIM** (Basic Iterative Method) - PGD without random start


### Attack Comparison Table

### Method: FGSM (Fast Gradient Sign Method)
| Epsilon | Clean Acc | Adv Acc | Attack Success |
|:--------|:----------|:--------|:---------------|
| 0.007   | 85.10%    | 69.40%  | 16.40%         |
| 0.015   | 85.10%    | 51.00%  | 35.10%         |
| 0.031   | 85.10%    | 25.00%  | 61.60%         |

---

### Method: BIM (Basic Iterative Method)
| Epsilon | Clean Acc | Adv Acc | Attack Success |
|:--------|:----------|:--------|:---------------|
| 0.007   | 85.10%    | 68.60%  | 17.40%         |
| 0.015   | 85.10%    | 46.00%  | 40.60%         |
| 0.031   | 85.10%    | 12.60%  | 75.00%         |

---

### Method: PGD (Projected Gradient Descent)
| Epsilon | Clean Acc | Adv Acc | Attack Success |
|:--------|:----------|:--------|:---------------|
| 0.007   | 85.10%    | 68.60%  | 17.40%         |
| 0.015   | 85.10%    | 46.00%  | 41.00%         |
| 0.031   | 85.10%    | 11.30%  | 76.10%         |

---

### Key Findings

### 1. Attack Strength Hierarchy

**PGD > BIM > FGSM** (at higher epsilon values)

| Epsilon | FGSM Success | BIM Success | PGD Success | Winner         |
|---------|--------------|-------------|-------------|----------------|
| 0.007   | 16.40%       | 17.40%      | 17.40%      | BIM/PGD (tied) |
| 0.015   | 35.10%       | 40.60%      | 41.00%      | **PGD**        |
| 0.031   | 61.60%       | 75.00%      | 76.10%      | **PGD**        |

**Analysis**: 
- At low epsilon (0.007), all methods perform similarly
- As epsilon increases, iterative methods (BIM/PGD) significantly outperform FGSM
- PGD is consistently the strongest attack, especially at ε=0.031 (76.1% success!)

### 2. Epsilon Impact

**Attack success increases dramatically with perturbation budget:**

| Method | ε=0.007 | ε=0.015 | ε=0.031 | Increase (0.007→0.031) |
|--------|---------|---------|---------|------------------------|
| FGSM   | 16.4%   | 35.1%   | 61.6%   | **+45.2%**             |
| BIM    | 17.4%   | 40.6%   | 75.0%   | **+57.6%**             |
| PGD    | 17.4%   | 41.0%   | 76.1%   | **+58.7%**             |

**Key Insight**: Doubling epsilon from 0.015 to 0.031 nearly doubles attack success for all methods!

#### Run Attacks

```bash
# FGSM attack
python experiments/experiment_adversarial.py \
    --attack fgsm \
    --epsilon 0.031

# PGD attack (stronger)
python experiments/experiment_adversarial.py \
    --attack pgd \
    --epsilon 0.031

# Compare all methods
python experiments/experiment_adversarial.py --compare
```

### Exercise 3: Adversarial Training (Robust Classifier)

**Goal**: Train a robust classifier using adversarial examples.

```bash
# Train with 50% adversarial examples
python experiments/train_robust.py \
    --epochs 100 \
    --adv-ratio 0.5 \
    --epsilon 0.031

# Train with 100% adversarial examples
python experiments/train_robust.py \
    --epochs 100 \
    --adv-ratio 1.0 \
    --epsilon 0.031
```


### Optional: Autoencoder for OOD Detection

**Goal**: Use reconstruction error for OOD detection.

#### Train Autoencoder

```bash
python experiments/train_autoencoder.py \
    --checkpoint ./checkpoints/classifier_baseline/best.pt \
    --freeze-encoder \
    --epochs 50
```

The autoencoder:
- Uses frozen encoder from pre-trained classifier
- Trains only decoder
- Lower reconstruction error = more likely ID

#### Use for OOD Detection

```bash
python experiments/experiment_ood_detection.py \
    --score-method mse \
    --ood-dataset cifar100_subset
```