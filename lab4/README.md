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

**Expected**: ~90-92% test accuracy

This creates `checkpoints/classifier_baseline/best.pt` used in all experiments.

---

### Exercise 1: OOD Detection

**Goal**: Detect out-of-distribution samples using different scoring methods.

#### Exercise 1.1: OOD Detection Pipeline

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

#### Exercise 1.2: OOD Metrics

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

**Goal**: Generate adversarial examples and evaluate model robustness.

#### Supported Attacks

1. **FGSM** (Fast Gradient Sign Method) - One-step attack
2. **PGD** (Projected Gradient Descent) - Iterative attack
3. **BIM** (Basic Iterative Method) - PGD without random start

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

**Trade-off:**
- Higher adversarial ratio → Better robustness, slightly lower clean accuracy
- Training uses PGD-7 (7 steps) for efficiency

---

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

**Note**: Requires autoencoder checkpoint at `checkpoints/autoencoder/best.pt`

**Expected**: AUROC 0.80-0.85 (slightly worse than energy score but interpretable)

---

## Complete Workflow

```bash
# 1. Train baseline classifier
python experiments/train_classifier.py --epochs 100

# 2. Test OOD detection
python experiments/experiment_ood_detection.py --compare

# 3. Test adversarial robustness
python experiments/experiment_adversarial.py --compare

# 4. Train robust classifier
python experiments/train_robust.py --epochs 100

# 5. Re-test adversarial robustness
python experiments/experiment_adversarial.py \
    --checkpoint ./checkpoints/robust_classifier/best.pt \
    --attack pgd

# 6. (Optional) Train autoencoder
python experiments/train_autoencoder.py --epochs 50

# 7. (Optional) Test autoencoder OOD detection
python experiments/experiment_ood_detection.py \
    --score-method mse \
    --ood-dataset fakedata
```
---

## Visualization Examples

All experiments generate plots in `./plots/`:

1. **OOD Detection**: Score histograms, ROC curves, PR curves
2. **Adversarial**: Clean vs adversarial images, perturbation analysis
3. **Training**: TensorBoard curves for loss/accuracy

---