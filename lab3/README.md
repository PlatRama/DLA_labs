# Deep Learning Applications - Lab 3: Transformers

Implementation of Lab3 exercises on working with Transformers in the HuggingFace ecosystem. The lab focuses on sentiment analysis using DistilBERT with different training strategies.
This project explores sentiment analysis using the [Cornell Rotten Tomatoes movie review dataset](https://huggingface.co/datasets/rotten_tomatoes).  


<details>
<summary><strong>Project Structure</strong></summary>

```
lab3_rewrite/
├── configs/
│   └── config.py              # Configuration classes
├── models/
│   └── sentiment_models.py    # Sentiment classification models
├── utils/
│   ├── data_utils.py          # Dataset loading and preprocessing
│   └── trainer.py             # Training loops
├── experiments/
│   ├── experiment_baseline_svm.py       # Exercise 1.3: SVM baseline
│   ├── experiment_baseline.py           # Exercise 2: Baseline with PyTorch
│   └── experiment_finetuning.py         # Exercise 2.3 & 3.1: Fine-tuning
├── features/                  # Extracted features
└── README.md
```
</details>

## Dataset

- Name: Rotten Tomatoes (via HuggingFace Datasets)
- Size: ~10,662 sentences
- Classes: Binary sentiment classification
    - 0 → Negative
    - 1 → Positive

### Dataset Split & Features Shape

| Split | Number of Samples | Feature Dimension | Shape |
| :--- | :--- | :--- | :--- |
| **Train** | 8530 | 768 | (8530, 768) |
| **Validation** | 1066 | 768 | (1066, 768) |
| **Test** | 1066 | 768 | (1066, 768) |


##  Stable Baseline with DistilBERT + SVM

I first build a stable baseline by using DistilBERT as a frozen feature extractor and training a Linear SVM classifier on top.


### Implementation details 

| Component             | Description                                                                          |
|-----------------------|--------------------------------------------------------------------------------------|
| **Model**             | [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) |
| **Feature Extractor** | Use `[CLS]` embedding from `last_hidden_state[:, 0, :]`                              |
| **Classifier**        | Linear SVM (`LinearSVC` from scikit-learn)                                           |
| **Batch Size**        | 64 (for feature extraction)                                                          |
| **Training**          | Only SVM is trained, DistilBERT remains frozen                                       |
| **Evaluation**        | Accuracy on val and test                                                             |


## Full Fine-tuning & LoRA-efficient Fine-tuning 
I also implemented full fine-tuning of DistilBERT and LoRA-efficient fine-tuning for comparison.

### model details 
| Method           | Trainable Parameters | Total Parameters |
|------------------|----------------------|------------------|
| Full Fine-tuning | 68M (all)            | 68M              |
| LoRA Fine-tuning | 739,586              | 68M              |


### Training Setup

| Parameter         | Value |
|:------------------|:------|
| **Batch size**    | 64    |
| **Epochs**        | 10    |
| **Optimizer**     | AdamW |
| **Learning rate** | 1e-4  |
| **Weight decay**  | 0.01  |
| **LoRA r**        | 16    |
| **LoRA alpha**    | 32    |
| **LoRA dropout**  | 0.1   |


### Results 
| Method           | Validation Accuracy | Test Accuracy |
|------------------|---------------------|---------------|
| Base line SVM    | 82.22%              | 79,83%        |
| Full Fine-tuning | 85.27%              | 83.86%        |
| LoRA Fine-tuning | 82.74%              | 81.14%        |


## How to run exercise

### Exercise 1: Sentiment Analysis (Warm-up)

#### Exercise 1.1 & 1.2: Dataset and Pretrained Model
- Load Rotten Tomatoes dataset
- Explore DistilBERT model and tokenizer

#### Exercise 1.3: Stable Baseline
**Goal**: Extract features from DistilBERT and train SVM classifier

```bash
python experiments/experiment_baseline_svm.py
```

### Exercise 2: Fine-tuning DistilBERT

#### Exercise 2.1: Token Preprocessing
Handled automatically by `data_utils.py`

#### Exercise 2.2 & 2.3: Fine-tuning
**Goal**: Fine-tune DistilBERT on Rotten Tomatoes

**Baseline with Frozen Backbone:**
```bash
python experiments/experiment_baseline.py
```

**Baseline with Trainable Backbone:**
```bash
python experiments/experiment_baseline.py --train-backbone
```

**Compare Both:**
```bash
python experiments/experiment_baseline.py --compare
```

**Full Fine-tuning with HF Trainer:**
```bash
python experiments/experiment_finetuning.py --method full
```

### Exercise 3: Advanced Topics

#### Exercise 3.1: Efficient Fine-tuning (LoRA)
**Goal**: Use LoRA for parameter-efficient fine-tuning

```bash
# LoRA fine-tuning
python experiments/experiment_finetuning.py --method lora

# Compare full vs LoRA
python experiments/experiment_finetuning.py --method compare
```

Edit `configs/config.py` to customize
