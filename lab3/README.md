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

## Exercises Overview

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

**Expected Results**: ~88-92% accuracy with full fine-tuning

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
