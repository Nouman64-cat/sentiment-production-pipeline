---
sidebar_position: 6
---

# Model Choices

Justification for selecting Logistic Regression and DistilBERT as the two model architectures.

## Overview

This project implements two fundamentally different approaches to sentiment analysis:

| Model                   | Type          | Use Case                                |
| ----------------------- | ------------- | --------------------------------------- |
| **Logistic Regression** | Classical ML  | High-throughput, low-latency production |
| **DistilBERT**          | Deep Learning | High-accuracy, nuanced understanding    |

## Model A: Classical Machine Learning

### Architecture

```
Raw Text → TF-IDF Vectorization → Logistic Regression → Prediction
```

### Why TF-IDF + Logistic Regression?

#### 1. Proven Baseline Performance

Logistic Regression with TF-IDF features has been a gold standard for text classification for decades:

- **Interpretable**: Coefficients directly show word importance
- **Fast Training**: Minutes vs hours for deep learning
- **No GPU Required**: Runs efficiently on any hardware

#### 2. Production Efficiency

| Metric          | Value           |
| --------------- | --------------- |
| Inference Speed | <1ms per sample |
| Model Size      | ~1-5 MB         |
| Memory Usage    | ~50 MB          |
| Cold Start      | ~100ms          |

#### 3. Hyperparameter Tuning

We use GridSearchCV to optimize:

```yaml
# Configured in src/config/ml_config.yaml
tfidf:
  max_features_options: [3000, 5000]
  ngram_range_options:
    - [1, 1]
    - [1, 2]

logistic_regression:
  C_options: [0.1, 1.0, 10.0]
```

- **max_features**: Vocabulary size limit
- **ngram_range**: Unigrams vs bigrams (captures phrases like "not good")
- **C**: Regularization strength (prevents overfitting)

#### Limitations

- **No Semantic Understanding**: "great" and "awesome" are unrelated to the model
- **Word Order Ignored**: Bag-of-words loses sequence information
- **Limited Generalization**: Struggles with unseen vocabulary

---

## Model B: Deep Learning (DistilBERT)

### Architecture

```
Raw Text → WordPiece Tokenization → DistilBERT Encoder → Classification Head → Prediction
```

### Why DistilBERT?

#### 1. Transformer Power, Smaller Footprint

DistilBERT is a distilled version of BERT that retains ~97% of BERT's performance while being:

- **40% smaller** (66M vs 110M parameters)
- **60% faster** inference
- **Suitable for fine-tuning** on consumer hardware

| Model      | Parameters | Size   | Relative Speed |
| ---------- | ---------- | ------ | -------------- |
| BERT-base  | 110M       | 440 MB | 1x             |
| DistilBERT | 66M        | 260 MB | 1.6x           |
| BERT-large | 340M       | 1.3 GB | 0.5x           |

#### 2. Contextual Understanding

Unlike TF-IDF, transformers understand:

- **Word Order**: "not good" ≠ "good not"
- **Context**: "bank" (river vs financial) distinguished by context
- **Semantics**: "excellent" and "amazing" have similar representations

#### 3. Transfer Learning

Pre-trained on massive text corpora, DistilBERT brings:

- General language understanding
- Robust vocabulary handling (WordPiece tokenization)
- Ability to fine-tune on small datasets

### Training Configuration

```yaml
# Configured in src/config/dl_config.yaml
training:
  epochs: 3
  batch_size: 8
  max_len: 128
  learning_rate: 2e-5
```

| Hyperparameter | Value | Rationale                              |
| -------------- | ----- | -------------------------------------- |
| Epochs         | 3     | Prevents overfitting on small datasets |
| Batch Size     | 8     | Fits in consumer GPU memory            |
| Max Length     | 128   | Covers most sentiment texts            |
| Learning Rate  | 2e-5  | Standard for transformer fine-tuning   |

### Training Features

- **AdamW Optimizer**: Weight decay for regularization
- **Linear Learning Rate Schedule**: Warm-up then decay
- **Early Stopping**: Patience of 3 epochs
- **Gradient Clipping**: Max norm of 1.0

#### Limitations

- **Slow Inference**: 10-50x slower than classical ML
- **Large Model Size**: ~260 MB vs ~1 MB
- **GPU Preferred**: CPU inference is sluggish
- **Resource Intensive**: Higher memory and compute costs

---

## Why Two Models?

### Complementary Strengths

```
                    ┌─────────────────────────────────────┐
                    │         Decision Matrix             │
                    └─────────────────────────────────────┘

    High Throughput ◀─────────────────────────▶ High Accuracy
    Low Latency                                 Deep Understanding

    [  Logistic Regression  ]     vs     [  DistilBERT  ]
          Real-time API                   Batch Processing
          Cost-Sensitive                  Quality-Critical
          Simple Sentiment                Nuanced Analysis
```

### Recommended Use Cases

| Scenario                      | Recommended Model   |
| ----------------------------- | ------------------- |
| Real-time API with SLA <100ms | Logistic Regression |
| Batch processing overnight    | DistilBERT          |
| High-volume, cost-sensitive   | Logistic Regression |
| Complex/nuanced sentiment     | DistilBERT          |
| Edge deployment               | Logistic Regression |
| Research/Analysis             | DistilBERT          |

### Production Strategy

For production systems, consider:

1. **Default to ML Model**: Use for 90%+ of requests
2. **Fallback to DL Model**: When confidence is low
3. **A/B Testing**: Compare performance on live traffic
4. **Ensemble**: Combine predictions for critical decisions

---

## Alternative Models Considered

### Why Not BERT-base?

- Too large for production APIs
- Marginal accuracy gain over DistilBERT
- 2x slower inference

### Why Not RoBERTa?

- Larger than DistilBERT
- Better for very large datasets
- Overkill for binary sentiment

### Why Not Naive Bayes?

- Actually considered as alternative to LogReg
- LogReg performs better on TF-IDF features
- LogReg offers L2 regularization

### Why Not LSTM/GRU?

- Replaced by transformers in modern NLP
- Similar accuracy, more complexity
- No pre-training benefits

---

## Summary

| Aspect       | Logistic Regression | DistilBERT        |
| ------------ | ------------------- | ----------------- |
| **Speed**    | ⚡ Sub-millisecond  | 🐢 10-50ms        |
| **Size**     | 📦 ~1 MB            | 📦 ~260 MB        |
| **Accuracy** | ✅ Good             | ✅ Better         |
| **Nuance**   | ❌ Limited          | ✅ Excellent      |
| **Cost**     | 💵 Very Low         | 💵 Higher         |
| **Best For** | Production APIs     | Accuracy-Critical |
