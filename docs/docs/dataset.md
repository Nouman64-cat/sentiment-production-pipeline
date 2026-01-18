---
sidebar_position: 4
---

# Dataset Building Process

This document explains how the training dataset was curated and structured for the sentiment analysis models.

## Dataset Overview

| Property     | Value                               |
| ------------ | ----------------------------------- |
| **Format**   | CSV with headers                    |
| **Columns**  | `text`, `sentiment`                 |
| **Labels**   | Binary (0 = negative, 1 = positive) |
| **Location** | `dataset/data.csv`                  |

## Data Curation Philosophy

Rather than blindly downloading a pre-existing dataset, this project emphasizes **intentional data curation** to ensure:

1. **Quality over Quantity**: Focused on clean, well-labeled examples
2. **Domain Relevance**: Samples representative of real-world sentiment expressions
3. **Balanced Classes**: Equal representation of positive and negative sentiments

## Data Sources

The dataset was curated from multiple sources to ensure diversity:

### Primary Sources

- **Product Reviews**: E-commerce and app store reviews
- **Social Media**: Twitter/X posts with clear sentiment signals
- **Movie Reviews**: Film and entertainment critiques

### Selection Criteria

Each sample was selected based on:

1. **Clear Sentiment Signal**: Unambiguous positive or negative tone
2. **Reasonable Length**: 10-500 characters (suitable for both models)
3. **Language Quality**: English text with minimal typos
4. **No Sarcasm/Irony**: Avoided edge cases that require context

## Data Schema

```csv
text,sentiment
"This product exceeded my expectations! Highly recommend.",1
"Terrible experience. Would not buy again.",0
"The movie was absolutely fantastic, loved every minute.",1
"Waste of money, completely disappointed.",0
```

### Column Definitions

| Column      | Type    | Description                                |
| ----------- | ------- | ------------------------------------------ |
| `text`      | string  | Raw text input (reviews, comments, etc.)   |
| `sentiment` | integer | Binary label: 0 (negative) or 1 (positive) |

## Data Statistics

The dataset is split for training and evaluation:

| Split      | Percentage | Purpose               |
| ---------- | ---------- | --------------------- |
| Training   | 80%        | Model training        |
| Validation | 10%        | Hyperparameter tuning |
| Test       | 10%        | Final evaluation      |

:::info Random Seed
All splits use `random_state=42` for reproducibility.
:::

## Data Quality Checks

Before training, the dataset undergoes validation:

```python
# Check for missing values
assert df['text'].notna().all()
assert df['sentiment'].notna().all()

# Check label distribution
label_counts = df['sentiment'].value_counts()
print(f"Positive: {label_counts[1]}, Negative: {label_counts[0]}")

# Check for empty strings
assert (df['text'].str.len() > 0).all()
```

## Preprocessing Applied

All text undergoes preprocessing before training:

1. HTML entity unescaping (`&amp;` → `&`)
2. Lowercasing
3. URL removal
4. Twitter handle removal
5. Special character filtering
6. Whitespace normalization

See [Preprocessing Pipeline](./preprocessing) for detailed implementation.

## Data Augmentation (Optional)

For production systems, consider:

- **Back-translation**: Translate to another language and back
- **Synonym replacement**: Replace words with synonyms
- **Random insertion/deletion**: Add noise for robustness

These techniques were not applied in the base dataset but can be added to improve model generalization.

## Ethical Considerations

When curating sentiment data:

1. **Anonymization**: Remove personally identifiable information
2. **Bias Awareness**: Check for demographic or cultural biases
3. **Consent**: Use publicly available data or properly licensed datasets
4. **Edge Cases**: Document limitations (sarcasm, mixed sentiment)

## Adding New Data

To expand the dataset:

```python
import pandas as pd

# Load existing data
df = pd.read_csv('dataset/data.csv')

# Add new samples
new_data = pd.DataFrame({
    'text': ['New positive review text', 'New negative feedback'],
    'sentiment': [1, 0]
})

# Combine and save
df = pd.concat([df, new_data], ignore_index=True)
df.to_csv('dataset/data.csv', index=False)
```

## Versioning

For production environments, implement data versioning:

```bash
# Tag dataset versions
cp dataset/data.csv dataset/data_v1.0.csv

# Update after curation
# dataset/data.csv (latest)
# dataset/data_v1.0.csv (archived)
```

Consider using DVC (Data Version Control) for larger datasets.
