---
sidebar_position: 5
---

# Preprocessing Pipeline

Detailed documentation of the text cleaning and normalization steps applied to all input data.

## Overview

The preprocessing pipeline ensures consistent text formatting across:

- Training data preparation
- API inference requests
- Analysis notebooks

All preprocessing is centralized in `src/preprocessing/clean.py`.

## Implementation

```python
import re
import html

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Step 1: Unescape HTML entities
    text = html.unescape(text)

    # Step 2: Convert to lowercase
    text = text.lower()

    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Step 4: Remove Twitter/social media handles
    text = re.sub(r'@\w+', '', text)

    # Step 5: Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)

    # Step 6: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
```

## Step-by-Step Breakdown

### Step 1: HTML Entity Unescaping

**Purpose**: Convert HTML entities back to their character equivalents.

| Before   | After |
| -------- | ----- |
| `&amp;`  | `&`   |
| `&lt;`   | `<`   |
| `&gt;`   | `>`   |
| `&quot;` | `"`   |

**Why**: Many datasets scraped from the web contain HTML entities that should be normalized.

```python
>>> import html
>>> html.unescape("Tom &amp; Jerry")
'Tom & Jerry'
```

### Step 2: Lowercasing

**Purpose**: Normalize case to reduce vocabulary size.

| Before             | After              |
| ------------------ | ------------------ |
| `AMAZING Product!` | `amazing product!` |
| `This is GREAT`    | `this is great`    |

**Why**: "Amazing" and "amazing" should be treated as the same word by the model.

### Step 3: URL Removal

**Purpose**: Remove HTTP/HTTPS/WWW links that don't contribute to sentiment.

**Pattern**: `r'http\S+|www\S+|https\S+'`

| Before                           | After        |
| -------------------------------- | ------------ |
| `Check this https://example.com` | `Check this` |
| `Visit www.example.com now`      | `Visit now`  |

**Why**: URLs are noise and vary widely without semantic value for sentiment.

### Step 4: Handle Removal

**Purpose**: Remove Twitter/social media mentions.

**Pattern**: `r'@\w+'`

| Before                    | After            |
| ------------------------- | ---------------- |
| `@elonmusk this is great` | `this is great`  |
| `Hey @user check this`    | `Hey check this` |

**Why**: Usernames don't indicate sentiment and create sparsity.

### Step 5: Special Character Filtering

**Purpose**: Remove emojis, symbols, and non-standard characters.

**Pattern**: `r'[^a-zA-Z0-9\s.,!?]'`

**Kept Characters**:

- Letters: `a-z`, `A-Z`
- Numbers: `0-9`
- Whitespace: ` `
- Punctuation: `. , ! ?`

| Before            | After           |
| ----------------- | --------------- |
| `This is 🔥🔥🔥`  | `This is`       |
| `Price: $50!!!`   | `Price 50!!!`   |
| `#blessed #happy` | `blessed happy` |

**Why**: Emojis and special symbols can be noisy and are not handled well by TF-IDF. For deep learning models, consider keeping emojis with emoji-to-text conversion.

### Step 6: Whitespace Normalization

**Purpose**: Collapse multiple spaces and trim edges.

**Pattern**: `r'\s+'` → `' '`

| Before               | After           |
| -------------------- | --------------- |
| `  extra   spaces  ` | `extra spaces`  |
| `newlines\n\nhere`   | `newlines here` |

**Why**: Consistent spacing ensures clean tokenization.

## Usage Examples

### In Training Scripts

```python
from src.preprocessing.clean import clean_text

df['text'] = df['text'].apply(clean_text)
```

### In API Endpoints

```python
from src.preprocessing.clean import clean_text

@app.post("/predict-ml")
def predict_ml(request: PredictionRequest):
    clean_input = clean_text(request.text)
    prediction = model.predict([clean_input])[0]
    return {"prediction": prediction}
```

### In Notebooks

```python
import sys
sys.path.insert(0, '..')  # Add project root
from src.preprocessing.clean import clean_text

test_text = "This is AMAZING!!! @user https://example.com 🎉"
print(clean_text(test_text))
# Output: "this is amazing!!!"
```

## Edge Cases

| Input   | Output | Note                          |
| ------- | ------ | ----------------------------- |
| `None`  | `""`   | Handles None gracefully       |
| `""`    | `""`   | Empty string unchanged        |
| `123`   | `""`   | Non-string returns empty      |
| `"   "` | `""`   | Whitespace-only returns empty |

## Testing

Unit tests are located in `tests/test_preprocessing.py`:

```python
def test_clean_text_basic():
    raw = "  Hello WORLD  "
    expected = "hello world"
    assert clean_text(raw) == expected

def test_clean_text_removes_urls():
    raw = "Check this https://google.com now"
    expected = "check this now"
    assert clean_text(raw) == expected
```

Run tests:

```bash
python -m pytest tests/test_preprocessing.py -v
```

## Considerations for Production

### What's Preserved

- Basic punctuation for sentiment signals (`!`, `?`)
- Numbers for context (ratings, prices)

### What's Lost

- Emojis (significant sentiment signals)
- Hashtags (topic context)
- Case emphasis (AMAZING vs amazing)

### Alternative Approaches

For deep learning models, consider:

```python
# Keep emojis, convert to text
import emoji
text = emoji.demojize(text)  # 🔥 → :fire:

# Or use a model that handles raw text
# (BERT tokenizers handle special characters)
```

## Performance

The `clean_text` function is highly optimized:

- **Regex Compilation**: Patterns are recompiled each call; for production, consider pre-compiling
- **Time Complexity**: O(n) where n is text length
- **Typical Latency**: <1ms for most inputs
