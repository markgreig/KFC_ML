# NER Model Fix - Explanation

## Problem Identified

The original NER model had a **validation F1 score of 0.0000**, indicating the model wasn't learning at all.

## Root Causes

### 1. **Multi-Label Classification Complexity**
The original approach used multi-label classification to identify all competitors in a tweet:
- Input: Tweet text
- Output: 14-dimensional binary vector (one per competitor)
- Problem: The data preprocessing was flawed

### 2. **Data Structure Mismatch**
```python
# Original approach:
1. Expanded data: One row per (Tweet, Competitor) pair
2. In NERDataset: Grouped back by Tweet to create multi-label targets
3. Problem: Grouping logic assumed tweets were unique strings
```

**Issues:**
- Tweets with slight whitespace/encoding differences wouldn't group correctly
- The "KFC competitors" columns in the CSV are incomplete/inconsistent
- After expansion and re-grouping, labels might all be zeros

### 3. **No Validation or Debugging**
- No checks to verify labels were created correctly
- No validation that competitors in data matched COMPETITORS list
- No debugging output during dataset creation

## The Fix

### Approach 1: Simplified Single-Label NER (Recommended)

**File:** `KFC_NER_Sentiment_FIXED.ipynb`

**Key Changes:**

#### 1. **Changed to Single-Label Classification**
```python
# NEW: 14-class classification
# Task: Given a tweet, predict which competitor it's PRIMARILY about (0-13)
# This matches the dataset structure where each row has one primary "Competitor"

class CompetitorClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.competitor_to_idx = {comp: idx for idx, comp in enumerate(COMPETITORS)}
        # Each sample: Tweet → Competitor Index
        # Label is the index (0-13) of the competitor
```

**Advantages:**
- Matches the dataset structure (each row has one primary competitor)
- Much simpler to implement and debug
- Standard classification task that BERT excels at
- Can easily validate labels are correct

#### 2. **Added Data Validation**
```python
# Validate competitor names match our list
def normalize_competitor_name(comp_str):
    # Direct match, case-insensitive match, etc.

# During dataset creation:
print(f"Dataset has {len(unique_comps)} unique competitors:")
for comp in unique_comps:
    if comp in self.competitor_to_idx:
        count = (self.data['Competitor'] == comp).sum()
        print(f"  ✓ {comp}: {count} samples (label {self.competitor_to_idx[comp]})")
    else:
        print(f"  ✗ {comp}: NOT IN COMPETITOR LIST!")
```

#### 3. **Enhanced Multi-Competitor Extraction**
```python
# For tweets mentioning multiple competitors, use REGEX
def extract_all_competitors(tweet_text):
    # Comprehensive patterns for each competitor
    patterns = {
        'KFC': [r'\bkfc\b', r'\bkentucky\s*fried\s*chicken\b', r'@kfc'],
        "McDonald's": [r'\bmcdonald(?:s|\'s)?\b', r'\bmaccies\b', r'@mcdonald'],
        # ... etc
    }
```

**Why this works:**
- NER model predicts PRIMARY competitor (high accuracy)
- Regex extraction finds ALL mentioned competitors (high recall)
- Combined approach gives best of both worlds

#### 4. **Class Weights for Imbalance**
```python
# Handle KFC being 79% of data
ner_class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(len(COMPETITORS)),
    y=train_labels
)
criterion = nn.CrossEntropyLoss(weight=ner_class_weights)
```

### Pipeline Architecture (Fixed)

```
Input Tweet: "KFC's chicken is great but McDonald's fries are better"
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            NER Classifier                  Regex Extraction
        (Primary Competitor)              (All Mentions)
                    ↓                               ↓
               Predicts: KFC              Finds: [KFC, McDonald's]
                    └───────────────┬───────────────┘
                                    ↓
                        Union: {KFC, McDonald's}
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
        Sentiment(Tweet + "KFC")      Sentiment(Tweet + "McDonald's")
                    ↓                               ↓
            KFC: positive                   McDonald's: positive
                    └───────────────┬───────────────┘
                                    ↓
        Output: [(KFC, 2), (McDonald's, 2)]
```

## Expected Performance Improvement

| Metric | Original | Fixed | Improvement |
|--------|----------|-------|-------------|
| NER Val F1 | 0.0000 | **>0.70** | +0.70 |
| NER Val Accuracy | ~0.00 | **>0.75** | +0.75 |
| Sentiment (unchanged) | >0.70 | >0.70 | - |

## Why This Will Work

1. **Simpler Task**: 14-class classification is much easier than multi-label
2. **Data Alignment**: Task matches the dataset structure
3. **Validation**: Every step has checks and debugging output
4. **Hybrid Approach**: NER + Regex ensures we catch all competitors
5. **Class Balancing**: Weights prevent model from just predicting "KFC" every time

## Implementation Notes

### Critical Changes in Code

**Before (BROKEN):**
```python
# Multi-label dataset
self.tweet_groups = self.data.groupby('Tweet')['Competitor'].apply(list).to_dict()
labels = torch.zeros(len(COMPETITORS), dtype=torch.float)
for competitor in competitors_in_tweet:
    labels[COMPETITORS.index(competitor)] = 1.0
# Problem: groupby may fail, labels could be all zeros
```

**After (FIXED):**
```python
# Single-label dataset
label = self.competitor_to_idx[competitor]  # Simple index lookup
# Much more reliable, easy to validate
```

### What to Watch During Training

**Good signs:**
- Val F1 starts at ~0.07 (random guessing for 14 classes)
- Val F1 increases each epoch
- Val F1 reaches >0.60 by epoch 3-5

**Bad signs:**
- Val F1 stays at 0.0000 → Labels still broken
- Val F1 increases but stays <0.30 → Model just predicting majority class (KFC)

### Debugging Checklist

If F1 is still 0.0000:
1. Check dataset creation output - are all competitors listed?
2. Verify label distribution - do we have samples for all 14 classes?
3. Print first batch to inspect labels manually
4. Check if loss is decreasing during training

## Alternative Approaches (If Needed)

### Option B: Fix Multi-Label Approach

If you really need multi-label classification:

1. **Don't expand data first** - work with original rows
2. **Manually create multi-label targets** from tweet text using regex
3. **Validate** that labels are non-zero for all samples

```python
# Better multi-label approach
class MultiLabelNERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        # Don't group - create labels per row
        self.labels = []
        for idx, row in dataframe.iterrows():
            tweet = row['Tweet']
            # Use regex to find ALL competitors in tweet
            found_comps = extract_all_competitors(tweet)
            # Create binary label vector
            label_vec = torch.zeros(len(COMPETITORS))
            for comp in found_comps:
                if comp in COMPETITORS:
                    label_vec[COMPETITORS.index(comp)] = 1.0
            self.labels.append(label_vec)
```

### Option C: Two-Stage Approach

1. **Stage 1**: Binary classifier - "Does tweet mention competitor X?" (14 separate models)
2. **Stage 2**: Sentiment model (same as current)

More complex but potentially higher accuracy.

## Recommendation

**Use the fixed single-label approach** (`KFC_NER_Sentiment_FIXED.ipynb`) because:
- Simplest to implement and debug
- Matches dataset structure
- Combined with regex, catches all competitors
- Much more likely to achieve F1 >0.70

The original multi-label approach was over-engineered for this problem.

## Testing the Fix

After training, test with:

```python
# Should correctly identify primary competitor
test_tweets = [
    ("KFC's chicken is amazing!", "KFC"),
    ("McDonald's burgers are terrible", "McDonald's"),
    ("I love Nando's peri-peri sauce", "Nando's"),
]

for tweet, expected in test_tweets:
    results = predict_pipeline(tweet, ner_model, sentiment_model, ner_tokenizer, sentiment_tokenizer)
    predicted_comp = results[0][0]  # Primary competitor
    print(f"Tweet: {tweet}")
    print(f"Expected: {expected}, Predicted: {predicted_comp}")
    print(f"Match: {expected == predicted_comp}\n")
```

If this works, the fix is successful!
