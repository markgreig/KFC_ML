# CRITICAL BUG FIX - Sentiment Training Data

## Problem Identified

**Error:** `KeyError: 'SENTIMENT'`

**Root Cause:**
The notebook incorrectly uses `df_large_clean` (from "KFC_social_data.xlsx - Sheet1.csv") for BOTH NER and Sentiment training. However:

- ✅ `df_large` **HAS** `Competitor` labels → Good for NER training
- ❌ `df_large` **DOES NOT HAVE** clean `SENTIMENT` labels → Cannot train sentiment model

**Correct approach:**
- Use `df_large_clean` for **NER training** (3,183 rows with Competitor labels)
- Use `df_train_sample_clean` for **Sentiment training** (265 rows with SENTIMENT labels)

## Fix Implementation

### Step 1: Separate Data Preparation

**Add this cell AFTER data loading (around cell 10):**

```python
# IMPORTANT: Separate datasets for NER and Sentiment
# NER: Use large dataset (has Competitor labels)
# Sentiment: Use training sample (has SENTIMENT labels)

print("="*70)
print("PREPARING DATASETS FOR DIFFERENT TASKS")
print("="*70)

# NER Dataset (use large dataset)
df_large_clean = prepare_dataset(df_large, "Large dataset for NER")

# Sentiment Dataset (use training sample)
df_train_sample_clean = prepare_dataset(df_train_sample, "Training sample for Sentiment")

print("\n" + "="*70)
print("DATASET SUMMARY:")
print(f"  NER training data: {len(df_large_clean)} rows (from large CSV)")
print(f"  Sentiment training data: {len(df_train_sample_clean)} rows (from training sample)")
print("="*70)
```

### Step 2: Create Separate Train/Val Splits

**Replace the existing train/val split cell with:**

```python
# ============================================================
# TRAIN/VAL SPLIT - SEPARATE FOR NER AND SENTIMENT
# ============================================================

# NER: Split large dataset (for competitor identification)
print("\nCreating NER train/val split...")
ner_train_df, ner_val_df = train_test_split(
    df_large_clean,
    test_size=0.2,
    random_state=SEED,
    stratify=df_large_clean['Competitor']
)

print(f"NER Dataset Split:")
print(f"  Training: {len(ner_train_df)} samples")
print(f"  Validation: {len(ner_val_df)} samples")
print(f"  Competitors: {ner_train_df['Competitor'].nunique()}")

# Sentiment: Split training sample (for sentiment classification)
print("\nCreating Sentiment train/val split...")
sentiment_train_df, sentiment_val_df = train_test_split(
    df_train_sample_clean,
    test_size=0.2,
    random_state=SEED,
    stratify=df_train_sample_clean['SENTIMENT']
)

print(f"\nSentiment Dataset Split:")
print(f"  Training: {len(sentiment_train_df)} samples")
print(f"  Validation: {len(sentiment_val_df)} samples")
print(f"  Sentiment distribution:")
sentiment_dist = sentiment_train_df['SENTIMENT'].value_counts().sort_index()
for sent, count in sentiment_dist.items():
    print(f"    {SENTIMENT_MAP[sent]:8s}: {count} ({count/len(sentiment_train_df)*100:.1f}%)")
```

### Step 3: Update NER Dataset Creation

**Find the NER dataset creation cell and change variable names:**

```python
# NER datasets - use NER-specific splits
print(f"Loading NER tokenizer: {NER_MODEL_NAME}")
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)

print("\nCreating NER datasets...")
ner_train_dataset = CompetitorDataset(ner_train_df, ner_tokenizer, MAX_SEQ_LENGTH)  # Changed from train_df
ner_val_dataset = CompetitorDataset(ner_val_df, ner_tokenizer, MAX_SEQ_LENGTH)      # Changed from val_df
ner_test_dataset = CompetitorDataset(df_test_clean, ner_tokenizer, MAX_SEQ_LENGTH)

# DataLoaders
ner_train_loader = DataLoader(ner_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
ner_val_loader = DataLoader(ner_val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)
ner_test_loader = DataLoader(ner_test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

print(f"\n✓ NER DataLoaders created")
```

### Step 4: Update NER Class Weights

**Find the NER class weights cell:**

```python
# Calculate class weights for NER - use ner_train_df
train_labels = [ner_train_dataset.competitor_to_idx[comp] for comp in ner_train_df['Competitor']]  # Changed from train_df
ner_class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(len(COMPETITORS)),
    y=train_labels
)
ner_class_weights = torch.tensor(ner_class_weights, dtype=torch.float).to(device)

print("NER Class Weights:")
for i, comp in enumerate(COMPETITORS):
    print(f"  {comp:20s}: {ner_class_weights[i]:.3f}")
```

### Step 5: Update Sentiment Dataset Creation

**Find the sentiment dataset creation cell:**

```python
# Sentiment datasets - use sentiment-specific splits
print(f"Loading Sentiment tokenizer: {SENTIMENT_MODEL_NAME}")
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

print("\nCreating Sentiment datasets...")
sentiment_train_dataset = SentimentDataset(sentiment_train_df, sentiment_tokenizer, MAX_SEQ_LENGTH)  # Changed from train_df
sentiment_val_dataset = SentimentDataset(sentiment_val_df, sentiment_tokenizer, MAX_SEQ_LENGTH)      # Changed from val_df
sentiment_test_dataset = SentimentDataset(df_test_clean, sentiment_tokenizer, MAX_SEQ_LENGTH)

print(f"  Train: {len(sentiment_train_dataset)} samples")
print(f"  Val: {len(sentiment_val_dataset)} samples")
print(f"  Test: {len(sentiment_test_dataset)} samples")

# DataLoaders
sentiment_train_loader = DataLoader(sentiment_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
sentiment_val_loader = DataLoader(sentiment_val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)
sentiment_test_loader = DataLoader(sentiment_test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

print(f"\n✓ Sentiment DataLoaders created")
```

### Step 6: Update Sentiment Class Weights

**Find the sentiment class weights cell:**

```python
# Sentiment class weights - use sentiment_train_df
sentiment_class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(3),
    y=sentiment_train_df['SENTIMENT'].values  # Changed from train_df
)
sentiment_class_weights = torch.tensor(sentiment_class_weights, dtype=torch.float).to(device)

print("Sentiment Class Weights:")
for i, label in SENTIMENT_MAP.items():
    print(f"  {label:8s}: {sentiment_class_weights[i]:.3f}")
```

## Summary of Changes

| Component | OLD (Broken) | NEW (Fixed) |
|-----------|--------------|-------------|
| **NER Training Data** | `train_df` (from df_large) | `ner_train_df` (from df_large) ✅ |
| **NER Validation Data** | `val_df` (from df_large) | `ner_val_df` (from df_large) ✅ |
| **Sentiment Training Data** | `train_df` (❌ no SENTIMENT) | `sentiment_train_df` (from df_train_sample) ✅ |
| **Sentiment Validation Data** | `val_df` (❌ no SENTIMENT) | `sentiment_val_df` (from df_train_sample) ✅ |

## Why This Works

**NER Model:**
- Needs: `Competitor` labels
- Uses: `df_large_clean` (3,183 rows)
- Has: All competitor labels
- Result: ✅ Can train successfully

**Sentiment Model:**
- Needs: `SENTIMENT` labels
- Uses: `df_train_sample_clean` (265 rows)
- Has: Clean sentiment labels (0, 1, 2)
- Result: ✅ Can train successfully

## Trade-off Note

**Sentiment model has less training data (265 vs 3,183 rows):**
- This is unavoidable - only the training sample has sentiment labels
- Can mitigate by:
  - Using pre-trained Twitter sentiment model (already doing this ✅)
  - Data augmentation (synonym replacement, back-translation)
  - More epochs (5-7 instead of 5)
  - Lower learning rate for more careful training

## Expected Performance Impact

| Model | Training Data | Expected F1 |
|-------|---------------|-------------|
| **NER** | 3,183 rows | **>0.75** (lots of data) |
| **Sentiment** | 265 rows | **>0.65** (less data, but pre-trained model helps) |

**Note:** Using `twitter-roberta-base-sentiment-latest` helps significantly since it's already pre-trained on tweet sentiment!

## Quick Verification

After applying fixes, verify in your notebook:

```python
# Check that data is correctly assigned
print("NER training data columns:", ner_train_df.columns.tolist())
print("Sentiment training data columns:", sentiment_train_df.columns.tolist())

# Verify SENTIMENT column exists
assert 'SENTIMENT' in sentiment_train_df.columns, "❌ SENTIMENT column missing!"
print("✅ Sentiment data has SENTIMENT column")

# Check data sizes
print(f"\nData sizes:")
print(f"  NER train: {len(ner_train_df)} rows")
print(f"  Sentiment train: {len(sentiment_train_df)} rows")
```

Expected output:
```
NER training data columns: ['Competitor', 'Tweet', 'SENTIMENT', ...]
Sentiment training data columns: ['Competitor', 'Tweet', 'SENTIMENT', ...]
✅ Sentiment data has SENTIMENT column
Data sizes:
  NER train: 2546 rows
  Sentiment train: 212 rows
```

## Alternative: Label More Data

If sentiment model performance is poor due to limited data (265 rows), consider:

1. **Use both datasets** - Label the large dataset with sentiment
2. **Semi-supervised learning** - Use NER model to filter, then manually label
3. **Weak supervision** - Use lexicon-based labels for large dataset (not ideal but better than nothing)

For now, the fix above allows the notebook to run successfully!
