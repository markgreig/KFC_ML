# Sentiment Model Performance Improvements

## Performance Feedback Analysis

**Current Performance:**
- NER Model: **F1 = 0.9019** (90.2%) ✅ **Excellent - Keep as is**
- Sentiment Model: **F1 = 0.5798** (58.0%) ⚠️ **Needs Improvement**

**Target:** Improve Sentiment F1 from 58.0% to **>70%**

---

## Root Cause Analysis

The sentiment model's moderate performance (58%) is likely due to:

1. **Small Training Set** - Only 212 training samples (265 total after split)
2. **Class Imbalance** - One sentiment class may dominate
3. **Simple Contextualization** - Basic prompt "This tweet is about {competitor}."
4. **Standard Cross-Entropy Loss** - Doesn't handle imbalance well
5. **Limited Training** - Only 5 epochs may not be enough
6. **Learning Rate** - 2e-5 might be too high for fine-grained sentiment

---

## Recommended Improvements

### 1. Data Augmentation (High Impact) 🎯

**Problem:** Only 212 training samples is very small for deep learning

**Solution:** Augment training data to 600-800 samples using:

```python
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Synonym Replacement
aug_synonym = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)

# Back Translation (slower but effective)
aug_back_translation = nas.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en',
    device='cuda'
)

# Contextual Word Embeddings
aug_contextual = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute",
    aug_p=0.3
)

def augment_text(text, num_aug=2):
    """Generate augmented versions of text"""
    augmented = []

    # Original
    augmented.append(text)

    # Synonym replacement
    try:
        augmented.append(aug_synonym.augment(text))
    except:
        pass

    # Contextual substitution
    try:
        augmented.append(aug_contextual.augment(text))
    except:
        pass

    # Back translation (slower, use sparingly)
    if num_aug > 2:
        try:
            augmented.append(aug_back_translation.augment(text))
        except:
            pass

    return augmented[:num_aug+1]

# Apply to sentiment training data
def augment_sentiment_dataset(df, aug_factor=3):
    """
    Augment sentiment training dataset

    Args:
        df: Original dataframe
        aug_factor: How many augmented versions per sample

    Returns:
        Augmented dataframe
    """
    augmented_rows = []

    print(f"Augmenting {len(df)} samples with factor {aug_factor}...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Add original
        augmented_rows.append(row.to_dict())

        # Add augmented versions
        text = row['Tweet']
        aug_texts = augment_text(text, num_aug=aug_factor-1)

        for aug_text in aug_texts[1:]:  # Skip first (original)
            aug_row = row.to_dict()
            aug_row['Tweet'] = aug_text
            augmented_rows.append(aug_row)

    augmented_df = pd.DataFrame(augmented_rows)
    print(f"  Original: {len(df)} → Augmented: {len(augmented_df)} samples")

    return augmented_df
```

**Implementation:**
- Add this after sentiment train/val split
- Apply ONLY to `sentiment_train_df`, NOT validation
- Expect ~640 training samples (212 * 3)

**Expected Impact:** +5-10% F1

---

### 2. Focal Loss for Class Imbalance (High Impact) 🎯

**Problem:** Standard CrossEntropyLoss treats all samples equally, even if one class dominates

**Solution:** Implement Focal Loss that focuses on hard-to-classify examples

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Class weights (optional)
        gamma: Focusing parameter (default: 2.0)
            - gamma=0: equivalent to CrossEntropyLoss
            - gamma>0: down-weights easy examples
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

**Implementation:**
- Replace `nn.CrossEntropyLoss` with `FocalLoss` in sentiment training
- Use gamma=2.0 (standard value)
- Combine with class weights for best results

**Expected Impact:** +3-5% F1

---

### 3. Optimized Hyperparameters (Medium Impact)

**Changes:**

| Parameter | Original | Improved | Reason |
|-----------|----------|----------|--------|
| Learning Rate | 2e-5 | **1e-5** | Finer adjustments for pre-trained model |
| Epochs | 5 | **10** | More training time with small dataset |
| Warmup Ratio | 0.1 | **0.2** | More gradual warmup |
| Scheduler | Linear | **Cosine** | Better convergence |

**Implementation:**

```python
# Sentiment model hyperparameters
SENTIMENT_LEARNING_RATE = 1e-5  # Lower
SENTIMENT_EPOCHS = 10  # More
SENTIMENT_WARMUP_RATIO = 0.2  # Higher

# Use cosine scheduler instead of linear
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**Expected Impact:** +2-4% F1

---

### 4. Early Stopping (Medium Impact)

**Problem:** May overfit or underfit without monitoring

**Solution:** Stop training when validation F1 plateaus

```python
class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False

        if self.mode == 'max':
            if val_score > self.best_score + self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if val_score < self.best_score - self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

# Usage in training loop
early_stopping = EarlyStopping(patience=3, mode='max')

for epoch in range(epochs):
    # ... training code ...

    if early_stopping(val_f1):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
```

**Expected Impact:** +1-2% F1 (prevents overfitting)

---

### 5. Better Contextualization (Low-Medium Impact)

**Problem:** Current prompt is too simple: "This tweet is about {competitor}."

**Solution:** Richer context prompt

```python
class ImprovedSentimentDataset(Dataset):
    """Enhanced sentiment dataset with better contextualization"""

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tweet = row['Tweet']
        competitor = row['Competitor']
        sentiment = row['SENTIMENT']

        # IMPROVED: More natural contextualization
        text = f"Tweet: {tweet} | Sentiment about {competitor}:"

        # Alternative formats to try:
        # text = f"{tweet} [SEP] What is the sentiment about {competitor}?"
        # text = f"Analyze sentiment for {competitor}: {tweet}"

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }
```

**Expected Impact:** +1-3% F1

---

### 6. Detailed Analysis & Error Analysis (Important for Debugging)

**Add after training:**

```python
def analyze_sentiment_performance(model, val_loader, dataset_obj):
    """Detailed performance analysis"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Classification report
    print("\\n" + "="*70)
    print("SENTIMENT MODEL DETAILED ANALYSIS")
    print("="*70)
    print("\\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=['negative', 'neutral', 'positive'],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Sentiment Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/sentiment_confusion_matrix.png', dpi=300)
    plt.show()

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    print("\\nPer-Class Metrics:")
    for i, label in enumerate(['negative', 'neutral', 'positive']):
        print(f"  {label:8s}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

    # Identify misclassified examples
    all_probs = np.array(all_probs)
    misclassified = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]

    print(f"\\nMisclassified: {len(misclassified)} / {len(all_labels)} ({len(misclassified)/len(all_labels)*100:.1f}%)")

    # Show most confident wrong predictions
    if len(misclassified) > 0:
        misclass_confidence = [np.max(all_probs[i]) for i in misclassified]
        top_mistakes = sorted(zip(misclassified, misclass_confidence), key=lambda x: x[1], reverse=True)[:5]

        print("\\nTop 5 Most Confident Mistakes:")
        for idx, conf in top_mistakes:
            true_label = SENTIMENT_MAP[all_labels[idx]]
            pred_label = SENTIMENT_MAP[all_preds[idx]]
            print(f"  True: {true_label:8s} | Pred: {pred_label:8s} | Confidence: {conf:.3f}")

    return all_preds, all_labels, all_probs
```

---

## Implementation Priority

**HIGH PRIORITY (Do First):**
1. ✅ **Data Augmentation** - Biggest impact for small dataset
2. ✅ **Focal Loss** - Better handle imbalance
3. ✅ **Optimized Hyperparameters** - Lower LR, more epochs

**MEDIUM PRIORITY (Do Second):**
4. ✅ **Early Stopping** - Prevent overfitting
5. ✅ **Better Contextualization** - Improved prompts

**LOW PRIORITY (Nice to Have):**
6. ✅ **Detailed Analysis** - Understanding performance

---

## Expected Performance Improvement

**Conservative Estimate:**
- Current: 58.0% F1
- With all improvements: **68-75% F1** (+10-17 points)

**Breakdown:**
- Data Augmentation: +5-10%
- Focal Loss: +3-5%
- Hyperparameters: +2-4%
- Early Stopping: +1-2%
- Better Context: +1-3%

---

## Quick Implementation Checklist

- [ ] Install nlpaug: `!pip install -q nlpaug`
- [ ] Add FocalLoss class before training
- [ ] Add augmentation function after data loading
- [ ] Apply augmentation to sentiment_train_df only
- [ ] Update sentiment hyperparameters (LR, epochs, warmup)
- [ ] Replace CrossEntropyLoss with FocalLoss for sentiment
- [ ] Add EarlyStopping class and use in training
- [ ] Update SentimentDataset with better contextualization
- [ ] Add detailed analysis function after training
- [ ] Run analysis and check confusion matrix

---

## Code Integration Points

### Where to Add in Current Notebook:

1. **After Cell 2 (Setup):** Add nlpaug installation
2. **After Cell 7 (Config):** Update sentiment hyperparameters
3. **Before Cell 22 (train_model):** Add FocalLoss and EarlyStopping classes
4. **After Cell 17 (train/val split):** Add data augmentation for sentiment_train_df
5. **Cell 26 (SentimentDataset):** Update contextualization
6. **Cell 29 (Train Sentiment):** Use FocalLoss and early stopping
7. **After Cell 30 (Sentiment plots):** Add detailed analysis

---

## Alternative: Try Different Sentiment Models

If improvements plateau, try these alternative pre-trained models:

```python
# Current
SENTIMENT_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

# Alternatives (may perform better):
# 1. Larger model
SENTIMENT_MODEL_NAME = 'cardiffnlp/twitter-roberta-large-sentiment-latest'

# 2. Different architecture
SENTIMENT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'

# 3. More recent sentiment model
SENTIMENT_MODEL_NAME = 'finiteautomata/bertweet-base-sentiment-analysis'
```

Test each and compare F1 scores.

---

## Monitoring During Training

Watch for these signs:

**Good Signs:**
- ✅ Val F1 steadily increasing
- ✅ Train/Val loss gap is small (<0.2)
- ✅ All classes have >0.5 F1

**Bad Signs:**
- ❌ Val F1 plateaus early (epoch 2-3)
- ❌ Large train/val gap (>0.5) = overfitting
- ❌ One class has very low F1 (<0.3) = imbalance issue

---

## Summary

The sentiment model's current 58% F1 is due to the small training dataset (212 samples). The most impactful improvements are:

1. **Data Augmentation** (3x increase to ~640 samples)
2. **Focal Loss** (better handle class imbalance)
3. **Lower Learning Rate + More Epochs** (1e-5, 10 epochs)

With these changes, expect sentiment F1 to reach **68-75%**, a significant improvement that will make the overall pipeline much more reliable.
