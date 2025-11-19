# Quick Integration Guide - Add These Cells to Your Notebook

I've started creating the improved notebook but let me give you the **exact cells to add/modify** so you can see everything clearly. The improved notebook is at: `KFC_Complete_NER_Sentiment_v2_IMPROVED.ipynb`

## Cells Already Updated ✅

1. **Cell 0 (Header)** - Updated with improvement details
2. **Cell 2 (Installation)** - Added nlpaug
3. **Cell 3 (Imports)** - Added augmentation libraries
4. **Cell 7 (Config)** - Updated sentiment hyperparameters

## Cells You Need to Add (Copy-Paste Ready)

### INSERT AFTER CELL 17 (after train/val split):

#### New Markdown Cell:
```markdown
## 6b. Sentiment Data Augmentation (IMPROVED)
```

#### New Code Cell - Data Augmentation:
```python
# ============================================================
# DATA AUGMENTATION FOR SENTIMENT TRAINING (IMPROVED)
# ============================================================

print("\\n" + "="*70)
print("AUGMENTING SENTIMENT TRAINING DATA")
print("="*70)
print(f"Original training samples: {len(sentiment_train_df)}")
print(f"Target: {len(sentiment_train_df) * AUGMENTATION_FACTOR} samples ({AUGMENTATION_FACTOR}x)")

# Create augmenters
try:
    aug_synonym = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)
    print("✓ Synonym augmenter loaded")
except Exception as e:
    print(f"⚠ Synonym augmenter failed: {e}")
    aug_synonym = None

try:
    aug_contextual = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased',
        action="substitute",
        aug_p=0.3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("✓ Contextual augmenter loaded")
except Exception as e:
    print(f"⚠ Contextual augmenter failed: {e}")
    aug_contextual = None

# Augment data
augmented_rows = []

for idx, row in tqdm(sentiment_train_df.iterrows(), total=len(sentiment_train_df), desc="Augmenting"):
    # Add original
    augmented_rows.append(row.to_dict())

    tweet = row['Tweet']

    # Try synonym replacement
    if aug_synonym is not None and len(augmented_rows) < (idx + 1) * AUGMENTATION_FACTOR:
        try:
            aug_text = aug_synonym.augment(tweet)
            if isinstance(aug_text, list):
                aug_text = aug_text[0]
            if aug_text != tweet:
                aug_row = row.to_dict()
                aug_row['Tweet'] = aug_text
                augmented_rows.append(aug_row)
        except:
            pass

    # Try contextual substitution
    if aug_contextual is not None and len(augmented_rows) < (idx + 1) * AUGMENTATION_FACTOR:
        try:
            aug_text = aug_contextual.augment(tweet)
            if isinstance(aug_text, list):
                aug_text = aug_text[0]
            if aug_text != tweet:
                aug_row = row.to_dict()
                aug_row['Tweet'] = aug_text
                augmented_rows.append(aug_row)
        except:
            pass

sentiment_train_df_augmented = pd.DataFrame(augmented_rows)

print(f"\\n✓ Augmentation complete!")
print(f"  Original: {len(sentiment_train_df)} samples")
print(f"  Augmented: {len(sentiment_train_df_augmented)} samples")
print(f"  Increase: {len(sentiment_train_df_augmented) / len(sentiment_train_df):.1f}x")

print(f"\\nAugmented Sentiment Distribution:")
for sent, count in sentiment_train_df_augmented['SENTIMENT'].value_counts().sort_index().items():
    print(f"  {SENTIMENT_MAP[sent]:8s}: {count} ({count/len(sentiment_train_df_augmented)*100:.1f}%)")

print("="*70)
```

### INSERT BEFORE NER TRAINING (before cell-22):

#### New Markdown Cell:
```markdown
## 6c. Focal Loss & Early Stopping Classes (IMPROVED)
```

#### New Code Cell - Classes:
```python
# ============================================================
# FOCAL LOSS FOR HANDLING CLASS IMBALANCE (IMPROVED)
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Focuses training on hard-to-classify examples.
    Down-weights easy examples to prevent them from dominating training.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================
# EARLY STOPPING TO PREVENT OVERFITTING (IMPROVED)
# ============================================================

class EarlyStopping:
    """Stop training when validation F1 stops improving"""
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
            improved = val_score > self.best_score + self.min_delta
        else:
            improved = val_score < self.best_score - self.min_delta

        if improved:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

print("✓ FocalLoss class defined")
print("✓ EarlyStopping class defined")
```

### MODIFY CELL-22 (NER training function):

Keep the existing `train_model` function but ADD this new improved version for sentiment right after it:

```python
def train_sentiment_model_improved(model, train_loader, val_loader, epochs, learning_rate,
                                   class_weights, early_stopping_patience=3, model_name="sentiment_model"):
    """
    IMPROVED training function for sentiment model with:
    - Focal Loss
    - Early Stopping
    - Cosine scheduler
    - Better progress tracking
    """
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * epochs // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * SENTIMENT_WARMUP_RATIO)

    # IMPROVED: Use cosine scheduler for sentiment
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # IMPROVED: Use Focal Loss instead of CrossEntropyLoss
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_LOSS_GAMMA)

    # IMPROVED: Add early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    best_val_f1 = 0

    print(f"\\nTraining {model_name} with IMPROVEMENTS...")
    print(f"  Epochs: {epochs}, Steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Using: Focal Loss (gamma={FOCAL_LOSS_GAMMA}), Cosine Scheduler, Early Stopping (patience={early_stopping_patience})")
    print(f"  Training samples: {len(train_loader.dataset)}\\n")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.logits, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * GRAD_ACCUM_STEPS
            train_pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUM_STEPS:.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)

        print(f"\\nResults:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val F1 (macro): {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"  ✓ New best F1! Saving model...")
            torch.save(model.state_dict(), f'{MODEL_SAVE_DIR}/{model_name}_best.pt')

        # IMPROVED: Check early stopping
        if early_stopping(val_f1):
            print(f"  ⚠ Early stopping triggered (no improvement for {early_stopping_patience} epochs)")
            break

        print()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\\n✓ Training complete! Best val F1: {best_val_f1:.4f}")
    print(f"  Stopped at epoch {epoch + 1}/{epochs}")

    return model, history

print("✓ Improved sentiment training function defined")
```

### MODIFY CELL-26 (SentimentDataset):

Replace the existing contextualization with:

```python
class SentimentDataset(Dataset):
    """IMPROVED: Competitor-aware sentiment dataset with better contextualization"""

    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tweet = row['Tweet']
        competitor = row['Competitor']
        sentiment = row['SENTIMENT']

        # IMPROVED: Better contextualization
        text = f"Tweet: {tweet} | Sentiment about {competitor}:"

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

print("✓ IMPROVED SentimentDataset defined")
```

### MODIFY CELL-27 (Create Sentiment Datasets):

Change to use AUGMENTED data:

```python
# Load sentiment tokenizer
print(f"Loading Sentiment tokenizer: {SENTIMENT_MODEL_NAME}")
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

# Create datasets - IMPROVED: Use AUGMENTED training data
print("\\nCreating IMPROVED Sentiment datasets...")
sentiment_train_dataset = SentimentDataset(sentiment_train_df_augmented, sentiment_tokenizer, MAX_SEQ_LENGTH)  # AUGMENTED!
sentiment_val_dataset = SentimentDataset(sentiment_val_df, sentiment_tokenizer, MAX_SEQ_LENGTH)  # Keep validation original
sentiment_test_dataset = SentimentDataset(df_test_clean, sentiment_tokenizer, MAX_SEQ_LENGTH)

print(f"  Train (AUGMENTED): {len(sentiment_train_dataset)} samples")
print(f"  Val: {len(sentiment_val_dataset)} samples")
print(f"  Test: {len(sentiment_test_dataset)} samples")

# DataLoaders
sentiment_train_loader = DataLoader(sentiment_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
sentiment_val_loader = DataLoader(sentiment_val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)
sentiment_test_loader = DataLoader(sentiment_test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

print(f"\\n✓ IMPROVED Sentiment DataLoaders created with augmented training data")
```

### MODIFY CELL-28 (Sentiment Class Weights):

Update to use AUGMENTED data:

```python
# Sentiment class weights - IMPROVED: Use augmented training data
sentiment_class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(3),
    y=sentiment_train_df_augmented['SENTIMENT'].values  # AUGMENTED!
)
sentiment_class_weights = torch.tensor(sentiment_class_weights, dtype=torch.float).to(device)

print("Sentiment Class Weights (from augmented data):")
for i, label in SENTIMENT_MAP.items():
    print(f"  {label:8s}: {sentiment_class_weights[i]:.3f}")
```

### MODIFY CELL-29 (Train Sentiment):

Use the improved training function:

```python
# Train Sentiment model with IMPROVEMENTS
print(f"Initializing Sentiment model: {SENTIMENT_MODEL_NAME}")
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL_NAME,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# IMPROVED: Use new training function with all improvements
sentiment_model, sentiment_history = train_sentiment_model_improved(
    sentiment_model,
    sentiment_train_loader,
    sentiment_val_loader,
    epochs=SENTIMENT_EPOCHS,
    learning_rate=SENTIMENT_LEARNING_RATE,
    class_weights=sentiment_class_weights,
    early_stopping_patience=SENTIMENT_EARLY_STOP_PATIENCE,
    model_name="sentiment_model_improved"
)
```

### INSERT AFTER CELL-30 (after sentiment plots):

#### New Markdown Cell:
```markdown
## 8b. Detailed Sentiment Analysis (IMPROVED)
```

#### New Code Cell - Detailed Analysis:
```python
# ============================================================
# DETAILED SENTIMENT PERFORMANCE ANALYSIS (IMPROVED)
# ============================================================

print("\\n" + "="*70)
print("SENTIMENT MODEL DETAILED ANALYSIS")
print("="*70)

sentiment_model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(sentiment_val_loader, desc="Analyzing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = sentiment_model(input_ids, attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Classification report
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
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['negative', 'neutral', 'positive'],
    yticklabels=['negative', 'neutral', 'positive']
)
plt.title('Sentiment Confusion Matrix - IMPROVED MODEL', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/sentiment_confusion_matrix_improved.png', dpi=300)
plt.show()

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, average=None
)

print("\\nPer-Class Metrics:")
print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
print("-" * 50)
for i, label in enumerate(['negative', 'neutral', 'positive']):
    print(f"{label:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")

# Overall metrics
overall_f1 = f1.mean()
print(f"\\nOverall F1 (Macro Avg): {overall_f1:.4f}")
print(f"Improvement from baseline: {overall_f1 - 0.58:.4f} ({(overall_f1 - 0.58)/0.58*100:.1f}% increase)")

# Confidence analysis
avg_confidence = np.max(all_probs, axis=1).mean()
print(f"\\nAverage Prediction Confidence: {avg_confidence:.3f}")

# Misclassifications
misclassified = all_preds != all_labels
num_misclass = misclassified.sum()
print(f"\\nMisclassifications: {num_misclass} / {len(all_labels)} ({num_misclass/len(all_labels)*100:.1f}%)")

print("\\n" + "="*70)
```

---

## Summary of Changes

**BEFORE (Original Performance):**
- Sentiment F1: 58.0%
- Training samples: 212
- Loss: CrossEntropyLoss
- LR: 2e-5, Epochs: 5
- No augmentation
- No early stopping

**AFTER (Expected Performance with All Improvements):**
- Sentiment F1: **68-75%** 🎯
- Training samples: **~640 (3x augmented)**
- Loss: **Focal Loss (gamma=2.0)**
- LR: **1e-5**, Epochs: **10**, Warmup: **0.2**
- **Data augmentation** (synonym + contextual)
- **Early stopping** (patience=3)
- **Better contextualization**
- **Detailed analysis** with confusion matrix

---

## Ready to Use!

The notebook `KFC_Complete_NER_Sentiment_v2_IMPROVED.ipynb` has the header, installation, imports, and config already updated. Just add the cells above and you're ready to run!

**Expected time to integrate:** 10-15 minutes of copy-paste
**Expected training time:** ~30-45 minutes (sentiment training will take longer with more epochs and augmented data)
**Expected improvement:** +10-17 percentage points in sentiment F1
