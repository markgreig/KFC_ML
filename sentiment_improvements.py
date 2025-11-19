"""
Sentiment Model Improvement Functions

This module contains all the functions and classes needed to improve
the sentiment model from F1=58% to F1>70%.

Usage:
    1. Import this module in your notebook
    2. Use the functions to augment data, train with focal loss, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# 1. FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Class weights (tensor or None)
        gamma: Focusing parameter (default: 2.0)
            - gamma=0: equivalent to CrossEntropyLoss
            - gamma>0: down-weights easy examples
        reduction: 'mean', 'sum', or 'none'

    Example:
        >>> criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        >>> loss = criterion(outputs.logits, labels)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits from model
            targets: (batch_size,) - ground truth class labels

        Returns:
            Focal loss value
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none'
        )

        # Probability of correct class
        p_t = torch.exp(-ce_loss)

        # Focal loss: (1 - p_t)^gamma * CE
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# 2. EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting

    Monitors a metric (e.g., validation F1) and stops training when it
    stops improving for 'patience' epochs.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'max' (for F1, accuracy) or 'min' (for loss)

    Example:
        >>> early_stopping = EarlyStopping(patience=3, mode='max')
        >>> for epoch in range(epochs):
        >>>     val_f1 = train_one_epoch()
        >>>     if early_stopping(val_f1):
        >>>         print("Early stopping triggered")
        >>>         break
    """
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        """
        Args:
            val_score: Current validation metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            return False

        # Check if improved
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

    def reset(self):
        """Reset the early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# ============================================================================
# 3. DATA AUGMENTATION
# ============================================================================

def create_augmenters(device='cuda'):
    """
    Create text augmentation models

    Args:
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary of augmenters
    """
    augmenters = {}

    try:
        # Synonym replacement (fast)
        augmenters['synonym'] = naw.SynonymAug(
            aug_src='wordnet',
            aug_p=0.3  # Replace 30% of words
        )
        print("✓ Synonym augmenter loaded")
    except Exception as e:
        print(f"⚠ Synonym augmenter failed: {e}")

    try:
        # Contextual word embeddings (medium speed)
        augmenters['contextual'] = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute",
            aug_p=0.3,
            device=device
        )
        print("✓ Contextual augmenter loaded")
    except Exception as e:
        print(f"⚠ Contextual augmenter failed: {e}")

    try:
        # Back translation (slow but effective)
        augmenters['back_translation'] = nas.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en',
            device=device
        )
        print("✓ Back-translation augmenter loaded")
    except Exception as e:
        print(f"⚠ Back-translation augmenter failed: {e}")

    return augmenters


def augment_text(text, augmenters, num_aug=2):
    """
    Generate augmented versions of text

    Args:
        text: Original text to augment
        augmenters: Dictionary of augmentation models
        num_aug: Number of augmented versions to generate

    Returns:
        List of augmented texts (including original)
    """
    augmented = [text]  # Always include original

    # Try synonym replacement
    if 'synonym' in augmenters and len(augmented) < num_aug + 1:
        try:
            aug_text = augmenters['synonym'].augment(text)
            if isinstance(aug_text, list):
                aug_text = aug_text[0]
            if aug_text != text and aug_text not in augmented:
                augmented.append(aug_text)
        except:
            pass

    # Try contextual substitution
    if 'contextual' in augmenters and len(augmented) < num_aug + 1:
        try:
            aug_text = augmenters['contextual'].augment(text)
            if isinstance(aug_text, list):
                aug_text = aug_text[0]
            if aug_text != text and aug_text not in augmented:
                augmented.append(aug_text)
        except:
            pass

    # Try back-translation (use sparingly, it's slow)
    if 'back_translation' in augmenters and len(augmented) < num_aug + 1:
        try:
            aug_text = augmenters['back_translation'].augment(text)
            if isinstance(aug_text, list):
                aug_text = aug_text[0]
            if aug_text != text and aug_text not in augmented:
                augmented.append(aug_text)
        except:
            pass

    return augmented


def augment_sentiment_dataset(df, augmenters, aug_factor=3):
    """
    Augment sentiment training dataset

    Args:
        df: Original dataframe with 'Tweet', 'Competitor', 'SENTIMENT' columns
        augmenters: Dictionary of augmentation models from create_augmenters()
        aug_factor: Total multiplier (e.g., 3 = original + 2 augmented per sample)

    Returns:
        Augmented dataframe

    Example:
        >>> augmenters = create_augmenters(device='cuda')
        >>> augmented_df = augment_sentiment_dataset(
        >>>     sentiment_train_df,
        >>>     augmenters,
        >>>     aug_factor=3
        >>> )
    """
    augmented_rows = []

    print(f"\\nAugmenting {len(df)} samples with factor {aug_factor}...")
    print(f"Target size: {len(df) * aug_factor} samples")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        # Add original
        augmented_rows.append(row.to_dict())

        # Generate augmented versions
        text = row['Tweet']
        aug_texts = augment_text(text, augmenters, num_aug=aug_factor-1)

        # Add augmented versions (skip first which is original)
        for aug_text in aug_texts[1:]:
            aug_row = row.to_dict()
            aug_row['Tweet'] = aug_text
            augmented_rows.append(aug_row)

    augmented_df = pd.DataFrame(augmented_rows)

    print(f"✓ Augmentation complete!")
    print(f"  Original samples: {len(df)}")
    print(f"  Augmented samples: {len(augmented_df)}")
    print(f"  Increase: {len(augmented_df) / len(df):.1f}x")

    # Show sentiment distribution
    print(f"\\nAugmented Sentiment Distribution:")
    for sent, count in augmented_df['SENTIMENT'].value_counts().sort_index().items():
        sent_name = {0: 'negative', 1: 'neutral', 2: 'positive'}[sent]
        print(f"  {sent_name:8s}: {count} ({count/len(augmented_df)*100:.1f}%)")

    return augmented_df


# ============================================================================
# 4. DETAILED PERFORMANCE ANALYSIS
# ============================================================================

def analyze_sentiment_performance(model, val_loader, device, save_dir='./results'):
    """
    Comprehensive performance analysis for sentiment model

    Args:
        model: Trained sentiment model
        val_loader: Validation data loader
        device: 'cuda' or 'cpu'
        save_dir: Directory to save plots

    Returns:
        Tuple of (predictions, labels, probabilities)
    """
    SENTIMENT_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("\\n" + "="*70)
    print("SENTIMENT MODEL DETAILED ANALYSIS")
    print("="*70)

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
    plt.title('Sentiment Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sentiment_confusion_matrix.png', dpi=300, bbox_inches='tight')
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
    overall_precision = precision.mean()
    overall_recall = recall.mean()

    print("\\nOverall Metrics (Macro Avg):")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1: {overall_f1:.4f}")

    # Confidence analysis
    avg_confidence = np.max(all_probs, axis=1).mean()
    print(f"\\nAverage Prediction Confidence: {avg_confidence:.3f}")

    # Misclassification analysis
    misclassified = all_preds != all_labels
    num_misclassified = misclassified.sum()

    print(f"\\nMisclassifications: {num_misclassified} / {len(all_labels)} ({num_misclassified/len(all_labels)*100:.1f}%)")

    if num_misclassified > 0:
        # Most confident mistakes
        misclass_confidence = np.max(all_probs[misclassified], axis=1)
        top_k = min(5, num_misclassified)
        top_indices = np.argsort(misclass_confidence)[-top_k:][::-1]

        print(f"\\nTop {top_k} Most Confident Mistakes:")
        print(f"{'True':<10} {'Predicted':<10} {'Confidence':<12}")
        print("-" * 35)

        misclass_indices = np.where(misclassified)[0]
        for idx in top_indices:
            orig_idx = misclass_indices[idx]
            true_label = SENTIMENT_MAP[all_labels[orig_idx]]
            pred_label = SENTIMENT_MAP[all_preds[orig_idx]]
            conf = misclass_confidence[idx]
            print(f"{true_label:<10} {pred_label:<10} {conf:<12.3f}")

    print("\\n" + "="*70)

    return all_preds, all_labels, all_probs


# ============================================================================
# 5. USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use these improvements in your notebook
    """
    example_code = '''
# ============================================================
# EXAMPLE USAGE IN YOUR NOTEBOOK
# ============================================================

# 1. Import this module
import sentiment_improvements as si

# 2. Create augmenters (do this ONCE)
augmenters = si.create_augmenters(device='cuda')

# 3. Augment sentiment training data (ONLY training, not validation!)
sentiment_train_df_augmented = si.augment_sentiment_dataset(
    sentiment_train_df,
    augmenters,
    aug_factor=3  # 212 samples → ~640 samples
)

# 4. Use augmented data for training dataset
sentiment_train_dataset = SentimentDataset(
    sentiment_train_df_augmented,  # Use augmented version
    sentiment_tokenizer,
    MAX_SEQ_LENGTH
)

# Keep validation dataset unchanged
sentiment_val_dataset = SentimentDataset(
    sentiment_val_df,  # Original validation data
    sentiment_tokenizer,
    MAX_SEQ_LENGTH
)

# 5. Use Focal Loss instead of CrossEntropyLoss
criterion = si.FocalLoss(
    alpha=sentiment_class_weights,  # Class weights
    gamma=2.0  # Standard focal loss parameter
)

# 6. Initialize early stopping
early_stopping = si.EarlyStopping(
    patience=3,
    min_delta=0.001,
    mode='max'
)

# 7. Training loop with improvements
for epoch in range(SENTIMENT_EPOCHS):
    # ... training code ...

    # Validation
    val_f1 = validate(...)

    # Check early stopping
    if early_stopping(val_f1):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# 8. Detailed analysis after training
preds, labels, probs = si.analyze_sentiment_performance(
    sentiment_model,
    sentiment_val_loader,
    device,
    save_dir=RESULTS_DIR
)

# This will print:
# - Classification report with precision/recall/F1 per class
# - Confusion matrix (saved as image)
# - Most confident mistakes
# - Overall performance metrics
    '''

    print(example_code)


if __name__ == "__main__":
    print("Sentiment Improvement Functions Loaded!")
    print("\\nAvailable functions:")
    print("  1. FocalLoss - Class for focal loss")
    print("  2. EarlyStopping - Class for early stopping")
    print("  3. create_augmenters() - Create text augmentation models")
    print("  4. augment_sentiment_dataset() - Augment training data")
    print("  5. analyze_sentiment_performance() - Detailed analysis")
    print("\\nRun example_usage() to see how to use these in your notebook")
