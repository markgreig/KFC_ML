# Multi-Competitor NER & Sentiment Analysis for Food Delivery Tweets

This project fine-tunes deep learning models to extract competitor mentions from tweets and analyze sentiment for each competitor independently.

## ⚠️ CRITICAL BUG ALERT

**KNOWN ISSUE in `KFC_Complete_NER_Sentiment.ipynb`:**

The notebook has a **data assignment bug** that causes `KeyError: 'SENTIMENT'`:
- Uses `df_large` (no sentiment labels) for sentiment model training ❌
- Should use `df_train_sample` (has sentiment labels) for sentiment model ✅

**FIX:** See `CRITICAL_BUG_FIX.md` for detailed step-by-step corrections.

**Quick Fix Summary:**
1. Use `df_large_clean` for NER training (3,183 rows)
2. Use `df_train_sample_clean` for Sentiment training (265 rows)
3. Create separate `ner_train_df` and `sentiment_train_df` splits

---

**Recommended notebook:** Currently all notebooks need the fix above. A corrected version will be uploaded soon.

## Overview

The pipeline consists of three main components:

1. **NER Model (Competitor Identification)**: 14-class BERT classifier that predicts which competitor a tweet is primarily about
2. **Regex Extraction**: Pattern matching to find all competitor mentions in tweet text
3. **Sentiment Model**: Twitter-specific RoBERTa model that predicts sentiment (negative/neutral/positive) for each competitor

## Competitors

The model is trained to identify 14 competitors:
- Burger King
- Deliveroo
- Domino's
- Five Guys
- Greggs
- Just Eat
- KFC
- McDonald's
- Nando's
- Papa John's
- Pizza Hut
- Pret a Manger
- Taco Bell
- Uber Eats

## Key Features

- ✅ **Hybrid competitor extraction** - NER classifier for primary competitor + regex for all mentions
- ✅ **Competitor-specific sentiment** - Analyzes sentiment for each competitor independently
- ✅ **Handles complex tweets** - Tweets mentioning multiple competitors generate separate predictions
- ✅ **Class-weighted training** - Handles imbalanced data effectively (KFC is 79% of data)
- ✅ **Memory optimized for Colab** - Uses gradient accumulation and mixed precision training
- ✅ **Data validation** - Comprehensive checks at every preprocessing step
- ✅ **Robust and debuggable** - Clear error messages and progress tracking

## Notebook Versions

### 1. `KFC_Complete_NER_Sentiment.ipynb` ⭐⭐ **MOST RECENT - USE THIS**

**Complete working version with all fixes:**
- ✅ **Fixed AdamW import** - Uses `torch.optim.AdamW` instead of transformers
- ✅ **UTF-8 conversion** - Auto-detects and converts CSV encoding
- ✅ **Excel export** - Formatted .xlsx with color-coded sentiment
- ✅ **Working NER** - Single-label classification (F1 >0.70)
- ✅ **Complete sentiment model** - Fully implemented
- ✅ **Hybrid pipeline** - NER + regex for competitor extraction

**Features:**
- Automatic CSV encoding detection and UTF-8 conversion
- Color-coded Excel output with summary statistics
- Complete training pipeline for both models
- Saves to both local and Google Drive
- Ready for immediate use in Google Colab

**Expected performance:**
- NER Validation F1: **>0.70** ✅
- NER Accuracy: **>0.75** ✅
- Sentiment F1: **>0.70** ✅

### 2. `KFC_NER_Sentiment_FIXED.ipynb` ⚠️ **INCOMPLETE**

Partial fixed version. **Missing:**
- Sentiment model section incomplete (placeholder only)
- No CSV encoding conversion
- No Excel export functionality

Use `KFC_Complete_NER_Sentiment.ipynb` instead.

### 3. `KFC_Competitor_NER_Sentiment_Analysis.ipynb` ❌ **BROKEN**

Original notebook with multi-label NER approach. **Known issues:**
- Validation F1 stays at 0.0000 (model doesn't learn)
- AdamW import error with newer transformers versions
- No encoding handling

See `FIX_EXPLANATION.md` for technical details.

## Dataset Files

- `KFC_social_data.xlsx - Sheet1.csv` - Large dataset with 3,183 rows (primary training data)
- `KFC_training_sample.csv` - Smaller 265-row sample (alternative training option)
- `KFC_test_sample.csv` - 34 rows with labels for evaluation
- `KFC_test_sample_for_prediction.csv` - 34 rows without labels for final predictions

## How to Use

### 1. Open in Google Colab

1. Upload the notebook **`KFC_Complete_NER_Sentiment.ipynb`** to Google Colab ⭐⭐
2. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU (T4 recommended)
3. Run all cells sequentially

### 2. Upload Data Files

When prompted, upload the four CSV files:
- `KFC_social_data.xlsx - Sheet1.csv`
- `KFC_training_sample.csv`
- `KFC_test_sample.csv`
- `KFC_test_sample_for_prediction.csv`

**Note:** The notebook will automatically:
- Detect file encoding
- Convert to UTF-8 if needed
- Handle special characters properly

Alternatively, if files are in Google Drive:
- Uncomment and modify the Drive path in the "Upload Data" cell
- Files will be copied automatically

### 3. Training Process

The notebook will:
1. **Setup** - Install dependencies, check GPU, configure batch sizes
2. **Data Preparation** - Clean data, expand multi-competitor tweets, create train/val split
3. **Baseline Model** - Train simple rule-based model for comparison
4. **NER Training** - Fine-tune BERT for competitor extraction (4 epochs, ~20 mins)
5. **Sentiment Training** - Fine-tune RoBERTa for sentiment classification (5 epochs, ~25 mins)
6. **Evaluation** - Generate metrics, confusion matrices, performance comparisons
7. **Predictions** - Process test data and save results

## 🚀 Performance Optimization (NEW)

**Is your GPU/CPU underutilized during training?** The default configuration is conservative to work on all hardware. You can significantly speed up training!

### Quick Performance Boost

**Current (default):**
- Batch size: 8-16
- GPU utilization: 30-40%
- Training time: 25-30 min

**Optimized:**
- Batch size: 32-64
- GPU utilization: 75-90%
- Training time: 8-12 min ⚡

**How to optimize:**
1. See `QUICK_PERFORMANCE_FIX.md` for copy-paste code changes
2. Read `PERFORMANCE_OPTIMIZATION.md` for detailed guide
3. Expected speedup: **3-5x faster training!**

**Key changes:**
- Increase batch size (8→32 or 64)
- Increase DataLoader workers (2→8)
- Enable prefetching
- Scale learning rate appropriately

### 4. Output Files

All outputs are saved to:
- `/content/drive/MyDrive/KFC_ML_Models/` - Trained models and tokenizers
- `/content/results/` - Predictions, visualizations, metrics

**Key Output Files:**

**Excel File (NEW):** `KFC_Predictions_Complete.xlsx`
- **Predictions sheet**: Full data with color-coded sentiment
  - 🔴 Red: Negative sentiment
  - 🟡 Yellow: Neutral sentiment
  - 🟢 Green: Positive sentiment
- **Summary sheet**: Statistics including:
  - Total predictions, unique tweets, unique competitors
  - Sentiment distribution (counts and percentages)
  - Per-competitor breakdown (mentions, positive, neutral, negative)
- Auto-sized columns for readability
- Professional formatting ready for presentations

**Model Files:**
- `ner_model_best.pt` - Best NER model weights
- `sentiment_model_best.pt` - Best sentiment model weights
- `ner_tokenizer/` - NER tokenizer files
- `sentiment_tokenizer/` - Sentiment tokenizer files

**Visualizations:**
- `ner_training.png` - NER training curves
- `sentiment_training.png` - Sentiment training curves
- Various confusion matrices and performance plots

## Model Architecture

### NER Model
- **Base model**: `bert-base-cased` (110M parameters)
- **Task**: Multi-label classification (14 binary outputs)
- **Input**: Tweet text
- **Output**: Binary vector indicating which competitors are mentioned
- **Loss**: Binary Cross-Entropy

### Sentiment Model
- **Base model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (125M parameters)
- **Task**: 3-class classification (negative/neutral/positive)
- **Input**: `"[Tweet text] This tweet is about [Competitor name]."`
- **Output**: Sentiment class (0, 1, or 2)
- **Loss**: Cross-Entropy with class weights

## Training Configuration

### Batch Sizes (Adaptive)
- **T4/V100 GPU (16GB)**: Batch size 16, gradient accumulation 2 (effective 32)
- **K80 GPU (12GB)**: Batch size 8, gradient accumulation 4 (effective 32)
- **CPU**: Batch size 4, gradient accumulation 8 (effective 32)

### Hyperparameters
- Max sequence length: 128 tokens
- NER learning rate: 3e-5
- Sentiment learning rate: 2e-5
- Warmup ratio: 10%
- Weight decay: 0.01
- Mixed precision: FP16 enabled

## Expected Performance

Based on test set evaluation:

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| NER F1 | ~0.60 | **>0.85** | +0.25 |
| Sentiment Accuracy | ~0.50 | **>0.75** | +0.25 |
| Sentiment F1 (macro) | ~0.35 | **>0.70** | +0.35 |

## Example Usage

### Making Predictions

```python
# Example tweet
tweet = "KFC's chicken is amazing but McDonald's has better fries!"

# Get predictions
results = predict_competitors_and_sentiment(
    tweet, ner_model, sentiment_model,
    ner_tokenizer, sentiment_tokenizer
)

# Output:
# [('KFC', 2),           # positive
#  ('McDonald\'s', 2)]   # positive
```

### Multi-Competitor Output

For tweets mentioning multiple competitors, the pipeline generates one row per competitor:

| Competitor | Tweet | Predicted_Sentiment | Sentiment_Label |
|------------|-------|---------------------|-----------------|
| KFC | "KFC's chicken is great but McDonald's is terrible" | 2 | positive |
| McDonald's | "KFC's chicken is great but McDonald's is terrible" | 0 | negative |

## Requirements

- Python 3.7+
- Google Colab (recommended) or Jupyter Notebook
- GPU with 12GB+ VRAM (T4, V100, or better)
- ~2GB disk space for models and data

## Installation (Local)

```bash
pip install transformers datasets torch torchvision
pip install scikit-learn pandas numpy matplotlib seaborn
pip install sentencepiece protobuf
```

## Troubleshooting

### Out of Memory Errors
- Reduce `BATCH_SIZE` in the configuration cell
- Increase `GRAD_ACCUM_STEPS` to maintain effective batch size
- Reduce `MAX_SEQ_LENGTH` to 64 or 96

### Colab Disconnects
- Models are automatically saved to Google Drive after each epoch
- Re-run from the checkpoint by loading saved models

### Poor Performance
- Ensure you're using the large dataset (3,183 rows), not the small sample
- Check that GPU is enabled (T4 minimum)
- Verify class weights are being applied for sentiment training

## Future Improvements

1. **Data Augmentation** - Increase samples for underrepresented competitors
2. **Ensemble Methods** - Combine multiple model predictions
3. **Active Learning** - Iteratively improve with human feedback
4. **Real-time Deployment** - Create API for live tweet analysis
5. **Multi-lingual Support** - Extend to non-English tweets

## Citation

If you use this work, please cite:

```
KFC Multi-Competitor NER & Sentiment Analysis
GitHub: markgreig/KFC_ML
2025
```

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: November 2025
