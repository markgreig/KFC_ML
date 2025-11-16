# Multi-Competitor NER & Sentiment Analysis for Food Delivery Tweets

This project fine-tunes deep learning models to extract competitor mentions from tweets and analyze sentiment for each competitor independently.

## ⚠️ IMPORTANT: Use the Fixed Version

**Recommended notebook:** `KFC_NER_Sentiment_FIXED.ipynb`

The original notebook had an issue with NER validation F1 being 0.0000. The fixed version uses a simplified, more robust approach. See `FIX_EXPLANATION.md` for details.

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

### 1. `KFC_NER_Sentiment_FIXED.ipynb` ⭐ **RECOMMENDED**

**What's different:**
- NER uses **single-label classification** (14-class: which competitor is tweet about?)
- Added **data validation** at every step with debugging output
- Enhanced **regex-based extraction** for finding all competitor mentions
- **Hybrid pipeline**: NER classifier + regex extraction = best of both worlds
- Much simpler and more robust

**Expected performance:**
- NER Validation F1: **>0.70** ✅
- NER Accuracy: **>0.75** ✅
- Sentiment F1: **>0.70** ✅

### 2. `KFC_Competitor_NER_Sentiment_Analysis.ipynb` ⚠️ **HAS ISSUES**

Original notebook with multi-label NER approach. **Known issue:** Validation F1 stays at 0.0000 (model doesn't learn). Use the fixed version instead.

See `FIX_EXPLANATION.md` for technical details on what went wrong and how it was fixed.

## Dataset Files

- `KFC_social_data.xlsx - Sheet1.csv` - Large dataset with 3,183 rows (primary training data)
- `KFC_training_sample.csv` - Smaller 265-row sample (alternative training option)
- `KFC_test_sample.csv` - 34 rows with labels for evaluation
- `KFC_test_sample_for_prediction.csv` - 34 rows without labels for final predictions

## How to Use

### 1. Open in Google Colab

1. Upload the notebook `KFC_NER_Sentiment_FIXED.ipynb` to Google Colab (⭐ recommended)
2. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU (T4 recommended)
3. Run all cells sequentially

### 2. Upload Data Files

When prompted, upload the four CSV files listed above.

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

### 4. Output Files

All outputs are saved to:
- `/content/drive/MyDrive/KFC_ML_Models/` - Trained models and tokenizers
- `/content/results/` - Predictions, visualizations, metrics

Key files:
- `KFC_test_predictions.csv` - Final predictions with one row per competitor
- `ner_best_model.pt` - Best NER model weights
- `sentiment_best_model.pt` - Best sentiment model weights
- `config.json` - Model configuration and performance metrics
- Various PNG files with training curves and confusion matrices

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
