# Quick Performance Fix - Maximize GPU/CPU Utilization

## TL;DR - Apply These Changes

If your GPU/CPU are underutilized during training, apply these fixes to the notebook:

### 1. Increase Batch Size (Cell: GPU Configuration)

**Find this cell:**
```python
if gpu_memory_gb >= 15:
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2
else:
    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4
```

**Replace with:**
```python
# OPTIMIZED: Maximize GPU utilization
if gpu_memory_gb >= 40:  # A100 40GB
    BATCH_SIZE = 64
    GRAD_ACCUM_STEPS = 1
    NUM_WORKERS = 16
elif gpu_memory_gb >= 24:  # RTX 3090/4090
    BATCH_SIZE = 48
    GRAD_ACCUM_STEPS = 1
    NUM_WORKERS = 12
elif gpu_memory_gb >= 16:  # T4/V100 16GB
    BATCH_SIZE = 32
    GRAD_ACCUM_STEPS = 1
    NUM_WORKERS = 8
else:  # K80 12GB
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2
    NUM_WORKERS = 4

EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS
PREFETCH_FACTOR = 4

print(f"OPTIMIZED Configuration:")
print(f"  Physical batch: {BATCH_SIZE}")
print(f"  Gradient accum: {GRAD_ACCUM_STEPS}")
print(f"  Effective batch: {EFFECTIVE_BATCH_SIZE}")
print(f"  DataLoader workers: {NUM_WORKERS}")
print(f"  Prefetch factor: {PREFETCH_FACTOR}")
```

**Expected improvement**: 2-4x faster training

### 2. Optimize DataLoaders (All DataLoader cells)

**Find:**
```python
DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
```

**Replace with:**
```python
DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,        # Use optimized worker count
    pin_memory=True,                # Faster GPU transfer
    prefetch_factor=PREFETCH_FACTOR, # Load batches ahead
    persistent_workers=True          # Keep workers alive
)
```

**Apply to all 6 DataLoaders:**
- ner_train_loader
- ner_val_loader
- ner_test_loader
- sentiment_train_loader
- sentiment_val_loader
- sentiment_test_loader

**Expected improvement**: 30-50% faster data loading

### 3. Scale Learning Rate (Training cells)

**Find:**
```python
NER_LEARNING_RATE = 2e-5
SENTIMENT_LEARNING_RATE = 2e-5
```

**Replace with:**
```python
# Scale LR with batch size (sqrt scaling)
BASE_LR = 2e-5
BATCH_SIZE_SCALE_FACTOR = np.sqrt(EFFECTIVE_BATCH_SIZE / 8)  # Base was 8

NER_LEARNING_RATE = BASE_LR * BATCH_SIZE_SCALE_FACTOR
SENTIMENT_LEARNING_RATE = BASE_LR * BATCH_SIZE_SCALE_FACTOR

print(f"Scaled learning rates:")
print(f"  NER LR: {NER_LEARNING_RATE:.2e}")
print(f"  Sentiment LR: {SENTIMENT_LEARNING_RATE:.2e}")
print(f"  Scale factor: {BATCH_SIZE_SCALE_FACTOR:.2f}x")
```

**Expected improvement**: Better convergence with larger batches

### 4. Add GPU Monitoring (New cell after imports)

**Add this new cell:**
```python
# GPU Monitoring
!pip install -q pynvml

import pynvml
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def print_gpu_utilization():
    \"\"\"Print current GPU utilization\"\"\"
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)

    print(f"\\nGPU Utilization:")
    print(f"  GPU Compute: {util.gpu}%")
    print(f"  GPU Memory: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB ({mem.used/mem.total*100:.1f}%)")

    # Target metrics
    if util.gpu < 70:
        print(f"  ⚠️  GPU utilization is low! Consider increasing batch size.")
    else:
        print(f"  ✅ GPU utilization is good!")

    return util.gpu, mem.used/mem.total*100

# Check initial utilization
print_gpu_utilization()
```

**Call during training:**
```python
# In training loop, every 50 steps:
if step % 50 == 0:
    gpu_util, mem_util = print_gpu_utilization()
```

### 5. Enable Mixed Precision (Already enabled, but optimize)

**Find autocast usage, enhance with:**
```python
# Check if bfloat16 is supported (better than float16)
use_bf16 = torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"Using mixed precision: {autocast_dtype}")

# In training:
with autocast(dtype=autocast_dtype):
    outputs = model(input_ids, attention_mask)
```

## Quick Batch Size Test

**Add this cell to find your GPU's max batch size:**
```python
def find_max_batch_size(model_name, max_length=128, start_size=8):
    \"\"\"Binary search for maximum batch size\"\"\"
    print(f"Finding maximum batch size for {model_name}...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=14
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    batch_size = start_size
    max_working = start_size

    while batch_size <= 512:
        try:
            # Create dummy batch
            dummy_text = ["This is a test tweet"] * batch_size
            inputs = tokenizer(
                dummy_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Forward + backward
            outputs = model(**inputs, labels=torch.zeros(batch_size, dtype=torch.long).to(device))
            loss = outputs.loss
            loss.backward()

            max_working = batch_size
            print(f"  ✅ Batch size {batch_size}: OK")

            # Try larger
            batch_size = int(batch_size * 1.5)
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Batch size {batch_size}: OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    del model
    torch.cuda.empty_cache()

    print(f"\\nMaximum batch size: {max_working}")
    print(f"Recommended (80% of max): {int(max_working * 0.8)}")
    return max_working

# Run test
max_bs = find_max_batch_size(NER_MODEL_NAME)
```

## Expected Results After Optimization

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Batch Size | 8-16 | 32-64 | 2-4x |
| GPU Utilization | 30-40% | 75-90% | 2.5x |
| Training Time (5 epochs) | 25-30 min | 8-12 min | 3x faster |
| Samples/sec | 100 | 300-400 | 3-4x |
| GPU Memory Usage | 4-6 GB | 12-15 GB | Using available |

## Verification

After applying changes, **check these metrics during training:**

✅ **Good Performance:**
- GPU Utilization: 75-95%
- GPU Memory: 80-95% used
- Training speed: >250 samples/sec
- No "waiting for data" delays

❌ **Still Underutilized:**
- GPU Utilization: <60%
- GPU Memory: <70% used
- Training speed: <150 samples/sec

**If still underutilized:**
1. Increase BATCH_SIZE further (test with find_max_batch_size)
2. Increase NUM_WORKERS to 12 or 16
3. Reduce GRAD_ACCUM_STEPS to 1
4. Consider using -large or -xl model variants

## Safety Notes

- **Start conservative**: Apply changes gradually
- **Monitor first epoch**: Watch for OOM errors
- **Save checkpoints**: Longer training with larger batches
- **Adjust LR**: May need to tune if accuracy drops

## One-Line Summary

**Change batch size from 8→32, workers from 2→8, add prefetching, scale LR = 3-5x faster training!**
