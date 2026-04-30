# Multi-Label Audio Genre Classification using Deep Learning

---

## Motivation

Music genre classification is a fundamental task in Music Information Retrieval (MIR) with significant applications in recommendation systems, music streaming services, content organization, and automated music curation. Traditional manual genre labeling is time-consuming and subjective. This project addresses the need for robust, automated approaches to classify music tracks into multiple genres simultaneously.

**Key Motivations:**

- **Multi-label Nature**: Real-world music often spans multiple genres (e.g., "Alternative Rock" + "Indie")
- **Audio Diversity**: The FMA dataset contains diverse audio characteristics requiring models to learn robust representations
- **Model Comparison**: Evaluate different deep learning paradigms (CNN-RNN, Feature Extraction + ML, Transformer) on the same task
- **Production-Ready Pipeline**: Create reliable, reproducible pipelines for audio processing and classification

---

## Problem Definition

### Task

**Multi-label audio genre classification** on the Free Music Archive (FMA) dataset.

### Objectives

1. Train models to classify music tracks into **8 top genres** from the FMA Small subset
2. Handle **multi-label assignments** (tracks can belong to multiple genres simultaneously)
3. Optimize for **F1-score (macro)** to balance precision and recall across all genres
4. Create reproducible pipelines with consistent train/val/test splits
5. Compare performance across different architectural approaches

### Challenges

- **Multi-label imbalance**: Some genres are more prevalent than others
- **Audio quality variations**: MP3 compression artifacts and encoding issues
- **Temporal dependencies**: Genre characteristics span entire tracks (30+ seconds)
- **Computational constraints**: Training large models on limited GPU resources
- **Overfitting risk**: Small dataset relative to model capacity

### Evaluation Metrics

- **F1-macro**: Unweighted average F1 score across all classes
- **F1-micro**: Weighted F1 score (favors prevalent classes)
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Precision/Recall**: Per-class performance analysis
- **Hamming Loss**: Fraction of incorrectly predicted labels



---

## Methodology

### Audio Processing Pipeline

1. **Data Loading**: MP3/WAV files from FMA Small dataset (8000 tracks)
2. **Audio Resampling**: Normalize to 16 kHz sample rate
3. **Spectrogram Generation**:
   - Mel-scale filterbank with 128 bins
   - 25ms window, 10ms hop length
   - 1945 time frames per track (30 seconds)
4. **Augmentation**: Mixup-style mixing during training
5. **Normalization**: Per-sample mean-variance normalization

### Train/Val/Test Split

- **Training**: 60% (4800 tracks)
- **Validation**: 20% (1600 tracks)
- **Testing**: 20% (1600 tracks)
- **Seeding**: Deterministic split via fixed random seed

### Hyperparameter Optimization

Each model was optimized for the target task:

| Aspect            | CRNN                        | VGGish+XGBoost | AST               |
| ----------------- | --------------------------- | -------------- | ----------------- |
| **Batch Size**    | 16                          | N/A            | 3                 |
| **Learning Rate** | 0.001                       | N/A            | 1e-4              |
| **Epochs**        | 20                          | Single train   | 15                |
| **Optimizer**     | Adam                        | N/A            | Adam              |
| **Weight Decay**  | 1e-4                        | N/A            | 0.0               |
| **Scheduler**     | CosineAnnealingWarmRestarts | N/A            | Linear            |
| **Loss Function** | BCEWithLogitsLoss           | N/A            | BCEWithLogitsLoss |

---

## Dataset

### Free Music Archive (FMA) Small

- **Total Tracks**: 8000 unique music files
- **Genres**: 8 top genres selected from 114 total
- **Genre Distribution**: Multi-label (tracks can have multiple genre tags)
- **Audio Format**: MP3 (22050 Hz mono initially, resampled to 16 kHz)
- **Track Duration**: ~30 seconds each
- **Total Duration**: ~67 hours of audio

### Genre Classes

1. Genre ID 21
2. Genre ID 10
3. Genre ID 17
4. Genre ID 38
5. Genre ID 12
6. Genre ID 2
7. Genre ID 15
8. Genre ID 1235

---

## 💻 Installation & Setup

### System Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration, optional but recommended)
- **cuDNN**: 8.0+ (if using GPU)
- **FFmpeg**: 4.2+ (for audio conversion)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional)

---

### Step 1: Repository Setup

```bash
# Clone the repository
git clone <repository-url>
cd Deep-Learning-Project

# Verify Python version
python --version  # Should be 3.8+

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install base requirements
pip install -r requirements.txt

# For AST pipeline (Audio Spectrogram Transformer) - (BEST MODEL)
pip install -r requirements_ast.txt

# For VGGish+XGBoost pipeline
pip install -r requirements_vggish.txt

# Optional: For web application
pip install flask flask-cors python-dotenv
```

### Step 3: GPU Setup (Optional but Recommended)

```bash
# Verify CUDA installation
nvidia-smi

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Step 4: Download Dataset

```bash
# The FMA Small dataset should be in: fma_small/
# If not present, download from: https://github.com/mdeff/fma

# Verify dataset structure
ls fma_small/ | head -5  # Should show: 000, 001, 002, ...
find fma_small -name "*.mp3" | wc -l  # Should show ~8000
```

### Step 5: Audio Data Preparation

```bash
# Diagnose audio quality (first 100 files)
python diagnose_audio_corruption.py --audio-dir fma_small --max-files 100

# Fix corrupted audio and convert to WAV (takes 2-4 hours)
python fix_audio_corruption.py \
    --audio-dir fma_small \
    --sample-rate 16000 \
    --workers 8
```



## Web Application

###### Starting the Application

```bash
# Navigate to app directory
cd app

# Option 1: Run with Flask development server
python main.py

# Option 2: Run with uvicorn
pip install uvicorn

#Run the app
uvicorn app.main:app --reload --port 8000
```

**Access the web interface:**

- Open browser: `http://localhost:8000`
- API endpoint: `http://localhost:8000/api/predict`

### Web Interface Features

1. **Upload Audio**
   
   - Drag-and-drop MP3/WAV files
   - Maximum file size: 500MB
   - Supported formats: MP3, WAV, FLAC, OGG

2. **Real-time Predictions**
   
   - Instant genre classification
   - Confidence scores per genre
   - Genre ranking display
   - Visualization of probability distribution

3. **Batch Processing**
   
   - Upload multiple files
   - Parallel processing with worker threads
   - CSV/JSON export of results
   - Progress tracking

4. **Model Selection & Comparison**
   
   - Switch between AST, VGGish+XGBoost, CRNN
   - Performance comparison
   - Real-time metrics display
   - Model info and statistics

---



### Results and Analysis

### Overall Comparison

| Model              | F1-Macro | F1-Micro | ROC-AUC-Macro | Precision | Recall |
| ------------------ | -------- | -------- | ------------- | --------- | ------ |
| **CRNN**           | 0.0129   | 0.1265   | (not used)    | 0.0204    | 0.0117 |
| **VGGish+XGBoost** | 0.5948   | 0.6149   | 0.9153        | 0.7991    | 0.4954 |
| **AST**            | 0.6567   | 0.6686   | 0.9311        | 0.6423    | 0.7044 |

### CRNN Results

**Model Configuration:**

- Parameters: 12.5M
- Architecture: 4 conv blocks + 2-layer BiLSTM
- Training Time: ~2 hours (20 epochs)

**Performance:**

- Best Epoch: 17/20
- Best F1-Macro: 0.0129
- Best F1-Micro: 0.1265
- ROC-AUC-Micro: 0.8939 (only strong metric)

**Analysis:**

- Severe underfitting despite 12.5M parameters
- High ROC-AUC but low F1 indicates poor threshold calibration
- Reinitialized classifier weights (size mismatch: 527 → 8 classes) forced training from scratch
- Low learning rate (2e-05) insufficient for random classifier convergence
- Recommendation: Increase learning rate, extend training, or use warmup strategy

### VGGish + XGBoost Results

**Model Configuration:**

- Feature Extractor: Pre-trained VGGish (128-d embeddings)
- Classifier: XGBoost with default hyperparameters
- Training Time: <5 minutes

**Performance on Validation Set:**

- F1-Macro: 0.5948
- F1-Micro: 0.6149
- ROC-AUC-Macro: 0.9153
- ROC-AUC-Micro: 0.9227
- Subset Accuracy: 0.4929

**Analysis:**

- Strong baseline performance: 59.5% F1-macro
- Transfer learning (AudioSet) provides excellent initialization
- High ROC-AUC (0.92) indicates good ranking ability
- Moderate F1 suggests precision-recall tradeoff
- Genre ID 21 performs excellently (F1=0.82)
- Genre ID 10 performs poorly (F1=0.21) → class imbalance issue
- XGBoost provides interpretable feature importance
- Efficient production-ready model

### AST (Audio Spectrogram Transformer) Results

**Model Configuration:**

- Architecture: Patched Vision Transformer (ViT)
- Pre-training: AudioSet checkpoint (527 classes)
- Classifier Head: Reinitialized for 8-class task
- Training Time: ~4 hours (15 epochs to convergence)

**Training Progression (Selected Epochs):**

| Epoch               | F1-Macro   | F1-Micro   | Precision  | Recall     | ROC-AUC-Macro |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ------------- |
| 1                   | 0.4135     | 0.4316     | 0.4143     | 0.4348     | 0.8088        |
| 3                   | 0.6105     | 0.6133     | 0.5927     | 0.6313     | 0.8945        |
| 5                   | 0.6720     | 0.6726     | 0.6428     | 0.7044     | 0.9265        |
| 7                   | 0.7492     | 0.7482     | 0.7117     | 0.7772     | 0.9483        |
| 10                  | 0.7610     | 0.7625     | 0.7168     | 0.8123     | 0.9532        |
| **Best (Epoch 10)** | **0.6567** | **0.6686** | **0.6423** | **0.7044** | **0.9311**    |

**Key Observations:**

- Rapid convergence: F1-macro improved from 0.41 to 0.75 in first 7 epochs
- Peak performance at epoch 10 before noise accumulation
- Well-calibrated confidence estimates (high ROC-AUC)
- Steady improvement in recall, slight precision degradation after epoch 7
- Best overall F1-macro: 0.6567 (9% improvement over VGGish+XGBoost)

**Analysis:**

- Pre-trained transformer backbone provides strong feature representations
- Optimized hyperparameters (higher LR, longer training) enable effective learning
- Patch-based processing captures local and global audio patterns
- Attention mechanism helps identify genre-discriminative spectral regions
- Early stopping would have captured peak F1-macro at epoch 10
- Production recommendation: Use checkpoint from epoch 10

---

## Continual Learning & Future Enhancements

### Overview

We implement continual learning strategies to update our models with new music genres or audio patterns without forgetting previously learned knowledge.

### Strategies Used

- **Elastic Weight Consolidation (EWC)** – Protects important model weights when learning new tasks.
- **Experience Replay** – Maintains a small buffer of past examples mixed with new data.
- **Learning without Forgetting (LwF)** – Uses soft targets from the original model to preserve old knowledge without storing old data.

### Workflow

A monthly update cycle: collect new labeled tracks → select strategy → fine-tune with lower learning rate → evaluate on old and new tasks → promote to production if original performance stays above 95%.

---
