# Gene-Family-Classification
# DNA Sequence Classification into Functional Gene Families

This repository contains an end-to-end Deep Learning pipeline developed as part of my Master's Degree (M.Sc.) studies. The project models biological DNA sequences through a Natural Language Processing (NLP) lens, benchmarking classical Machine Learning against an optimized hybrid CNN-BiLSTM neural network to classify genomic data into 7 distinct functional categories.

## 📌 Project Overview
In genomics, automated identification of gene functions is a critical challenge. Raw DNA sequences cannot be directly processed by machine learning algorithms as raw strings. To address this, this project applies a **6-mer sliding-window tokenization strategy**, breaking down long genomic sequences into overlapping biological "words" of length $k=6$. 

This transformation turns raw biological data into a natural language format, allowing the models to treat a DNA strand as a "sentence" where 6-mers represent the individual words.

### Dataset Profile
The pipeline utilizes **4,380 unique DNA sequences** sourced from the Kaggle Gene Classification dataset. Each entry consists of a raw nucleotide string ($A, C, G, T$) associated with a specific integer class label ranging from 0 to 6. These labels correspond to seven distinct functional gene families:
*   **Class 0:** G protein-coupled receptors
*   **Class 1:** Tyrosine kinases
*   **Class 2:** Tyrosine phosphatases
*   **Class 3:** Synthetases
*   **Class 4:** Synthases
*   **Class 5:** Ion channels
*   **Class 6:** Transcription factors
*   You can find the dataset here -> [Kaggle Gene Classification Dataset](https://www.kaggle.com/datasets/nageshsinghmahayan/dna-sequence-dataset).

### Technical Challenges Addressed
1. **Severe Class Imbalance:** The dataset is highly skewed. Class 6 (Transcription factors) dominates with **1,343 samples**, whereas Class 5 (Ion channels) is severely underrepresented with only **240 samples**. To prevent the model from simply memorizing the majority class, performance is evaluated using **Macro-F1 scores** and stratified training logic.
2. **Variable Sequence Lengths:** DNA strands do not have a fixed size. This was resolved via dynamic sequence padding and truncation using Keras `pad_sequences` to ensure a uniform input dimension of 500.

---

## 🧱 Architectural Evolution & Solutions

### 1. The Baseline Bottleneck
The initial implementation relied on a Bag-of-Words (BoW) strategy using `CountVectorizer` followed by a 1D-CNN and a `Flatten` layer. Critical structural flaws included:
*   **Loss of Biological Syntax:** The BoW approach flattened the DNA sequence into a frequency histogram, completely discarding the sequential order of nucleotide bases.
*   **Parameter Explosion:** Passing the high-dimensional feature space through a `Flatten` layer caused a massive parameter explosion (**~1.5 Million trainable parameters**), leading to an extreme risk of overfitting on our 4,380 samples.

### 2. The Optimized Hybrid Architecture (Proposed Model)
To overcome these limitations, the network was re-engineered as a dense, sequential deep learning pipeline:
*   **Continuous Vector Embeddings:** Maps each 6-mer into a 64-dimensional continuous vector space to capture "semantic" biological similarities.
*   **Localized Motif Extraction (1D-CNN):** Automatically extracts local conserved sequences (motifs) using 64 filters and a kernel size of 5.
*   **Bidirectional Temporal Context (BiLSTM):** Processes the sequence in both directions ($5' \rightarrow 3'$ and $3' \rightarrow 5'$) via a Bidirectional LSTM layer (32 units) to understand the full structural context of the DNA strand.

### 💻 Model Architecture Implementation

Here is the core TensorFlow/Keras implementation of our hybrid network as deployed in the pipeline:

```python
import tensorflow as tf
from tensorflow.keras import layers

def create_hybrid_model(vocab_size, max_len=500):
    model = tf.keras.Sequential([
        layers.Input(shape=(max_len,)),
        
        # 1. Dense Continuous Embeddings
        layers.Embedding(input_dim=vocab_size, output_dim=64),
        
        # 2. Localized Motif Extraction (1D-CNN)
        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=4),
        
        # 3. Bidirectional Temporal Context (BiLSTM)
        layers.Bidirectional(layers.LSTM(32)),
        
        # 4. Dense Layers & Regularization
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
