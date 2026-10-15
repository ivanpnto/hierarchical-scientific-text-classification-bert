# Hierarchical Scientific Text Classification with BERT

This project implements a **hierarchical text classification system** for scientific abstracts using a modern Transformer-based NLP model: **BERT base uncased**[cite: 1].

The input is the abstract of a scientific article, and the model predicts:
1. **domain**: The general scientific field (e.g., `CS`, `Medical`, `Civil`, `ECE`, etc.)[cite: 1].
2. **area**: A more specific subcategory inside that domain (e.g., `Machine learning`, `Cancer`, `Cryptography`, etc.)[cite: 1].

## Dataset
The dataset used is `river-martin/web-of-science-with-label-texts` from Hugging Face[cite: 1].

## Architecture
- **Shared Encoder**: `bert-base-uncased`[cite: 1].
- **Global Domain Head**: A linear classifier predicting the domain[cite: 1].
- **Domain-Specific Area Heads**: One separate linear classifier per domain[cite: 1].

## Usage

### Training
```bash
python -m src.train --model_name bert-base-uncased --output_dir outputs --epochs 3 --batch_size 8 --lr 2e-5 --max_length 256 --seed 42 --weight_decay 0.01
```[cite: 1]

### Evaluation
```bash
python -m src.evaluate --model_name bert-base-uncased --output_dir outputs --batch_size 8 --max_length 256
```[cite: 1]
```[cite: 1]