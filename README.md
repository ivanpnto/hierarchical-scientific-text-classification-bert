# Hierarchical Scientific Text Classification with BERT

A two-level text classifier for scientific abstracts. Given an abstract, the model predicts:

1. **Domain:** the general scientific field (e.g. `CS`, `Medical`, `Civil`, `ECE`).
2. **Area:** the specific subfield within that domain (e.g. `Machine learning`, `Cancer`, `Cryptography`).

## Approach

```
                                   ┌──► Domain head ──────────────► domain
abstract ──► BERT encoder ──► h ───┤
         (bert-base-uncased,       └──► Area head of that domain ─► area
              shared)
```

- **Architecture:** one shared `bert-base-uncased` encoder, one global domain head and one linear area head per domain.
- **Training:** the loss is the sum of the domain cross-entropy and the area cross-entropy. The area loss uses the head of the *true* domain, so each area head only learns from abstracts of its own domain.
- **Inference:** the *predicted* domain routes each abstract to its area head. As a result, a domain error always causes an area error. For this reason the evaluation reports **hierarchical exact match** (domain and area both correct) alongside the per-level metrics.
- **Optimization:** AdamW (learning rate 2e-5, weight decay 0.01), linear warmup over 10% of the steps followed by linear decay, gradient clipping at 1.0 and mixed precision on GPU. The checkpoint with the lowest validation loss is kept.

## Dataset

[`river-martin/web-of-science-with-label-texts`](https://huggingface.co/datasets/river-martin/web-of-science-with-label-texts) from Hugging Face: Web of Science abstracts labeled with domain and area, with the train, validation and test splits provided by the dataset.

## Results

Test set (7,047 abstracts), `bert-base-uncased`, 3 epochs, maximum sequence length 256, seed 42, trained on a single NVIDIA T4.

| Level | Accuracy | Macro F1 |
|---|---|---|
| Domain | **90.3%** | 90.4% |
| Area | 77.6% | 74.5% |
| **Hierarchical exact match** (domain and area correct) | **77.5%** | — |

- **No overfitting:** validation loss decreased in every epoch (1.415 → 1.065 → 1.018), and test scores match validation scores (domain 90.2%, area 78.2%).
- **Where the errors come from:** when the domain is correct, the area is correct 85.8% of the time (77.5 / 90.3). Domain errors cost 9.7 points of exact match, and area errors within a correctly predicted domain cost another 12.8 points.
- **Class imbalance:** the area macro F1 is lower than the area accuracy, so rare or ambiguous areas are harder to classify than frequent ones.
- **Note on the area metric:** areas are compared by name, and a few area names appear in more than one domain. This is why the area accuracy is slightly higher than the hierarchical exact match, which is the stricter metric.

| Epoch | Train loss | Val loss | Val domain acc | Val area acc |
|---|---|---|---|---|
| 1 | 2.691 | 1.415 | 88.0% | 71.4% |
| 2 | 1.126 | 1.065 | 90.2% | 77.0% |
| 3 | 0.779 | 1.018 | 90.2% | 78.2% |

### Domain confusion matrix (test set)

Rows are true domains and columns are predicted domains.

| True \ Predicted | CS | Civil | ECE | MAE | Medical | Psychology | Biochemistry | Recall |
|---|---|---|---|---|---|---|---|---|
| **CS** | **936** | 12 | 15 | 7 | 11 | 9 | 13 | 93.3% |
| **Civil** | 24 | **572** | 7 | 12 | 7 | 5 | 7 | 90.2% |
| **ECE** | 20 | 18 | **697** | 10 | 3 | 3 | 0 | 92.8% |
| **MAE** | 13 | 14 | 6 | **475** | 8 | 6 | 7 | 89.8% |
| **Medical** | 6 | 10 | 2 | 8 | **2005** | 98 | 62 | 91.5% |
| **Psychology** | 15 | 6 | 5 | 5 | 114 | **911** | 4 | 85.9% |
| **Biochemistry** | 11 | 3 | 0 | 1 | 90 | 9 | **765** | 87.0% |

More than half of the domain errors (364 of 686) are confusions between Medical and either Psychology or Biochemistry. These life-science domains share vocabulary, for example in clinical psychology or medical biochemistry. The four engineering domains are rarely confused with each other.

## Project structure

```
src/
├── data.py       Dataset loading, label encoding and tokenization
├── model.py      HierarchicalBertClassifier (shared encoder + domain/area heads)
├── train.py      Training loop, validation and checkpointing
├── evaluate.py   Test evaluation, confusion matrices and predictions
└── utils.py      Seeding, metrics, plots and the label encoder
```

## Getting started

Requires Python 3.9 or newer. A CUDA-capable GPU is strongly recommended: on a single NVIDIA T4 (for example, a free Google Colab GPU runtime), training takes about 10 minutes per epoch.

```bash
git clone https://github.com/ivanpnto/hierarchical-scientific-text-classification-bert.git
cd hierarchical-scientific-text-classification-bert
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Training

```bash
python -m src.train --model_name bert-base-uncased --output_dir outputs \
  --epochs 3 --batch_size 8 --lr 2e-5 --max_length 256 --seed 42 --weight_decay 0.01
```

### Evaluation

```bash
python -m src.evaluate --model_name bert-base-uncased --output_dir outputs \
  --batch_size 8 --max_length 256
```

Both commands write their results to `outputs/` (ignored by Git):

| File | Content |
|---|---|
| `model.pt` | Best checkpoint (lowest validation loss) |
| `label_encoder.json` | Domain and area label mappings |
| `test_metrics.json` | Accuracy, macro F1, per-class report and hierarchical exact match |
| `confusion_domain.png`, `confusion_area.png` | Confusion matrices |
| `predictions.csv` | True and predicted labels for every test abstract |

## License

Released under the [MIT License](LICENSE).
