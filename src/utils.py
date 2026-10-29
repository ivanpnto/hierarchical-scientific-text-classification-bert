import random
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def set_seed(seed: int = 42) -> None:
    """Fix all common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    average: str = "macro",
) -> Dict[str, Any]:
    """Return a compact metric dictionary."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(labels),
            zero_division=0,
            output_dict=True,
        ),
    }

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    title: str,
    output_path: str | Path,
) -> None:
    """Save a confusion matrix plot using matplotlib only."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), max(6, len(labels) * 0.35)))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

@dataclass
class HierarchicalLabelEncoder:
    """
    Stores the mappings needed for hierarchical prediction.
    - domain is a global label space
    - area is local to each domain
    """
    domain2id: Dict[str, int]
    id2domain: list[str]
    area2id_by_domain: Dict[str, Dict[str, int]]
    id2area_by_domain: Dict[str, list[str]]

    @property
    def num_domains(self) -> int:
        return len(self.id2domain)

    def num_areas_for_domain(self, domain_name: str) -> int:
        return len(self.id2area_by_domain[domain_name])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain2id": self.domain2id,
            "id2domain": self.id2domain,
            "area2id_by_domain": self.area2id_by_domain,
            "id2area_by_domain": self.id2area_by_domain,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HierarchicalLabelEncoder":
        return cls(
            domain2id={str(k): int(v) for k, v in data["domain2id"].items()},
            id2domain=[str(x) for x in data["id2domain"]],
            area2id_by_domain={
                str(domain): {str(area): int(idx) for area, idx in mapping.items()}
                for domain, mapping in data["area2id_by_domain"].items()
            },
            id2area_by_domain={
                str(domain): [str(x) for x in items]
                for domain, items in data["id2area_by_domain"].items()
            },
        )

    def encode_domain(self, domain: str) -> int:
        return self.domain2id[domain]

    def encode_area(self, domain: str, area: str) -> int:
        return self.area2id_by_domain[domain][area]

    def decode_domain(self, domain_id: int) -> str:
        return self.id2domain[int(domain_id)]

    def decode_area(self, domain: str, area_id: int) -> str:
        return self.id2area_by_domain[domain][int(area_id)]
