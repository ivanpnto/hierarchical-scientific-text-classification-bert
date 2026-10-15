import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, List, Optional, Tuple

class HierarchicalBertClassifier(nn.Module):
    """
    Hierarchical classifier:
    - BERT encoder shared by all tasks
    - one global domain head
    - one area head per domain
    """
    def __init__(
        self,
        domain_names: List[str],
        area_names_by_domain: Dict[str, List[str]],
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.domain_names = list(domain_names)
        self.area_names_by_domain = {k: list(v) for k, v in area_names_by_domain.items()}

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.domain_classifier = nn.Linear(hidden_size, len(self.domain_names))
        self.area_heads = nn.ModuleDict(
            {
                domain: nn.Linear(hidden_size, len(self.area_names_by_domain[domain]))
                for domain in self.domain_names
            }
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        return self.dropout(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        area_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.encode(input_ids, attention_mask)
        domain_logits = self.domain_classifier(features)

        output = {
            "domain_logits": domain_logits,
        }

        if domain_labels is not None and area_labels is not None:
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            area_loss = self._compute_area_loss(features, domain_labels, area_labels)
            output["loss"] = domain_loss + area_loss
            output["domain_loss"] = domain_loss.detach()
            output["area_loss"] = area_loss.detach()

        return output

    def _compute_area_loss(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
        area_labels: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = features.new_tensor(0.0)
        total_count = 0

        for domain_idx, domain_name in enumerate(self.domain_names):
            mask = domain_labels == domain_idx
            if mask.any():
                logits = self.area_heads[domain_name](features[mask])
                loss = F.cross_entropy(logits, area_labels[mask], reduction="sum")
                total_loss = total_loss + loss
                total_count += int(mask.sum().item())

        if total_count == 0:
            return total_loss

        return total_loss / total_count

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict domain and area.
        """
        self.eval()
        features = self.encode(input_ids, attention_mask)
        domain_logits = self.domain_classifier(features)
        domain_pred = domain_logits.argmax(dim=-1)

        area_pred = torch.empty_like(domain_pred)

        for domain_idx, domain_name in enumerate(self.domain_names):
            mask = domain_pred == domain_idx
            if mask.any():
                logits = self.area_heads[domain_name](features[mask])
                area_pred[mask] = logits.argmax(dim=-1)

        return domain_pred, area_pred