from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from typing import Dict, List
from .utils import HierarchicalLabelEncoder

DATASET_NAME = "river-martin/web-of-science-with-label-texts"

def load_raw_dataset() -> DatasetDict:
    """Load the dataset from Hugging Face."""
    return load_dataset(DATASET_NAME)

def build_label_encoder(dataset: DatasetDict) -> HierarchicalLabelEncoder:
    """Build a hierarchy-aware encoder."""
    all_splits = concatenate_datasets([dataset[split] for split in dataset.keys()])
    domains = sorted(set(all_splits["domain"]))
    domain2id = {domain: idx for idx, domain in enumerate(domains)}
    id2domain = domains

    area2id_by_domain: Dict[str, Dict[str, int]] = {}
    id2area_by_domain: Dict[str, List[str]] = {}

    for domain in domains:
        domain_rows = all_splits.filter(lambda ex: ex["domain"] == domain)
        areas = sorted(set(domain_rows["area"]))
        area2id_by_domain[domain] = {area: idx for idx, area in enumerate(areas)}
        id2area_by_domain[domain] = areas

    return HierarchicalLabelEncoder(
        domain2id=domain2id,
        id2domain=id2domain,
        area2id_by_domain=area2id_by_domain,
        id2area_by_domain=id2area_by_domain,
    )

def tokenize_and_encode(
    dataset: DatasetDict,
    encoder: HierarchicalLabelEncoder,
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
):
    """Tokenize abstracts and convert string labels to integer ids."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _encode_batch(batch):
        tokenized = tokenizer(batch["abstract"], truncation=True, max_length=max_length)
        tokenized["domain_id"] = [encoder.encode_domain(d) for d in batch["domain"]]
        tokenized["area_id"] = [encoder.encode_area(d, a) for d, a in zip(batch["domain"], batch["area"])]
        return tokenized

    remove_columns = list(dataset["train"].column_names)
    tokenized = dataset.map(_encode_batch, batched=True, remove_columns=remove_columns)
    return tokenized, tokenizer

def build_data_collator(tokenizer):
    """Pad batches dynamically and keep scalar labels intact."""
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def _collate(features):
        import torch

        domain_ids = [f.pop("domain_id") for f in features]
        area_ids = [f.pop("area_id") for f in features]
        batch = collator(features)
        batch["domain_labels"] = torch.tensor(domain_ids, dtype=torch.long)
        batch["area_labels"] = torch.tensor(area_ids, dtype=torch.long)
        return batch

    return _collate

def prepare_data(
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
):
    """Convenience function used by train/evaluate scripts."""
    raw = load_raw_dataset()
    encoder = build_label_encoder(raw)
    tokenized, tokenizer = tokenize_and_encode(raw, encoder, model_name=model_name, max_length=max_length)
    collator = build_data_collator(tokenizer)
    return tokenized, tokenizer, encoder, collator