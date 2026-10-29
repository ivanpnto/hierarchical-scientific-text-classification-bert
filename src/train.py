import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import List

from .utils import set_seed, save_json, compute_metrics, HierarchicalLabelEncoder
from .data import prepare_data
from .model import HierarchicalBertClassifier

def make_loader(dataset, collator, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )

def evaluate(
    model: HierarchicalBertClassifier,
    loader: DataLoader,
    encoder,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    true_domain_ids: List[int] = []
    pred_domain_ids: List[int] = []
    true_area_global: List[str] = []
    pred_area_global: List[str] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_labels = batch["domain_labels"].to(device)
        area_labels = batch["area_labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_labels=domain_labels,
            area_labels=area_labels,
        )

        loss = outputs["loss"]
        batch_size = input_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        domain_pred, area_pred = model.predict(input_ids, attention_mask)

        true_domain_ids.extend(domain_labels.cpu().tolist())
        pred_domain_ids.extend(domain_pred.cpu().tolist())

        for td, ta, pd, pa in zip(
            domain_labels.cpu().tolist(),
            area_labels.cpu().tolist(),
            domain_pred.cpu().tolist(),
            area_pred.cpu().tolist(),
        ):
            true_domain_name = encoder.decode_domain(td)
            pred_domain_name = encoder.decode_domain(pd)

            true_area_name = encoder.decode_area(true_domain_name, ta)
            pred_area_name = encoder.decode_area(pred_domain_name, pa)

            true_area_global.append(true_area_name)
            pred_area_global.append(pred_area_name)

    metrics_domain = compute_metrics(
        true_domain_ids,
        pred_domain_ids,
        labels=encoder.id2domain,
    )
    area_labels_sorted = sorted(set(true_area_global + pred_area_global))
    area2id = {label: idx for idx, label in enumerate(area_labels_sorted)}
    true_area_ids = [area2id[a] for a in true_area_global]
    pred_area_ids = [area2id[a] for a in pred_area_global]
    metrics_area = compute_metrics(true_area_ids, pred_area_ids, labels=area_labels_sorted)

    hierarchical_exact = sum(
        int((td == pd) and (ta == pa))
        for td, pd, ta, pa in zip(true_domain_ids, pred_domain_ids, true_area_global, pred_area_global)
    ) / max(1, len(true_domain_ids))

    return {
        "loss": total_loss / max(1, total_samples),
        "domain": metrics_domain,
        "area": metrics_area,
        "hierarchical_exact_match": hierarchical_exact,
        "true_domain_ids": true_domain_ids,
        "pred_domain_ids": pred_domain_ids,
        "true_area_global": true_area_global,
        "pred_area_global": pred_area_global,
    }

def train_one_epoch(model, loader, optimizer, scheduler, device, use_amp: bool):
    model.train()
    total_loss = 0.0
    total_samples = 0

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_labels = batch["domain_labels"].to(device)
        area_labels = batch["area_labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                domain_labels=domain_labels,
                area_labels=area_labels,
            )
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = input_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenized, tokenizer, encoder, collator = prepare_data(
        model_name=args.model_name,
        max_length=args.max_length,
    )

    train_ds = tokenized["train"]
    val_ds = tokenized["validate"]
    test_ds = tokenized["test"]

    train_loader = make_loader(train_ds, collator, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, collator, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, collator, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = HierarchicalBertClassifier(
        domain_names=encoder.id2domain,
        area_names_by_domain=encoder.id2area_by_domain,
        model_name=args.model_name,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_path = output_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, use_amp)
        val_result = evaluate(model, val_loader, encoder, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result['loss']:.4f} | "
            f"val_domain_acc={val_result['domain']['accuracy']:.4f} | "
            f"val_area_acc={val_result['area']['accuracy']:.4f}"
        )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            torch.save(model.state_dict(), best_path)
            save_json(output_dir / "label_encoder.json", encoder.to_dict())

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_result = evaluate(model, test_loader, encoder, device)

    save_json(
        output_dir / "test_metrics.json",
        {
            "test_loss": test_result["loss"],
            "domain": test_result["domain"],
            "area": test_result["area"],
            "hierarchical_exact_match": test_result["hierarchical_exact_match"],
        },
    )

    print("Final test metrics saved to", output_dir / "test_metrics.json")

if __name__ == "__main__":
    main()
