import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .utils import load_json, save_json, plot_confusion_matrix, HierarchicalLabelEncoder
from .data import prepare_data
from .model import HierarchicalBertClassifier
from .train import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    label_data = load_json(output_dir / "label_encoder.json")
    encoder = HierarchicalLabelEncoder.from_dict(label_data)

    tokenized, tokenizer, _, collator = prepare_data(
        model_name=args.model_name,
        max_length=args.max_length,
    )

    test_ds = tokenized["test"]
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalBertClassifier(
        domain_names=encoder.id2domain,
        area_names_by_domain=encoder.id2area_by_domain,
        model_name=args.model_name,
    ).to(device)
    model.load_state_dict(torch.load(output_dir / "model.pt", map_location=device))

    result = evaluate(model, test_loader, encoder, device)

    save_json(
        output_dir / "test_metrics.json",
        {
            "test_loss": result["loss"],
            "domain": result["domain"],
            "area": result["area"],
            "hierarchical_exact_match": result["hierarchical_exact_match"],
        },
    )

    plot_confusion_matrix(
        result["true_domain_ids"],
        result["pred_domain_ids"],
        labels=encoder.id2domain,
        title="Domain confusion matrix",
        output_path=output_dir / "confusion_domain.png",
    )

    area_labels = sorted(set(result["true_area_global"] + result["pred_area_global"]))
    area2id = {label: idx for idx, label in enumerate(area_labels)}
    true_area_ids = [area2id[a] for a in result["true_area_global"]]
    pred_area_ids = [area2id[a] for a in result["pred_area_global"]]

    plot_confusion_matrix(
        true_area_ids,
        pred_area_ids,
        labels=area_labels,
        title="Area confusion matrix",
        output_path=output_dir / "confusion_area.png",
    )

    pred_file = output_dir / "predictions.csv"
    with pred_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_domain", "pred_domain", "true_area", "pred_area"])
        for td, pd, ta, pa in zip(
            result["true_domain_ids"],
            result["pred_domain_ids"],
            result["true_area_global"],
            result["pred_area_global"],
        ):
            writer.writerow([encoder.decode_domain(td), encoder.decode_domain(pd), ta, pa])

    print("Saved metrics to", output_dir / "test_metrics.json")
    print("Saved confusion matrices to", output_dir / "confusion_domain.png", "and", output_dir / "confusion_area.png")
    print("Saved predictions to", pred_file)

if __name__ == "__main__":
    main()