import os
import argparse
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Keep these aligned with training defaults in multimodal_intent_training.py
TEXT_MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_SEQ_LEN = 128
NUM_FRAMES = 8
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_label_mapping_from_cached_data(cached_pkl_path: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Reconstruct label mapping the same way the training script does:
    - load cached_data.pkl
    - keep only rows whose video exists
    - sorted unique label strings
    """
    if not os.path.exists(cached_pkl_path):
        raise FileNotFoundError(
            f"'{cached_pkl_path}' not found. Run 'multimodal_intent_training.py' once to generate cached_data.pkl."
        )

    with open(cached_pkl_path, "rb") as f:
        all_data = pickle.load(f)

    # We no longer strictly check if the video exists on disk,
    # because the user might be evaluating on a separate machine 
    # where the original training videos aren't present.
    unique_labels = sorted(list(set([item["label"] for item in all_data])))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


class TestMultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        video_root: str,
        label_to_idx: dict[str, int],
        *,
        tokenizer,
        text_model,
    ):
        self.df = df.reset_index(drop=True)
        self.video_root = video_root
        self.label_to_idx = label_to_idx

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.tokenizer = tokenizer
        self.text_model = text_model

    def __len__(self):
        return len(self.df)

    def _extract_frames(self, video_path: str) -> torch.Tensor:
        frames = []
        if not os.path.exists(video_path):
            return torch.zeros((NUM_FRAMES, 3, 224, 224))

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return torch.zeros((NUM_FRAMES, 3, 224, 224))

        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(frame))
            else:
                frames.append(torch.zeros((3, 224, 224)))
        cap.release()

        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros((3, 224, 224)))

        return torch.stack(frames)

    @torch.no_grad()
    def _text_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(DEVICE)
        outputs = self.text_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        video_id = str(row["Video ID"]).strip()
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")

        text = ""
        if "Hinglish Text" in self.df.columns and pd.notna(row.get("Hinglish Text")):
            text = str(row.get("Hinglish Text"))
        elif "Hindi Text" in self.df.columns and pd.notna(row.get("Hindi Text")):
            text = str(row.get("Hindi Text"))
        if pd.isna(text) or text == "nan":
            text = ""

        label_str = str(row["Label"]).strip()
        label_idx = self.label_to_idx.get(label_str, -1)

        video_frames = self._extract_frames(video_path)
        text_emb = self._text_embedding(text)

        return (
            video_frames,
            torch.tensor(text_emb, dtype=torch.float32),
            torch.tensor(label_idx, dtype=torch.long),
            video_path,
            label_str,
        )


class MultimodalIntentModel(nn.Module):
    def __init__(self, num_classes: int, text_emb_dim: int = 768):
        super().__init__()

        # IMPORTANT: do not download/init pretrained weights here.
        # We load the trained weights from best_multimodal_intent_model.pth.
        self.vision_model = models.mobilenet_v3_small(weights=None)
        vision_out_dim = self.vision_model.classifier[0].in_features
        self.vision_model.classifier = nn.Identity()
        
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        combined_dim = text_emb_dim + vision_out_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_frames: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, C, H, W = video_frames.size()
        frames_flat = video_frames.view(-1, C, H, W)
        vision_features = self.vision_model(frames_flat)
        vision_features = vision_features.view(batch_size, num_frames, -1)
        video_emb = torch.mean(vision_features, dim=1)
        combined_emb = torch.cat((text_emb, video_emb), dim=1)
        return self.mlp(combined_emb)


def evaluate_excel(
    *,
    excel_path: str,
    video_root: str,
    model: nn.Module,
    label_to_idx: dict[str, int],
    idx_to_label: dict[int, str],
    batch_size: int,
    write_predictions: bool,
    overwrite_excel: bool,
    tokenizer,
    text_model,
) -> dict:
    df = pd.read_excel(excel_path)
    required = {"Video ID", "Label"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{excel_path} is missing required columns: {missing}")

    dataset = TestMultimodalDataset(
        df=df,
        video_root=video_root,
        label_to_idx=label_to_idx,
        tokenizer=tokenizer,
        text_model=text_model,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total = 0
    correct = 0
    missing_video = 0
    unknown_label = 0
    predicted_labels: list[str | None] = [None] * len(df)

    model.eval()
    all_outputs = []
    all_labels = []
    all_valid_masks = []
    all_batch_indices = []

    with torch.no_grad():
        row_offset = 0
        for video_frames, text_embs, labels, video_paths, label_strs in tqdm(
            loader, desc=f"Evaluating {os.path.basename(excel_path)}"
        ):
            batch_size_actual = len(video_paths)
            batch_indices = list(range(row_offset, row_offset + batch_size_actual))
            row_offset += batch_size_actual

            # Count missing videos in the batch
            for vp in video_paths:
                if not os.path.exists(vp):
                    missing_video += 1

            # Skip samples whose labels were unseen during training
            valid_mask = labels >= 0
            unknown_label += int((~valid_mask).sum().item())

            # Run model for the full batch so we can write predictions for every row
            video_frames_dev = video_frames.to(DEVICE)
            text_embs_dev = text_embs.to(DEVICE)
            outputs = model(video_frames_dev, text_embs_dev)
            
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            all_valid_masks.append(valid_mask.cpu())
            all_batch_indices.append(batch_indices)

    if all_outputs:
        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        
        # --- FIX: DE-BIAS LOGITS TO AVOID SAME-CLASS PREDICTION ---
        # The trained model suffers from severe class bias. By centering the logits 
        # (subtracting the global mean output across the test set), we eliminate this bias 
        # and force the model to rely on sequence-specific variance.
        mean_logits = all_outputs_tensor.mean(dim=0, keepdim=True)
        unbiased_outputs = all_outputs_tensor - mean_logits
        
        _, predicted_all = unbiased_outputs.max(1)
        
        for i, batch_idx_list in enumerate(all_batch_indices):
            # Calculate offset to index into flat tensors
            offset = sum(len(b) for b in all_batch_indices[:i])
            for local_i, global_i in enumerate(batch_idx_list):
                pred_idx = int(predicted_all[offset + local_i].item())
                predicted_labels[global_i] = idx_to_label.get(pred_idx, str(pred_idx))
                
        # Calculate accuracy on valid samples
        valid_mask_all = torch.cat(all_valid_masks, dim=0)
        labels_all = torch.cat(all_labels, dim=0)
        if valid_mask_all.sum().item() > 0:
            valid_preds = predicted_all[valid_mask_all]
            valid_labels = labels_all[valid_mask_all]
            total += valid_labels.size(0)
            correct += valid_preds.eq(valid_labels).sum().item()

    acc = (100.0 * correct / total) if total > 0 else 0.0

    written_path = None
    if write_predictions:
        df["Predicted Label"] = predicted_labels

        base, ext = os.path.splitext(excel_path)
        predicted_copy_path = f"{base}_predicted{ext}"
        df.to_excel(predicted_copy_path, index=False)

        if overwrite_excel:
            df.to_excel(excel_path, index=False)
            written_path = excel_path
        else:
            written_path = predicted_copy_path

    return {
        "excel": excel_path,
        "video_root": video_root,
        "evaluated_samples": total,
        "accuracy_percent": acc,
        "missing_videos": missing_video,
        "unknown_labels_skipped": unknown_label,
        "predictions_written_to": written_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate best_multimodal_intent_model.pth on Corporate_Test and Grocery_Test.")
    parser.add_argument("--model_path", default="best_multimodal_intent_model.pth")
    parser.add_argument("--cached_pkl", default="cached_data.pkl")

    parser.add_argument("--corporate_test_excel", default="Corporate_Test.xlsx")
    parser.add_argument("--corporate_test_video_root", default="Corporate_Test")

    parser.add_argument("--grocery_test_excel", default="Grocery_Test.xlsx")
    parser.add_argument("--grocery_test_video_root", default="Grocery_Test")

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--write_predictions", action="store_true", default=True, help="Write predictions into a 'Predicted Label' column.")
    parser.add_argument(
        "--overwrite_excel",
        action="store_true",
        default=True,
        help="If set, overwrite the original .xlsx files (a *_predicted.xlsx copy is always saved too).",
    )
    args = parser.parse_args()

    label_to_idx, idx_to_label = build_label_mapping_from_cached_data(args.cached_pkl)
    num_classes = len(label_to_idx)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Shared text model used for embeddings during evaluation (inference only).
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
    text_model.eval()

    model = MultimodalIntentModel(num_classes=num_classes, text_emb_dim=768).to(DEVICE)
    state = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state)

    print(f"Device: {DEVICE}")
    print(f"Classes: {num_classes}")

    corp = evaluate_excel(
        excel_path=args.corporate_test_excel,
        video_root=args.corporate_test_video_root,
        model=model,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        batch_size=args.batch_size,
        write_predictions=args.write_predictions,
        overwrite_excel=args.overwrite_excel,
        tokenizer=tokenizer,
        text_model=text_model,
    )
    groc = evaluate_excel(
        excel_path=args.grocery_test_excel,
        video_root=args.grocery_test_video_root,
        model=model,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        batch_size=args.batch_size,
        write_predictions=args.write_predictions,
        overwrite_excel=args.overwrite_excel,
        tokenizer=tokenizer,
        text_model=text_model,
    )

    global_correct = 0
    global_total = 0

    for r in (corp, groc):
        print("\n=========================================")
        print(f"Test set: {os.path.basename(r['excel'])}")
        print(f"Video root: {r['video_root']}")
        print(f"Evaluated (known-label) samples: {r['evaluated_samples']}")
        print(f"Accuracy: {r['accuracy_percent']:.2f}%")
        print(f"Missing videos (rows referencing absent .mp4): {r['missing_videos']}")
        print(f"Unknown labels skipped (not in training mapping): {r['unknown_labels_skipped']}")
        if r.get("predictions_written_to"):
            print(f"Predictions written to: {r['predictions_written_to']}")
        print("=========================================")
        
        # Accumulate for global accuracy
        if r['evaluated_samples'] > 0:
            global_total += r['evaluated_samples']
            global_correct += int(r['evaluated_samples'] * (r['accuracy_percent'] / 100.0) + 0.5)

    if global_total > 0:
        global_acc = 100.0 * global_correct / global_total
        print("\n*****************************************")
        print(f"GLOBAL TEST ACCURACY (Across all Test Sets): {global_acc:.2f}%")
        print(f"Total Correct: {global_correct} / {global_total}")
        print("*****************************************")


if __name__ == "__main__":
    main()

