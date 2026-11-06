if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from recsys_vnpt.domain_entrypoint import main

    raise SystemExit(main("movie", "train"))

import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time

from pathlib import Path
from processing import process_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from dcnv3 import DCNv3, TriBCE_Loss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from recsys_vnpt.feature_contract import FeatureContract
from recsys_vnpt.features import FeatureSchema
from recsys_vnpt.metrics_io import write_metrics_json
from recsys_vnpt.paths import domain_work_path, model_artifact_path, model_dir
from recsys_vnpt.timing import runtime_metrics

model_save_dir = model_dir("movie")
model_save_dir.mkdir(parents=True, exist_ok=True)

def print_class_distribution(y_data, name):
    total = len(y_data)
    class_counts = y_data.value_counts().to_dict()
    print(f"\n{name} Data - Total Samples: {total}")
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        percent = (count / total) * 100
        print(f"  Class {label}: {count} samples ({percent:.2f}%)")

def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, patience=3, scheduler=None):
    train_losses = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_scores = []

    model_path = model_artifact_path("movie", "best_model.pth")

    best_val_loss = float('inf')  
    if os.path.exists(model_path):
        previous_model = torch.load(model_path, map_location=device)
        best_previous_loss = previous_model['loss']
    else:
        print(f"No existing checkpoint at {model_path}; training from random initialization.")
        best_previous_loss = float('inf')
    early_stop_counter = 0       

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_y_true = []
        all_y_scores = []

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output['y_pred'], labels, output['y_d'], output['y_s'])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            all_y_true.extend(labels.detach().cpu().numpy())
            all_y_scores.extend(output['y_pred'].detach().cpu().numpy())

        all_y_true = np.array(all_y_true).flatten()
        all_y_scores = np.array(all_y_scores).flatten()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        y_pred = (all_y_scores > 0.5).astype(int)

        precision = precision_score(all_y_true, y_pred, zero_division=0)
        recall = recall_score(all_y_true, y_pred, zero_division=0)
        f1 = f1_score(all_y_true, y_pred, zero_division=0)
        auc = roc_auc_score(all_y_true, all_y_scores)
        accuracy = accuracy_score(all_y_true, y_pred)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

        print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}')
        print(f'\nAccuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'AUC: {auc:.4f}')

        val_loss = validate(model, val_loader, loss_fn, device)

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            if best_val_loss < best_previous_loss:
                save_checkpoint(model, epoch, optimizer, best_val_loss, domain="movie")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    all_y_true = []
    all_y_scores = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = loss_fn(output['y_pred'], labels, output['y_d'], output['y_s'])
            val_loss += loss.item()

            all_y_true.extend(labels.detach().cpu().numpy())
            all_y_scores.extend(output['y_pred'].detach().cpu().numpy())

    all_y_true = np.array(all_y_true).flatten()
    all_y_scores = np.array(all_y_scores).flatten()

    avg_val_loss = val_loss / len(val_loader)

    y_pred = (all_y_scores > 0.5).astype(int)

    precision = precision_score(all_y_true, y_pred, zero_division=0)
    recall = recall_score(all_y_true, y_pred, zero_division=0)
    f1 = f1_score(all_y_true, y_pred, zero_division=0)
    auc = roc_auc_score(all_y_true, all_y_scores)
    accuracy = accuracy_score(all_y_true, y_pred)

    print(f'\nValidation Loss: {avg_val_loss:.4f}')
    print(f'\nAccuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}\n')

    model.train()  
    return avg_val_loss  

def save_checkpoint(model, epoch, optimizer, loss, domain="movie"):
    model_path = model_artifact_path(domain, "best_model.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, model_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    project_root = Path().resolve()
    data_path = domain_work_path("movie", "train_data", "merged_user_item_duration.parquet", project_root=project_root)
    if os.path.exists(data_path):
        data = pd.read_parquet(data_path)
    else:
        data = process_data(data_path)

    metadata_cols = ["username", "content_id", "profile_id", "label_reason", "sample_type", "content_family"]
    train_data = data.drop(columns=[col for col in metadata_cols if col in data.columns])
    train_data = train_data.fillna(0)
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    train_data = train_data.dropna()
    y = train_data["label"]
    schema = FeatureSchema.fit(train_data)
    schema.save(model_artifact_path("movie", "feature_schema.json", project_root=project_root))
    model_config = {"input_dim": len(schema.feature_columns), "feature_columns": schema.feature_columns}
    feature_contract = FeatureContract.fit(
        domain="movie",
        frame=train_data,
        label_policy={"domain": "movie"},
        model_family="dense_mimic_baseline",
        model_config=model_config,
        numeric_columns=schema.feature_columns,
        categorical_columns=[],
        multivalue_columns=[],
    )
    feature_contract.save(model_artifact_path("movie", "feature_contract.json", project_root=project_root))
    X = schema.transform(train_data)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    print_class_distribution(pd.Series(y_train), "Training")
    print_class_distribution(pd.Series(y_val), "Validation")

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    input_dim = X_train_tensor.shape[1]
    model = DCNv3(input_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    loss_fn = TriBCE_Loss()

    start_time = time.time()
    for i in range(5):
        print(f"Training run no.{i+1}")
        epochs = 30
        patience = 4

        train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, patience=patience, scheduler=scheduler)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Elapsed time: {elapsed}')
    write_metrics_json(
        model_artifact_path("movie", "training_metrics.json", project_root=project_root),
        runtime_metrics(
            model_family="dense_mimic_baseline",
            training_time_seconds=elapsed,
        ),
    )
