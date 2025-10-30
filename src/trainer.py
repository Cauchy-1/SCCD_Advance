# python
# 文件：src/trainer.py —— 修改 evaluate，新增 macro/weighted 与逐类指标
from typing import Dict, List
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Trainer:
    def __init__(self, model, optimizer, device, scheduler=None, class_weights=None, clip_grad_norm: float = 0.0, label_smoothing: float = 0.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.clip_grad_norm = float(clip_grad_norm or 0.0)
        try:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing or 0.0))
        except TypeError:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def train_one_epoch(self, epoch: int, total_epochs: int, train_loader) -> float:
        self.model.train()
        running_loss, n_batches = 0.0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)
            logits = self.model(texts)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()
            running_loss += loss.item(); n_batches += 1
        return running_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        for texts, labels in data_loader:
            texts = texts.to(self.device)
            logits = self.model(texts)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)

        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='micro', zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        # 逐类 F1 可用于定位不平衡类表现
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        return {
            'accuracy': float(acc),
            'precision_micro': float(p_micro),
            'recall_micro': float(r_micro),
            'f1_micro': float(f1_micro),
            'precision_macro': float(p_macro),
            'recall_macro': float(r_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(p_w),
            'recall_weighted': float(r_w),
            'f1_weighted': float(f1_w),
            # 可选择性打印/记录
            'f1_per_class_mean': float(f1_per_class.mean()),
        }
