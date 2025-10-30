import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


class MultiTaskTrainer:
    def __init__(self, model, train_loader, eval_loader, learning_rate, epochs, device, loss_weights, logger):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.loss_weights = loss_weights
        self.logger = logger
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-6)
        self.task_names = list(self.model.task_num_labels.keys())

    def train(self):
        for epoch in range(self.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")
            avg_train_loss = self._train_epoch()
            self.logger.info(f"Epoch {epoch + 1} 平均训练损失: {avg_train_loss:.4f}")
            self.evaluate(epoch)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}

        for i, batch in enumerate(tqdm(self.train_loader, desc="训练中")):
            self.optimizer.zero_grad()
            batch_total_loss = 0

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            for task_name in self.task_names:
                labels = batch[task_name].to(self.device)

                # [修复] 检查批次中的所有标签是否都为-100
                if (labels == -100).all():
                    continue  # 如果是，则跳过此任务的当前批次

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_name=task_name,
                    labels=labels
                )
                loss = outputs.loss

                if torch.isnan(loss):
                    self.logger.warning(f"批次 {i}, 任务 '{task_name}' 的损失为 NaN，已跳过。")
                    continue

                batch_total_loss += loss * self.loss_weights.get(task_name, 1.0)

                # 只为有效标签（非-100）计算指标
                valid_indices = labels != -100
                if valid_indices.any():
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_preds[task_name].extend(preds[valid_indices].cpu().numpy())
                    all_labels[task_name].extend(labels[valid_indices].cpu().numpy())

            if isinstance(batch_total_loss, torch.Tensor) and batch_total_loss > 0:
                batch_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += batch_total_loss.item()

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        self._log_metrics(all_preds, all_labels, "训练")
        return avg_loss

    def evaluate(self, epoch):
        self.model.eval()
        total_eval_loss = 0
        all_preds = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc=f"Epoch {epoch + 1} 评估中"):
                batch_total_loss = 0
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                for task_name in self.task_names:
                    labels = batch[task_name].to(self.device)

                    # [修复] 检查批次中的所有标签是否都为-100
                    if (labels == -100).all():
                        continue # 如果是，则跳过此任务的当前批次

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_name=task_name,
                        labels=labels
                    )
                    loss = outputs.loss
                    if torch.isnan(loss):
                        continue

                    batch_total_loss += loss * self.loss_weights.get(task_name, 1.0)

                    # 只为有效标签（非-100）计算指标
                    valid_indices = labels != -100
                    if valid_indices.any():
                        preds = torch.argmax(outputs.logits, dim=-1)
                        all_preds[task_name].extend(preds[valid_indices].cpu().numpy())
                        all_labels[task_name].extend(labels[valid_indices].cpu().numpy())

                if isinstance(batch_total_loss, torch.Tensor):
                    total_eval_loss += batch_total_loss.item()

        avg_eval_loss = total_eval_loss / len(self.eval_loader) if len(self.eval_loader) > 0 else 0
        self.logger.info(f"Epoch {epoch + 1} 平均评估损失: {avg_eval_loss:.4f}")
        self._log_metrics(all_preds, all_labels, "评估")

    def _log_metrics(self, all_preds, all_labels, stage):
        for task_name in self.task_names:
            labels = all_labels[task_name]
            preds = all_preds[task_name]
            if not labels:
                self.logger.info(f"{stage}任务 '{task_name}' - 没有可评估的样本。")
                continue

            precision = precision_score(labels, preds, average='micro', zero_division=0)
            recall = recall_score(labels, preds, average='micro', zero_division=0)
            f1 = f1_score(labels, preds, average='micro', zero_division=0)

            self.logger.info(f"{stage}任务 '{task_name}' - Precision (Micro): {precision:.4f}")
            self.logger.info(f"{stage}任务 '{task_name}' - Recall (Micro): {recall:.4f}")
            self.logger.info(f"{stage}任务 '{task_name}' - F1 (Micro): {f1:.4f}")
