import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report
import logging
import numpy as np
from typing import Dict


class MultiTaskTrainer:
    def __init__(self,
                 model,
                 train_loader,
                 eval_loader,
                 learning_rate,
                 epochs,
                 device,
                 loss_weights: Dict[str, float],
                 logger=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )
        self.epochs = epochs
        self.device = device
        self.loss_weights = loss_weights  # [新增] 损失权重
        self.logger = logger or logging.getLogger('default_trainer_logger')
        self.task_names = list(self.model.task_num_labels.keys())

    def train(self):
        for epoch in range(self.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")
            avg_train_loss = self._train_epoch(epoch)
            self.logger.info(f"Epoch {epoch + 1} | 平均加权训练损失: {avg_train_loss:.4f}")
            self._evaluate()

    def _train_epoch(self, epoch_num):
        self.model.train()
        total_weighted_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num + 1} Training", leave=False)
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {task: batch[task].to(self.device) for task in self.task_names}

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # [修改] 实现加权损失计算
            if outputs.loss and isinstance(outputs.loss, dict):
                weighted_loss = 0
                for task_name, loss in outputs.loss.items():
                    weight = self.loss_weights.get(task_name, 1.0)
                    weighted_loss += weight * loss

                if weighted_loss > 0:
                    total_weighted_loss += weighted_loss.item()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    progress_bar.set_postfix({'loss': f'{weighted_loss.item():.4f}'})

        return total_weighted_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0

    def _evaluate(self):
        self.model.eval()
        all_preds = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

                for task_name in self.task_names:
                    preds = torch.argmax(logits[task_name], dim=1)
                    all_preds[task_name].extend(preds.cpu().numpy())
                    all_labels[task_name].extend(batch[task_name].cpu().numpy())

        self.logger.info("\n--- 评估结果 ---")
        for task_name in self.task_names:
            true_labels = np.array(all_labels[task_name])
            predictions = np.array(all_preds[task_name])

            valid_mask = true_labels != -100

            if not np.any(valid_mask):
                self.logger.info(f"任务: {task_name}\n无有效标签可供评估。")
                continue

            filtered_labels = true_labels[valid_mask]
            filtered_preds = predictions[valid_mask]

            report = classification_report(
                filtered_labels,
                filtered_preds,
                target_names=self.train_loader.dataset.get_labels_for_task(task_name),
                zero_division=0
            )
            self.logger.info(f"任务: {task_name}\n{report}")
        self.logger.info("---------------------\n")
