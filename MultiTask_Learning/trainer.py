import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report
import logging
import numpy as np


class MultiTaskTrainer:
    def __init__(self, model, train_loader, eval_loader, learning_rate, epochs, device, logger=None):
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

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('default_trainer_logger')
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        self.task_names = list(self.model.task_num_labels.keys())

    def train(self):
        """完整的训练和评估循环"""
        for epoch in range(self.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")
            avg_train_loss = self._train_epoch(epoch)
            self.logger.info(f"Epoch {epoch + 1} | 平均训练损失: {avg_train_loss:.4f}")
            self._evaluate()

    def _train_epoch(self, epoch_num):
        """单个训练轮次"""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num + 1} Training", leave=False)
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {task: batch[task].to(self.device) for task in self.task_names}

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None and loss > 0:
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0

    def _evaluate(self):
        """在评估集上评估模型"""
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

            # [修复] 过滤掉标签为 -100 的样本，以避免 sklearn 报错
            true_labels = np.array(all_labels[task_name])
            predictions = np.array(all_preds[task_name])

            # 创建一个布尔掩码，只选择标签不是-100的样本
            valid_mask = true_labels != -100

            # 如果没有任何有效样本，则跳过
            if not np.any(valid_mask):
                self.logger.info(f"任务: {task_name}\n无有效标签可供评估。")
                continue

            # 使用掩码来过滤标签和预测
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
