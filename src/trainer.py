# src/trainer.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import torchmetrics


class Trainer:
    def __init__(self, model, train_dataset, eval_dataset, learning_rate=2e-5, epochs=10, batch_size=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.batch_size)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=total_steps)

        # 训练指标
        self.train_accuracy = torchmetrics.Accuracy(task="binary").to(self.device)

        # --- 关键修改：为评估设置更全面的指标 ---
        self.eval_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="binary"),
            'precision': torchmetrics.Precision(task="binary"),
            'recall': torchmetrics.Recall(task="binary"),
            'f1_micro': torchmetrics.F1Score(task="binary", average='micro')
        }).to(self.device)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.train_accuracy.reset()

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs} [Training]")
        for batch in progress_bar:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            self.train_accuracy.update(preds, labels)
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(self.train_dataloader)
        final_train_acc = self.train_accuracy.compute().item()
        return avg_train_loss, final_train_acc

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        # --- 关键修改：重置评估指标集合 ---
        self.eval_metrics.reset()

        progress_bar = tqdm(self.eval_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs} [Evaluating]")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                # --- 关键修改：用评估指标集合更新状态 ---
                self.eval_metrics.update(preds, labels)

        avg_eval_loss = total_loss / len(self.eval_dataloader)
        # --- 关键修改：计算所有评估指标并返回 ---
        final_metrics = self.eval_metrics.compute()
        # 将Tensor转换为标量值
        final_metrics_scalar = {k: v.item() for k, v in final_metrics.items()}

        return avg_eval_loss, final_metrics_scalar
