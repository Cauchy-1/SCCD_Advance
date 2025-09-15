# src/run.py

import argparse
import logging
import os
from datetime import datetime
from dataset import SCCDDataset
from model import CyberbullyingClassifier
from trainer import Trainer


def main():
    log_dir = '../outputs/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="Train Cyberbullying Detection Model based on SCCD Paper")
    # ... (参数部分保持不变)
    parser.add_argument('--data_path', type=str, default='../datas/SCCD', help='Path to the dataset directory')
    parser.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='Hugging Face model name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for AdamW optimizer')
    parser.add_argument('--top_k_comments', type=int, default=5, help='Number of top-liked comments to use')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for tokenizer')
    args = parser.parse_args()

    print("Loading datasets...")
    train_dataset = SCCDDataset(
        data_path=args.data_path, split='train', model_name=args.model_name,
        max_length=args.max_length, top_k_comments=args.top_k_comments
    )
    eval_dataset = SCCDDataset(
        data_path=args.data_path, split='test', model_name=args.model_name,
        max_length=args.max_length, top_k_comments=args.top_k_comments
    )

    print("Initializing model...")
    model = CyberbullyingClassifier(model_name=args.model_name)

    print("Initializing trainer...")
    trainer = Trainer(
        model=model, train_dataset=train_dataset, eval_dataset=eval_dataset,
        learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size
    )

    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train_one_epoch(epoch)

        # --- 关键修改：接收评估指标字典 ---
        eval_loss, eval_metrics = trainer.evaluate(epoch)

        # --- 关键修改：格式化并记录所有评估指标 ---
        eval_metrics_str = ", ".join([f"Eval {k.capitalize()}={v:.4f}" for k, v in eval_metrics.items()])
        log_msg = (f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                   f"Eval Loss={eval_loss:.4f}, {eval_metrics_str}")

        print(log_msg)
        logging.info(log_msg)
    print("Training finished.")


if __name__ == '__main__':
    main()
