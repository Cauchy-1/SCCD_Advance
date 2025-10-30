# python
# 文件: src/run.py

import os
import random
import argparse
import numpy as np
import torch
import logging
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from model import CNNTextClassifier, LSTMTextClassifier
from dataset import SCCDDataset, generate_batch, load_pretrained_vectors
from trainer import Trainer


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, model_name: str):
    sanitized_model_name = model_name.replace('/', '_')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{sanitized_model_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()]
    )
    return log_filepath


def main():
    parser = argparse.ArgumentParser()
    # 数据与列名
    parser.add_argument('--data_file', type=str, default='datas/SCCD/comments.csv')
    parser.add_argument('--comment_col', type=str, default='comment_content')
    parser.add_argument('--label_col', type=str, default='label')

    # 模型与训练
    parser.add_argument('--arch', type=str, default='lstm', choices=['cnn', 'lstm'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--vector_path', type=str, default='datas/light_Tencent_AILab_ChineseEmbedding.bin')
    parser.add_argument('--freeze_embeddings', action='store_true')
    # 训练细节增强
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)

    # LSTM 特定
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false')

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    proj_root = base_dir.parent

    def resolve_path(p: str) -> str:
        path_obj = Path(p)
        return str(proj_root / path_obj) if not path_obj.is_absolute() else str(path_obj)

    args.data_file = resolve_path(args.data_file)
    args.output_dir = resolve_path(args.output_dir)
    args.vector_path = resolve_path(args.vector_path)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None and args.seed >= 0:
        set_seed(args.seed)

    # 日志
    log_dir_name = f"{args.arch.upper()}logs"
    log_dir = os.path.join(args.output_dir, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = setup_logging(log_dir, args.arch)
    logging.info(f"日志将保存在: {log_filepath}")

    logging.info("\n============================================================")
    logging.info("Starting New Training Session")
    logging.info("------------------------------------------------------------")
    logging.info(f"Model Arch: {args.arch.upper()}")
    logging.info("Hyperparameters:")
    logging.info(f"  - Epochs: {args.epochs}")
    logging.info(f"  - Batch Size: {args.batch_size}")
    logging.info(f"  - Learning Rate: {args.lr}")
    logging.info(f"  - Embedding Dim: {args.embed_dim}")
    logging.info(f"  - Pretrained Vectors: {args.vector_path if os.path.exists(args.vector_path) else 'None'}")
    logging.info(f"  - Freeze Embeddings: {args.freeze_embeddings}")
    if args.arch == 'lstm':
        logging.info(f"  - LSTM Hidden Dim: {args.hidden_dim}")
        logging.info(f"  - LSTM Layers: {args.n_layers}")
        logging.info(f"  - Bidirectional: {args.bidirectional}")
    logging.info("============================================================\n")

    # 数据集
    logging.info("Loading datasets and building vocab...")
    full_dataset = SCCDDataset(csv_file=args.data_file, comment_col=args.comment_col, label_col=args.label_col)
    if len(full_dataset) == 0:
        logging.error("数据清洗后，没有可用的数据样本。请检查CSV与数据清洗逻辑。")
        return

    vocab = full_dataset.vocab
    pad_idx = vocab.get('<pad>', 1)
    num_labels = full_dataset.num_labels
    logging.info(f"Vocabulary size: {len(vocab)}")

    # 分层划分，保证类别分布一致
    indices = np.arange(len(full_dataset))
    labels_for_split = full_dataset.get_all_label_ids()
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels_for_split,
        shuffle=True,
        random_state=(args.seed if args.seed is not None and args.seed >= 0 else None)
    )
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    logging.info(f"Dataset loaded and split into {len(train_ds)} training samples and {len(val_ds)} validation samples.")

    # collate
    collate_fn = partial(generate_batch, pad_idx=pad_idx, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    # 模型
    logging.info("Initializing model...")
    if args.arch.lower() == 'cnn':
        model = CNNTextClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            num_labels=num_labels,
            dropout_rate=args.dropout_rate
        )
    else:
        model = LSTMTextClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_labels=num_labels,
            n_layers=args.n_layers,
            bidirectional=args.bidirectional,
            dropout_rate=args.dropout_rate
        )

    # 预训练向量
    if os.path.exists(args.vector_path):
        logging.info(f"正在从 {args.vector_path} 加载预训练词向量...")
        pretrained_matrix = load_pretrained_vectors(args.vector_path, vocab, args.embed_dim)
        if pretrained_matrix is not None:
            model.embedding.weight.data.copy_(pretrained_matrix)
            logging.info("成功加载预训练词向量。")
            if args.freeze_embeddings:
                model.embedding.weight.requires_grad = False
                logging.info("Embedding layer is frozen.")
        else:
            logging.info("预训练向量维度不匹配，跳过加载。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 类别权重
    logging.info("Calculating class weights for unbalanced dataset...")
    all_label_ids = full_dataset.get_all_label_ids()
    classes = np.arange(num_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_label_ids)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    logging.info(f"Calculated class weights: {class_weights.detach().cpu().numpy()}")

    # 训练器
    logging.info("Initializing trainer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    trainer = Trainer(
        model, optimizer, device,
        scheduler=scheduler,
        class_weights=class_weights,
        clip_grad_norm=args.clip_grad_norm,
        label_smoothing=args.label_smoothing
    )

    # 训练循环
    logging.info("Starting training...")
    best_f1 = -1.0
    best_accuracy = -1.0
    best_precision = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_one_epoch(epoch, args.epochs, train_loader)
        metrics = trainer.evaluate(val_loader)
        # 使用 f1_macro 作为核心评估和早停指标，这在不平衡分类中更鲁棒
        f1_macro = metrics['f1_macro']

        if trainer.scheduler is not None:
            # 学习率调度器也应该监控核心指标
            trainer.scheduler.step(f1_macro)

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_accuracy = metrics['accuracy']
            # 您也可以保存对应的 macro precision 和 recall
            best_precision_macro = metrics['precision_macro']
            best_recall_macro = metrics['recall_macro']
            epochs_no_improve = 0
            save_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{args.arch}_best_model.bin')
            torch.save(model.state_dict(), save_path)
            logging.info(f"\n🎉 New best F1_macro score: {f1_macro:.4f}. Saving model...")
        else:
            epochs_no_improve += 1

        logging.info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
            f"Eval Precision_macro={metrics['precision_macro']:.4f}, "
            f"Eval Recall_macro={metrics['recall_macro']:.4f}, "
            f"Eval F1_macro={metrics['f1_macro']:.4f}, "
            f"Eval Accuracy(Micro F1)={metrics['accuracy']:.4f}"  # Micro F1 等于 Accuracy
        )

        if epochs_no_improve >= args.patience:
            logging.info(f"\nEarly stopping triggered after {args.patience} epochs with no improvement.")
            break

    logging.info("\nTraining finished.")
    logging.info(f"Best F1_macro score achieved: {best_f1:.4f}")
    # 可以打印在获得最佳f1_macro时对应的其他指标
    logging.info(f"Corresponding Accuracy: {best_accuracy:.4f}")
    logging.info(f"Corresponding Precision_macro: {best_precision_macro:.4f}")
    logging.info(f"Corresponding Recall_macro: {best_recall_macro:.4f}")


if __name__ == '__main__':
    main()
