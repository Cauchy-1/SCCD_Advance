# python
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import PreTrainedTokenizer


def setup_logging(task_name: str, output_dir: str):
    """配置日志记录，同时输出到文件和控制台"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"{task_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # 返回一个logger实例
    return logging.getLogger(task_name)


def load_and_prepare_dataset(
        train_file: str,
        test_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128
) -> (Dataset, Dataset):
    """加载、预处理并分词数据集"""
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    except FileNotFoundError as e:
        logging.error(f"数据文件未找到: {e}")
        sys.exit(1)

    # 简单清洗和标签转换
    for df in [train_df, test_df]:
        df.dropna(subset=['comment_content', 'label'], inplace=True)
        # 将 'CB' 设为 1 (正类), 'Non-CB' 设为 0 (负类)
        df['label'] = df['label'].apply(lambda x: 1 if x == 'CB' else 0)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_function(examples):
        return tokenizer(
            examples["comment_content"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    计算并返回一个包含 Precision, Recall, 和 Micro F1 的字典。
    这与论文 Table 6 的指标对齐。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 论文中的 Precision 和 Recall 更可能是在不平衡数据集上更具信息量的 macro average
    # 或者针对正类的 binary average。这里我们使用 macro。
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    # 论文中的 Micro F1 在二分类中等同于 Accuracy
    micro_f1 = accuracy_score(labels, predictions)

    return {
        "accuracy (micro_f1)": micro_f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }