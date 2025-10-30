import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import logging
from datetime import datetime

from data_loader import MultiTaskSCCDDataset
from model import MultiTaskBert
from trainer import MultiTaskTrainer

# --- 配置 ---
MODEL_NAME = "bert-base-chinese"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LENGTH = 128

# [新增] 任务筛选和损失加权配置
ACTIVE_TASKS = ['label', 'expression', 'target']
LOSS_WEIGHTS = {
    'label': 1.0,  # 主任务，权重最高
    'expression': 0.5,  # 辅助任务，权重较低
    'target': 0.5  # 辅助任务，权重较低
}

# -----------------

def setup_logging(log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    logger = logging.getLogger('improved_mtl_logger')
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    log_dir = current_dir / 'logs'
    logger = setup_logging(log_dir)

    train_file = project_root / 'datas' / 'SCCD' / 'train.csv'
    test_file = project_root / 'datas' / 'SCCD' / 'test.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"使用设备: {device}")
    logger.info(f"激活的任务: {ACTIVE_TASKS}")
    logger.info(f"任务权重: {LOSS_WEIGHTS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    logger.info("\n正在准备数据集...")
    train_dataset = MultiTaskSCCDDataset(
        csv_file=train_file, tokenizer=tokenizer, max_length=MAX_LENGTH, active_tasks=ACTIVE_TASKS
    )
    eval_dataset = MultiTaskSCCDDataset(
        csv_file=test_file, tokenizer=tokenizer, max_length=MAX_LENGTH, active_tasks=ACTIVE_TASKS
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    logger.info("数据集准备完成。")

    task_num_labels = {
        name: len(labels) for name, labels in MultiTaskSCCDDataset.ALL_TASK_LABELS.items()
        if name in ACTIVE_TASKS
    }
    model = MultiTaskBert.from_pretrained(
        MODEL_NAME, config=config, model_name=MODEL_NAME, task_num_labels=task_num_labels
    )

    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        loss_weights=LOSS_WEIGHTS,  # 传入权重
        logger=logger
    )
    trainer.train()


if __name__ == '__main__':
    main()
