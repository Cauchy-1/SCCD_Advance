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


# -----------------

def setup_logging(log_dir: Path):
    """配置日志记录，同时输出到文件和控制台"""
    log_dir.mkdir(exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename

    # 创建一个 logger
    logger = logging.getLogger('multi_task_logger')
    logger.setLevel(logging.INFO)

    # 创建文件 handler
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建控制台 handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # 创建 formatter 并将其添加到 handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将 handlers 添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def main():
    # --- 1. 设置路径、设备和日志 ---
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    log_dir = current_dir / 'logs'
    logger = setup_logging(log_dir)

    train_file = project_root / 'datas' / 'SCCD' / 'train.csv'
    test_file = project_root / 'datas' / 'SCCD' / 'test.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"使用设备: {device}")
    logger.info(f"训练文件: {train_file}")
    logger.info(f"测试文件: {test_file}")

    # --- 2. 加载 Tokenizer 和 Config ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # --- 3. 创建数据集和数据加载器 ---
    logger.info("\n正在准备数据集...")
    train_dataset = MultiTaskSCCDDataset(csv_file=train_file, tokenizer=tokenizer, max_length=MAX_LENGTH)
    eval_dataset = MultiTaskSCCDDataset(csv_file=test_file, tokenizer=tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    logger.info("数据集准备完成。")

    # --- 4. 初始化模型 ---
    # 定义每个任务的标签数量
    task_num_labels = {
        name: len(labels) for name, labels in MultiTaskSCCDDataset.TASK_LABELS.items()
    }
    model = MultiTaskBert.from_pretrained(
        MODEL_NAME,
        config=config,
        model_name=MODEL_NAME,  # 将 model_name 作为关键字参数传递
        task_num_labels=task_num_labels
    )

    # --- 5. 初始化并启动训练器 ---
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        logger=logger  # 传入 logger
    )

    trainer.train()


if __name__ == '__main__':
    main()
