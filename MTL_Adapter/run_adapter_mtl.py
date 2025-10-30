import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import logging
from datetime import datetime

from data_loader import MultiTaskSCCDDataset
from model import AdapterBert  # [修改] 导入新的Adapter模型
from trainer import MultiTaskTrainer

# --- 配置 ---
MODEL_NAME = "bert-base-chinese"
BATCH_SIZE = 16
LEARNING_RATE = 3e-5  # Adapter训练通常使用比完全微调稍大的学习率
EPOCHS = 5  # Adapter训练收敛更快，可以适当增加epoch
MAX_LENGTH = 128

# [配置] 沿用0.5权重的任务筛选和损失加权配置
ACTIVE_TASKS = ['label', 'expression', 'target']
LOSS_WEIGHTS = {
    'label': 1.0,
    'expression': 0.5,
    'target': 0.5
}


# -----------------

def setup_logging(log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    logger = logging.getLogger('adapter_mtl_logger')
    logger.setLevel(logging.INFO)

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

    # [修改] 初始化新的Adapter模型
    model = AdapterBert.from_pretrained(
        MODEL_NAME,
        config=config
    )
    # 调用新的方法来设置任务
    model.setup_tasks(task_num_labels)
    model.to(device)  # 将模型移动到设备
    # 打印可训练参数的数量，验证Adapter是否配置成功
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params}")
    logger.info(f"可训练参数量: {trainable_params} ({(100 * trainable_params / total_params):.2f}%)")

    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        loss_weights=LOSS_WEIGHTS,
        logger=logger
    )
    trainer.train()


if __name__ == '__main__':
    main()