import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from utils import load_and_prepare_dataset, compute_metrics

# --- 配置区 ---
# 模型标识符 (Hugging Face Hub)
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
# 任务名称，用于日志和输出文件夹
TASK_NAME = "roberta-finetune-sccd"


# --- End ---

def setup_logging(task_name, log_dir):
    """设置日志记录器，将日志同时输出到控制台和文件"""
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件命名: robertf_YYYY-MM-DD_HH-MM.log
    log_file_name = f"robertf_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    log_file_path = log_dir / log_file_name

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - INFO - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    # --- 1. 设置路径 ---
    project_root = Path(__file__).resolve().parent.parent
    train_file_path = project_root / 'datas' / 'SCCD' / 'train.csv'
    test_file_path = project_root / 'datas' / 'SCCD' / 'test.csv'
    output_dir = project_root / 'CL-CD_Table6_Replication' / 'outputs' / TASK_NAME
    log_dir = output_dir / 'logs'  # 修改：将日志目录设置在当前实验的输出文件夹内

    # --- 2. 设置日志 ---
    logger = setup_logging(TASK_NAME, log_dir)
    logger.info("=" * 30)
    logger.info(f"开始复现 Table 6: {TASK_NAME}")
    logger.info("=" * 30)

    # --- 3. 加载 Tokenizer 和模型 ---
    logger.info(f"正在加载模型和Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # --- 4. 加载并准备数据集 ---
    logger.info(f"正在从以下路径加载数据:\n  Train: {train_file_path}\n  Test:  {test_file_path}")
    train_dataset, test_dataset = load_and_prepare_dataset(
        train_file=str(train_file_path),
        test_file=str(test_file_path),
        tokenizer=tokenizer
    )
    logger.info(f"数据加载完成。训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")

    # --- 5. 定义训练参数 ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'runs'),
        # --- 日志、评估、保存策略 ---
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,  # 只保存最好的5个checkpoint
        # --------------------------------
        load_best_model_at_end=True,
        metric_for_best_model="accuracy (micro_f1)",
        report_to="none"
    )

    # --- 6. 初始化并启动 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # --- 7. 记录训练参数并开始训练 ---
    logger.info("=" * 30)
    logger.info("      训练参数      ")
    logger.info("=" * 30)
    for arg, value in sorted(training_args.to_dict().items()):
        logger.info(f"{arg}: {value}")
    logger.info("=" * 30)

    logger.info("开始模型训练...")
    trainer.train()
    logger.info("模型训练完成。")

    # --- 8. 评估最佳模型 ---
    logger.info("使用最佳模型在测试集上进行最终评估...")
    eval_results = trainer.evaluate()

    # --- 9. 打印最终结果 ---
    logger.info("\n" + "=" * 30)
    logger.info("      最终评估结果 (复现 Table 6)      ")
    logger.info("=" * 30)
    logger.info(f"模型: {MODEL_NAME}")
    logger.info(f"  - Precision (Macro): {eval_results['eval_precision_macro']:.4f}")
    logger.info(f"  - Recall (Macro):    {eval_results['eval_recall_macro']:.4f}")
    logger.info(f"  - Micro F1 (Acc):    {eval_results['eval_accuracy (micro_f1)']:.4f}")
    logger.info("=" * 30)
    logger.info(f"实验完成。所有日志和模型检查点保存在: {output_dir}")


if __name__ == "__main__":
    main()
