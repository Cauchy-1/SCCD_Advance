import os
import logging
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from utils import setup_logging, load_and_prepare_dataset, compute_metrics

# --- 配置区 ---
# 模型标识符 (Hugging Face Hub)
MODEL_NAME = "bert-base-chinese"
# 任务名称，用于日志和输出文件夹
TASK_NAME = "bert-finetune-sccd"


# --- End ---

# 自定义回调以记录每个评估步骤的结果
class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict,
                    **kwargs):
        # 记录除了训练历史之外的每个评估指标
        if state.is_world_process_zero:
            self.logger.info(f"Step {state.global_step} 评估结果:")
            for key, value in metrics.items():
                if key != "epoch":
                    self.logger.info(f"  - {key}: {value}")


def main():
    # --- 1. 设置路径 ---
    project_root = Path(__file__).resolve().parent.parent
    train_file_path = project_root / 'datas' / 'SCCD' / 'train.csv'
    test_file_path = project_root / 'datas' / 'SCCD' / 'test.csv'

    # 根据您的要求更新输出和日志目录
    output_base_dir = project_root / 'CL-CD_Table6_Replication' / 'outputs'
    model_output_dir = output_base_dir / TASK_NAME
    log_dir = output_base_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. 设置日志 ---
    # 日志文件名: 模型名 + 时间戳
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file_name = f"{TASK_NAME}_{current_time}.log"
    log_file_path = log_dir / log_file_name

    # 配置日志记录器
    logger = logging.getLogger(TASK_NAME)
    logger.setLevel(logging.INFO)
    # 移除默认的处理器，以防重复记录
    if logger.hasHandlers():
        logger.handlers.clear()
    # 文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

    logger.info("=" * 30)
    logger.info(f"开始复现 Table 6: {TASK_NAME}")
    logger.info("=" * 30)
    logger.info(f"日志文件保存在: {log_file_path}")

    # --- 3. 加载 Tokenizer 和模型 ---
    logger.info(f"正在加载模型和Tokenizer: {MODEL_NAME}")
    # 添加 trust_remote_code=True 尝试解决连接问题
    # 如果问题仍然存在，请考虑设置HTTP/HTTPS代理或使用模型的本地路径
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, trust_remote_code=True)

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
        output_dir=str(model_output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(model_output_dir, 'runs'),
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy (micro_f1)",
        report_to="none"
    )

    # 记录训练参数
    logger.info("--- 训练参数 ---")
    for arg, value in training_args.to_dict().items():
        logger.info(f"  - {arg}: {value}")
    logger.info("--------------------")

    # --- 6. 初始化并启动 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback(logger)]  # 添加自定义日志回调
    )

    logger.info("开始模型训练...")
    trainer.train()
    logger.info("模型训练完成。")

    # --- 7. 评估最佳模型 ---
    logger.info("使用最佳模型在测试集上进行最终评估...")
    eval_results = trainer.evaluate()

    # --- 8. 打印最终结果 ---
    logger.info("\n" + "=" * 30)
    logger.info("      最终评估结果 (复现 Table 6)      ")
    logger.info("=" * 30)
    logger.info(f"模型: {MODEL_NAME}")
    logger.info(f"  - Precision (Macro): {eval_results['eval_precision_macro']:.4f}")
    logger.info(f"  - Recall (Macro):    {eval_results['eval_recall_macro']:.4f}")
    logger.info(f"  - Micro F1 (Acc):    {eval_results['eval_accuracy (micro_f1)']:.4f}")
    logger.info("=" * 30)
    logger.info(f"实验完成。所有日志和模型检查点保存在: {model_output_dir}")


if __name__ == "__main__":
    main()
