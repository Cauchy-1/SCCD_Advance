# 复现 SCCD 论文 Table 6 实验
本项目包含了复现论文 "SCCD: A Session-based Dataset for Chinese Cyberbullying Detection" 中 Table 6 评论级别网络欺凌检测 (CL-CD) 任务的代码。

## 结构说明
- utils.py: 包含数据加载、指标计算等共享工具函数。

- run_bert_finetune.py: 用于微调 bert-base-chinese 模型。该脚本同时用于复现 Bert 和 COLDETECTOR 的结果，因为后者也是基于相同的预训练模型。

- run_roberta_finetune.py: 用于微调 hfl/chinese-roberta-wwm-ext 模型，以复现 Roberta 的结果。

- requirements.txt: 运行本项目所需的 Python 依赖。

- outputs/: 脚本运行时，会自动创建此目录来存放训练日志、模型检查点和评估结果。

``注意: 论文中提到的 Baidu TC 是一个在线 API 服务，无法通过开源代码复现，因此本项目未包含此项。``

# 运行步骤
1. 准备环境
请确保您的项目目录结构如下，并且 train.csv 和 test.csv 文件已存在于 datas/SCCD/ 目录下。


2. 安装依赖
在终端中，进入 CL-CD_Table6_Replication 目录，然后使用 pip 安装所有必需的库：

cd path/to/your/project/SCCD/CL-CD_Table6_Replication
pip install -r requirements.txt

3. 运行实验
您可以分别运行以下命令来启动不同模型的微调和评估过程。

复现 Bert / COLDETECTOR:

python run_bert_finetune.py

复现 Roberta:

python run_roberta_finetune.py

4. 查看结果
每个实验的训练过程和最终评估结果将实时打印在您的终端上，并同时保存在 outputs/<任务名称>/logs/ 目录下的日志文件中。

训练完成后，最终的评估指标 (Precision, Recall, Micro F1) 将被清晰地打印出来，您可以将这些结果与论文中的 Table 6 进行对比。

## Results

模型在所有关键指标上全面超越了论文中Table 6报告的BERT基线结果。




