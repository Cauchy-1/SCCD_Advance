# 中文网络欺凌检测项目 (SCCD)改进

本项目是围绕 “SCCD: A Session-based Dataset for Chinese Cyberbullying Detection” 数据集进行的一系列中文网络欺凌检测实验的集合。它包含了多种基于深度学习的模型实现，例如多任务学习、Adapter-tuning 等。


## 目录结构
~~~c E:\pythonCode\SCCD
├── CL-CD_Table6_Replication/
│   ├── README.md
│   ├── requirements.txt
│   ├── run_bert_finetune.py
│   ├── run_roberta_finetune.py
│   ├── utils.py
│   └── __pycache__/
│       └── utils.cpython-312.pyc
├── dataclean/
│   ├── cleaned_comments.csv
│   └── data_cleaner.py
├── datas/
│   ├── light_Tencent_AILab_ChineseEmbedding.bin
│   ├── Yang 等 - SCCD A Session-based Dataset for Chinese Cyberbullying Detection.pdf
│   └── SCCD/
│       ├── comments.csv
│       ├── posts.csv
│       ├── README.md
│       ├── reposts.csv
│       ├── test.csv
│       ├── train.csv
│       ├── users.csv
│       └── assets/
│           ├── construction.png
│           ├── examples.png
│           └── session.png
├── MTL_Adapter/
│   ├── data_loader.py
│   ├── model.py
│   ├── readme.md
│   ├── run_adapter_mtl.py
│   ├── trainer.py
│   ├── __pycache__/
│   │   ├── data_loader.cpython-312.pyc
│   │   ├── model.cpython-312.pyc
│   │   └── trainer.cpython-312.pyc
│   └── logs/
│       ├── 20251008_181054.log
│       ├── ... (and other log files)
├── MTL_Improved/
│   ├── data_loader.py
│   ├── model.py
│   ├── readme.md
│   ├── run_improved_mtl.py
│   ├── trainer.py
│   ├── __pycache__/
│   │   ├── data_loader.cpython-312.pyc
│   │   ├── model.cpython-312.pyc
│   │   └── trainer.cpython-312.pyc
│   └── logs/
│       ├── 20251004_170922.log
│       ├── ... (and other log files)
├── MultiTask_Learning/
│   ├── data_loader.py
│   ├── model.py
│   ├── README.md
│   ├── run_multitask.py
│   ├── trainer.py
│   ├── __pycache__/
│   │   ├── data_loader.cpython-312.pyc
│   │   ├── model.cpython-312.pyc
│   │   └── trainer.cpython-312.pyc
│   └── logs/
│       ├── 20251002_104428.log
│       ├── ... (and other log files)
└── src/
~~~
## 数据集

本项目使用的核心数据集是 SCCD，源文件位于 `datas/SCCD/` 目录下。该数据集以“会话”为单位，整合了帖子、评论、转发和用户信息，用于网络欺凌检测任务。

- `train.csv`: 训练集
- `test.csv`: 测试集
- `posts.csv`, `comments.csv`, `reposts.csv`, `users.csv`: 包含详细信息的原始数据表

## 模型与实验

项目中包含了多种检测方法的实现：

1.  **`MultiTask_Learning/`**: 一个基准的多任务学习（MTL）模型，用于同时学习多个相关任务。
2.  **`MTL_Improved/`**: 对基准 MTL 模型的改进版本。
3.  **`MTL_Adapter/`**: 采用 Adapter-tuning 技术的 MTL 模型。这是一种参数高效的微调方法，只训练少量插入到预训练模型中的 Adapter 模块，从而大幅减少训练参数量。
4.  **`CL-CD_Table6_Replication/`**: 通过标准的模型微调（Fine-tuning）方式复现了 BERT 和 RoBERTa 在该任务上的性能。

## 如何运行

每个实验目录（如 `MTL_Adapter/`）都是一个独立的模块，如何运行请移步对应模块下的README.md文件
