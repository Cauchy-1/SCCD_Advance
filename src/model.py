# src/model.py

import torch.nn as nn
from transformers import AutoModel


class CyberbullyingClassifier(nn.Module):
    def __init__(self, model_name, n_classes=2, dropout_rate=0.3):
        """
        初始化模型
        :param model_name: Hugging Face上的预训练模型名称
        :param n_classes: 分类数量
        :param dropout_rate: Dropout层的丢弃率，用于防止过拟合
        """
        super(CyberbullyingClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # --- 关键修改：在分类器中加入Dropout层 ---
        # 使用nn.Sequential将Dropout层和线性层组合起来
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.bert.config.hidden_size, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        """
        前向传播
        """
        # 从BERT模型获取输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # BERT的输出中，pooler_output是[CLS]标记经过线性层和Tanh激活函数后的表示
        # 它被设计用于句子级别的分类任务
        pooled_output = outputs.pooler_output

        # 将pooled_output送入我们定义的分类器（包含Dropout）
        logits = self.classifier(pooled_output)

        return logits
