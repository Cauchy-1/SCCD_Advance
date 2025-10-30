import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from adapters import BertAdapterModel
from adapters.models.bert.modeling_bert import BertOutput, BertSelfOutput


class AdapterBert(BertAdapterModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        # 初始化一个空属性，以防 setup_tasks 未被调用
        self.task_num_labels = {}
        self.active_tasks = []

    def setup_tasks(self, task_num_labels):
        """一个单独的方法来设置多任务 adapters 和 heads"""
        # [修改] 将 task_num_labels 保存为模型属性
        self.task_num_labels = task_num_labels
        self.active_tasks = list(task_num_labels.keys())

        # 遍历每个任务，为其添加Adapter和分类头
        for task_name, num_labels in task_num_labels.items():
            self.add_adapter(task_name, config="pfeiffer")
            self.add_classification_head(
                task_name,
                num_labels=num_labels
            )

        # 设置所有新添加的adapter为可训练状态
        self.train_adapter(self.active_tasks)

    def forward(self, input_ids, attention_mask, task_name, **kwargs):
        # 设置当前前向传播激活的adapter
        self.set_active_adapters(task_name)
        # 调用父类的 forward 方法，并指定 head
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head=task_name,
            **kwargs
        )