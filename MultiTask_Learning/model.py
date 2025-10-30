import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput


class MultiTaskBert(BertPreTrainedModel):
    def __init__(self, config, model_name, task_num_labels):
        super().__init__(config)
        self.task_num_labels = task_num_labels
        self.bert = BertModel.from_pretrained(model_name, config=config)

        self.classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, num_labels)
            for task, num_labels in task_num_labels.items()
        })

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        logits = {}
        for task, classifier in self.classifiers.items():
            logits[task] = classifier(pooled_output)

        total_loss = None
        if labels is not None:
            # [修复] 添加 ignore_index=-100 来解决 NaN loss 问题
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            total_loss = 0
            for task, task_logits in logits.items():
                if task in labels:
                    # 确保该批次中至少有一个有效标签
                    if labels[task].ne(-100).sum() > 0:
                        loss = loss_fct(task_logits.view(-1, self.task_num_labels[task]), labels[task].view(-1))
                        # 检查 loss 是否为 nan，如果是则跳过
                        if not loss.isnan():
                            total_loss += loss

        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
