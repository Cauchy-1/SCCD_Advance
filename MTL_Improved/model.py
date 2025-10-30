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

        logits = {
            task: classifier(pooled_output)
            for task, classifier in self.classifiers.items()
        }

        task_losses = {}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            for task, task_logits in logits.items():
                if task in labels and labels[task].ne(-100).sum() > 0:
                    loss = loss_fct(task_logits.view(-1, self.task_num_labels[task]), labels[task].view(-1))
                    if not loss.isnan():
                        task_losses[task] = loss

        return TokenClassifierOutput(
            loss=task_losses if task_losses else None, # [修改] 返回损失字典
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )