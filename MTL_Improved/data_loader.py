import pandas as pd
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class MultiTaskSCCDDataset(Dataset):
    """
    为多任务学习加载和预处理 SCCD 数据集。
    现在支持只选择激活的任务。
    """
    # 包含所有可能的任务及其标签
    ALL_TASK_LABELS = {
        'label': {'Non-CB': 0, 'CB': 1},
        'expression': {'Explicit': 0, 'Implicit': 1},
        'sarcasm': {'No': 0, 'Yes': 1},
        'target': {'Individual': 0, 'Group': 1}
    }

    def __init__(self,
                 csv_file: str,
                 tokenizer: PreTrainedTokenizer,
               active_tasks: List[str],
                 max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.active_tasks = active_tasks

        df = pd.read_csv(csv_file)
        df.dropna(subset=['comment_content'], inplace=True)
        self.texts = df['comment_content'].tolist()

        self.labels = {}
        for task_name in self.active_tasks:
            if task_name in self.ALL_TASK_LABELS:
                label_map = self.ALL_TASK_LABELS[task_name]
                self.labels[task_name] = [
                    label_map.get(x, -100) for x in df[task_name]
                ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        for task_name, labels in self.labels.items():
            item[task_name] = labels[idx]

        return item

    def get_labels_for_task(self, task_name: str) -> List[str]:
        if task_name not in self.ALL_TASK_LABELS:
            return []

        label_map = self.ALL_TASK_LABELS[task_name]
        return sorted(label_map.keys(), key=lambda k: label_map[k])
