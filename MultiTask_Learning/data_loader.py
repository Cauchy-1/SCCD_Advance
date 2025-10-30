import pandas as pd
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class MultiTaskSCCDDataset(Dataset):
    """
    为多任务学习加载和预处理 SCCD 数据集。
    - 主任务: label (是否网络欺凌)
    - 辅助任务: expression, sarcasm, target
    """
    TASK_LABELS = {
        'label': {'Non-CB': 0, 'CB': 1},
        'expression': {'Explicit': 0, 'Implicit': 1},
        'sarcasm': {'No': 0, 'Yes': 1},
        'target': {'Individual': 0, 'Group': 1}
    }

    def __init__(self, csv_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 加载数据
        df = pd.read_csv(csv_file)

        # 2. 清洗数据：移除 'comment_content' 列的空值
        df.dropna(subset=['comment_content'], inplace=True)
        self.texts = df['comment_content'].tolist()

        # 3. 编码所有任务的标签
        self.labels = {}
        for task_name, label_map in self.TASK_LABELS.items():
            # 使用 .get(x, -100) 来处理 NaN 或未知的标签值
            # -100 是 PyTorch CrossEntropyLoss 的默认 ignore_index
            self.labels[task_name] = [
                label_map.get(x, -100) for x in df[task_name]
            ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        text = self.texts[idx]

        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 整理返回的数据，包含所有任务的标签
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        for task_name, labels in self.labels.items():
            item[task_name] = labels[idx]

        return item

    def get_labels_for_task(self, task_name: str) -> List[str]:
        """
        [修复] 添加此方法来解决 AttributeError。
        为 classification_report 返回有序的标签名列表。
        """
        if task_name not in self.TASK_LABELS:
            return []

        label_map = self.TASK_LABELS[task_name]
        # 通过值（0, 1, ...）排序，以确保标签名顺序正确
        return sorted(label_map.keys(), key=lambda k: label_map[k])