# src/dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re  # 导入正则表达式库


class SCCDDataset(Dataset):
    def __init__(self, data_path, split='train', model_name='hfl/chinese-roberta-wwm-ext', max_length=512,
                 top_k_comments=5):
        self.data_path = data_path
        self.split = split
        self.max_length = max_length
        self.top_k_comments = top_k_comments

        self.posts_df = pd.read_csv(f"{data_path}/posts.csv")
        self.comments_df = pd.read_csv(f"{data_path}/comments.csv")

        self.posts_df['label_id'] = self.posts_df['label'].apply(lambda x: 1 if x == 'CB' else 0)

        # 为了保证每次运行的训练集和测试集划分一致，我们设置一个固定的随机种子
        shuffled_ids = self.posts_df['post_id'].unique()
        # 使用 pandas 的 sample 功能来随机打乱，frac=1 表示全部抽取，random_state 保证可复现
        shuffled_ids = pd.Series(shuffled_ids).sample(frac=1, random_state=42).tolist()

        train_size = int(len(shuffled_ids) * 0.8)
        if split == 'train':
            self.session_ids = shuffled_ids[:train_size]
        else:
            self.session_ids = shuffled_ids[train_size:]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.session_ids)

    def _clean_text(self, text):
        """
        简单的文本清洗函数
        - 移除URL
        - 移除@提及
        - 移除话题标签
        - 移除大部分特殊符号
        - 规范化空格
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __getitem__(self, idx):
        post_id = self.session_ids[idx]

        post_info = self.posts_df[self.posts_df['post_id'] == post_id].iloc[0]
        # --- 关键修改：清洗帖子内容 ---
        post_content = self._clean_text(post_info['post_content'])
        label = post_info['label_id']

        related_comments = self.comments_df[self.comments_df['post_id'] == post_id]
        top_comments = related_comments.sort_values(by='like_num', ascending=False).head(self.top_k_comments)

        context_text = post_content
        for comment_text in top_comments['comment_content']:
            # --- 关键修改：清洗评论内容 ---
            cleaned_comment = self._clean_text(comment_text)
            if cleaned_comment:  # 确保清洗后不为空
                context_text += f" [SEP] {cleaned_comment}"

        encoding = self.tokenizer.encode_plus(
            context_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
