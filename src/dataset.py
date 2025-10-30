# python
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from gensim.models import KeyedVectors
except Exception:
    KeyedVectors = None


def simple_zh_tokenize(text: str) -> List[str]:
    # 简单中文分词：按字符切分并去除空白
    if not isinstance(text, str):
        text = str(text)
    return [ch for ch in text.strip() if ch.strip() != '']


class SCCDDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        comment_col: str = 'comment_content',
        label_col: str = 'label',
        vocab: Optional[Dict[str, int]] = None,
        min_freq: int = 1,
        tokenizer=simple_zh_tokenize
    ):
        # 读取数据
        df = pd.read_csv(csv_file)
        if comment_col not in df.columns or label_col not in df.columns:
            raise ValueError(f'CSV缺少必要列: {comment_col} 或 {label_col}')
        df = df[[comment_col, label_col]].dropna().reset_index(drop=True)

        # 标签统一编码为整数
        raw_labels = df[label_col].astype(str).values
        unique_labels = sorted(list(set(raw_labels)))
        self.label2id = {lb: i for i, lb in enumerate(unique_labels)}
        self.id2label = {i: lb for lb, i in self.label2id.items()}
        df['label_id'] = [self.label2id[lb] for lb in raw_labels]
        self.num_labels = len(self.label2id)

        # 分词
        self.tokenizer = tokenizer
        df['tokens'] = df[comment_col].astype(str).map(self.tokenizer)

        # 构建或使用现有词表
        if vocab is None:
            from collections import Counter
            counter = Counter()
            for toks in df['tokens']:
                counter.update(toks)
            stoi: Dict[str, int] = {'<pad>': 0, '<unk>': 1}
            for tok, freq in counter.items():
                if freq >= min_freq and tok not in stoi:
                    stoi[tok] = len(stoi)
            self.vocab = stoi
        else:
            self.vocab = vocab

        self.unk_idx = self.vocab.get('<unk>', 1)
        self.pad_idx = self.vocab.get('<pad>', 0)

        # 文本转ID序列
        df['ids'] = df['tokens'].map(lambda toks: [self.vocab.get(t, self.unk_idx) for t in toks])

        # 最终数据
        self.data = df[[comment_col, 'ids', 'label_id']]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        row = self.data.iloc[idx]
        token_ids = row['ids']
        label_id = int(row['label_id'])
        return token_ids, label_id

    def get_all_label_ids(self) -> np.ndarray:
        return self.data['label_id'].values

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def generate_batch(batch, pad_idx: int, max_len: Optional[int] = None):
    # 可选裁剪到max_len以降低显存与噪声
    texts, labels = zip(*batch)  # texts: List[List[int]], labels: List[int]
    if max_len is not None and max_len > 0:
        texts = [x[:max_len] for x in texts]
    max_len_batch = max(len(x) for x in texts) if texts else 0
    padded = [x + [pad_idx] * (max_len_batch - len(x)) for x in texts]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def load_pretrained_vectors(vector_path: str, vocab: Dict[str, int], embed_dim: int) -> Optional[torch.Tensor]:
    if KeyedVectors is None:
        logging.warning("gensim 未安装，跳过预训练词向量加载。")
        return None
    logging.info(f"loading projection weights from {vector_path}")
    kv = KeyedVectors.load_word2vec_format(vector_path, binary=True)
    if kv.vector_size != embed_dim:
        logging.warning(f"预训练向量维度 {kv.vector_size} 与 embed_dim={embed_dim} 不一致。")
        return None

    mat = torch.empty(len(vocab), embed_dim, dtype=torch.float32)
    torch.nn.init.normal_(mat, mean=0.0, std=0.1)

    hit = 0
    for tok, idx in vocab.items():
        if tok in kv:
            # 关键：拷贝为可写的numpy，再转tensor，避免只读数组警告
            vec = np.array(kv[tok], copy=True)
            mat[idx] = torch.from_numpy(vec).to(torch.float32)
            hit += 1

    # 可选稳定化：将<pad>/<unk>置零
    pad_idx = vocab.get('<pad>', None)
    unk_idx = vocab.get('<unk>', None)
    if pad_idx is not None:
        mat[pad_idx].zero_()
    if unk_idx is not None:
        mat[unk_idx].zero_()

    logging.info(f"预训练覆盖词表: {hit}/{len(vocab)}")
    return mat
