# 为了确保在训练集和测试集中类别标签（label）的分布保持一致，
# 采用分层抽样  在清洗数据后，将其按 7:3 的比例分割为训练集和测试集，并分别保存
import pandas as pd
import re
import os
from pathlib import Path
from opencc import OpenCC
from sklearn.model_selection import train_test_split


def clean_text(text: str, s2t_converter: OpenCC) -> str:
    """
    根据 SCCD 论文中描述的步骤清洗和规范化单条评论文本。

    Args:
        text (str): 原始评论文本。
        s2t_converter (OpenCC): OpenCC 繁转简转换器实例。

    Returns:
        str: 清洗和规范化后的文本。
    """
    # 确保输入为字符串
    if not isinstance(text, str):
        text = str(text)

    # 1. 移除 URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 2. 移除 @USER 提及 (隐私处理)
    text = re.sub(r'@\S+', '', text)

    # 3. 移除表情符号 (包括微博风格的 [文字] 和 unicode emoji)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # 4. 移除 "转发微博" 等无关内容
    text = text.replace('转发微博', '')

    # 5. 繁体转简体
    text = s2t_converter.convert(text)

    # 6. 转为小写
    text = text.lower()

    # 7. 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def main():
    """
    主函数，用于加载、处理和保存数据。
    """
    # --- 1. 设置文件路径 ---
    # 获取当前脚本所在的目录 (dataclean/)
    current_dir = Path(__file__).parent
    # 获取项目根目录 (SCCD/)
    project_root = current_dir.parent

    input_path = project_root / 'datas' / 'SCCD' / 'comments.csv'
    # 定义输出目录为 datas/SCCD
    output_dir = project_root / 'datas' / 'SCCD'
    train_output_path = output_dir / 'train.csv'
    test_output_path = output_dir / 'test.csv'

    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)

    print(f"输入文件: {input_path}")
    print(f"训练集输出文件: {train_output_path}")
    print(f"测试集输出文件: {test_output_path}")

    # --- 2. 加载数据 ---
    print("\n正在加载原始数据...")
    if not input_path.exists():
        print(f"错误: 输入文件未找到: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"成功加载 {len(df)} 条评论。")

    # --- 3. 数据清洗 ---
    print("开始清洗文本数据...")

    # 初始化繁简体转换器
    # 修复了 OpenCC 初始化问题，使用 't2s' 而不是 't2s.json'
    s2t_converter = OpenCC('t2s')

    # 应用清洗函数到 'comment_content' 列
    # 使用 .copy() 避免 SettingWithCopyWarning
    df_cleaned = df.copy()
    df_cleaned['comment_content'] = df_cleaned['comment_content'].apply(lambda x: clean_text(x, s2t_converter))

    # 移除清洗后为空的行和原始的空值行
    original_rows = len(df_cleaned)
    df_cleaned.dropna(subset=['comment_content'], inplace=True)
    df_cleaned = df_cleaned[df_cleaned['comment_content'] != '']
    cleaned_rows = len(df_cleaned)
    print(f"移除了 {original_rows - cleaned_rows} 条空评论。")

    # --- 4. 划分训练集和测试集 ---
    print(f"正在将 {cleaned_rows} 条数据按 7:3 划分为训练集和测试集...")
    if 'label' not in df_cleaned.columns:
        print("错误: 'label' 列不存在，无法进行分层抽样。")
        return

    train_df, test_df = train_test_split(
        df_cleaned,
        test_size=0.3,
        random_state=42,  # 为了结果可复现
        stratify=df_cleaned['label']  # 根据标签进行分层抽样
    )
    print(f"训练集大小: {len(train_df)} 条")
    print(f"测试集大小: {len(test_df)} 条")

    # --- 5. 保存处理后的数据 ---
    print(f"正在保存训练集到 {train_output_path}...")
    train_df.to_csv(train_output_path, index=False)

    print(f"正在保存测试集到 {test_output_path}...")
    test_df.to_csv(test_output_path, index=False)

    print("数据处理和划分完成！")


if __name__ == '__main__':
    main()
