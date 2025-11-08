import torch
from torch.utils.data import Dataset
import pandas as pd


class BertDataset(Dataset):
    """
    BERT文本分类数据集

    和TextCNN Dataset的区别：
    - TextCNN: 使用自建词表，手动转ID
    - BERT: 使用BertTokenizer，自动处理

    Args:
        data_path: CSV文件路径（train.csv/val.csv/test.csv）
        tokenizer: BertTokenizer对象
        max_len: 最大序列长度，默认512
    """

    def __init__(self, data_path, tokenizer, max_len=512):
        # 读取CSV数据
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f" 加载数据: {len(self.data)} 条")
        print(f"Max length: {max_len}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取一条数据

        Returns:
            dict: {
                'input_ids': tensor [max_len],
                'attention_mask': tensor [max_len],
                'labels': tensor (标量)
            }
        """
        # 获取文本和标签
        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['label'])

        # === 核心：使用BertTokenizer处理文本 ===
        # 这一步替代了TextCNN中的"分词+建词表+转ID"
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # 返回PyTorch tensor
        )

        # encoding包含：
        # - input_ids: token的ID序列 [1, max_len]
        # - attention_mask: 标记哪些是真实token [1, max_len]

        # 返回字典格式
        return {
            'input_ids': encoding['input_ids'].flatten(),  # [max_len]
            'attention_mask': encoding['attention_mask'].flatten(),  # [max_len]
            'labels': torch.tensor(label, dtype=torch.long)
        }
# ===== 测试代码 =====
if __name__ == '__main__':
    from transformers import BertTokenizer

    # 测试Dataset
    print("测试BertDataset...")

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 创建dataset
    dataset = BertDataset(
        '../../data/processed/train.csv',
        tokenizer,
        max_len=128  # 测试时用短一点
    )

    print(f"\n数据集大小: {len(dataset)}")

    # 获取第一条数据
    sample = dataset[0]

    print(f"\n第一条数据:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  input_ids前20个: {sample['input_ids'][:20]}")
    print(f"  attention_mask前20个: {sample['attention_mask'][:20]}")
    print(f"  label: {sample['labels']}")

    # 解码看看
    decoded = tokenizer.decode(sample['input_ids'])
    print(f"\n  解码后的文本: {decoded[:100]}...")

    # 使用DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 获取一个batch
    batch = next(iter(loader))
    print(f"\nBatch数据:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")

    print("\n BertDataset测试通过！")