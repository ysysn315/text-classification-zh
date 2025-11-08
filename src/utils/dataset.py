"""
数据集类
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


class TextDataset(Dataset):
    """
    文本分类数据集
    
    Args:
        data_path: 处理后的pkl文件路径
    """
    
    def __init__(self, data_path):
        # 读取处理好的数据
        self.data = pd.read_pickle(data_path)
        
        # 提取需要的列
        self.texts = self.data['text'].values
        self.labels = self.data['label'].values  # 数字label（0-14）
        self.ids = self.data['ids'].values       # ID序列
        
        print(f"加载数据: {len(self.data)} 条")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取一条数据
        
        Returns:
            ids: tensor [seq_len]
            label: tensor (标量)
        """
        # 转为tensor
        ids = torch.LongTensor(self.ids[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return ids, label


# 测试代码
if __name__ == '__main__':
    # 测试Dataset
    dataset = TextDataset('../data/processed/train_processed.pkl')
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取第一条数据
    ids, label = dataset[0]
    print(f"\n第一条数据:")
    print(f"  IDs shape: {ids.shape}")
    print(f"  IDs前20个: {ids[:20]}")
    print(f"  Label: {label}")
    
    # 使用DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 获取一个batch
    for ids_batch, labels_batch in loader:
        print(f"\nBatch数据:")
        print(f"  IDs shape: {ids_batch.shape}")
        print(f"  Labels shape: {labels_batch.shape}")
        break
    
    print("\n Dataset测试通过！")














