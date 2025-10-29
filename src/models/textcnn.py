"""
TextCNN模型实现
用于文本分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN模型
    
    Args:
        vocab_size: 词表大小
        embed_dim: 词向量维度
        num_classes: 分类类别数
        num_filters: 每个卷积核的数量
        filter_sizes: 卷积核大小列表，如[3, 4, 5]
        dropout: dropout比例
    """
    
    def __init__(self, vocab_size, embed_dim=300, num_classes=14, 
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        
        # 1. Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. 多个卷积层（不同kernel size）
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,      # 输入通道=词向量维度
                out_channels=num_filters,   # 输出通道=卷积核数量
                kernel_size=k               # 卷积核大小（3/4/5-gram）
            )
            for k in filter_sizes
        ])
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len] - 输入的ID序列
        
        Returns:
            out: [batch_size, num_classes] - 分类logits
        """
        # 1. Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.embedding(x)
        
        # 2. 转换为Conv1d需要的格式: [batch_size, embed_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # 3. 多个卷积+ReLU+MaxPooling
        conv_outputs = []
        for conv in self.convs:
            # Conv1d: [batch_size, embed_dim, seq_len] -> [batch_size, num_filters, seq_len-k+1]
            conv_out = F.relu(conv(x))
            
            # MaxPool: [batch_size, num_filters, seq_len-k+1] -> [batch_size, num_filters, 1]
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            
            # Squeeze: [batch_size, num_filters, 1] -> [batch_size, num_filters]
            pooled = pooled.squeeze(2)
            
            conv_outputs.append(pooled)
        
        # 4. 拼接所有卷积结果: [batch_size, num_filters * len(filter_sizes)]
        x = torch.cat(conv_outputs, dim=1)
        
        # 5. Dropout
        x = self.dropout(x)
        
        # 6. 全连接层
        out = self.fc(x)
        
        return out


# 测试代码
if __name__ == '__main__':
    # 测试模型
    vocab_size = 10000
    embed_dim = 300
    num_classes = 15
    
    model = TextCNN(vocab_size, embed_dim, num_classes)
    
    # 模拟输入
    batch_size = 32
    seq_len = 512
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    out = model(x)
    
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {out.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ 模型测试通过！")





