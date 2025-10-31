"""
LSTM模型实现
用于文本分类任务
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM文本分类模型
    
    Args:
        vocab_size: 词表大小
        embed_dim: 词向量维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        num_classes: 分类类别数
        dropout: dropout比例
        bidirectional: 是否使用双向LSTM
    """
    
    def __init__(self, vocab_size, embed_dim=300, hidden_size=256, 
                 num_layers=2, num_classes=14, dropout=0.5, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 1. Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # 输入格式：[batch, seq, feature]
            dropout=dropout if num_layers > 1 else 0,  # LSTM层间的dropout
            bidirectional=bidirectional
        )
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. 全连接层
        # 如果是双向LSTM，hidden_size要×2
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len] - 输入的ID序列
        
        Returns:
            out: [batch_size, num_classes] - 分类logits
        """
        # 1. Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # 2. LSTM: [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, hidden_size*2]
        # output: 每个时间步的输出
        # (h_n, c_n): 最后一个时间步的hidden state和cell state
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # 3. 取最后一个时间步的输出
        # 如果是双向LSTM，h_n shape: [num_layers*2, batch_size, hidden_size]
        # 需要拼接最后一层的前向和后向
        if self.bidirectional:
            # h_n[-2]: 最后一层的前向hidden state
            # h_n[-1]: 最后一层的后向hidden state
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        
        # hidden shape: [batch_size, hidden_size*2] (如果双向)
        
        # 4. Dropout
        hidden = self.dropout(hidden)
        
        # 5. 全连接层
        out = self.fc(hidden)
        
        return out


# 测试代码
if __name__ == '__main__':
    # 测试模型
    vocab_size = 10000
    embed_dim = 300
    hidden_size = 256
    num_classes = 14
    
    model = LSTMClassifier(vocab_size, embed_dim, hidden_size, num_classes=num_classes)
    
    # 模拟输入
    batch_size = 32
    seq_len = 512
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    out = model(x)
    
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {out.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("LSTM模型测试通过！")


