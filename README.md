# 中文新闻文本分类

这个项目是基于完整的THUCNews数据集，采用TextCNN,LSTM,BERT三个方法对不同类型的新闻，进行文本分类
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

---

## 📖 项目简介

### 背景

文本分类是自然语言处理领域的基础任务，广泛应用于新闻分类、情感分析、垃圾邮件过滤等场景。本项目通过实现TextCNN，LSTM，BERT三种经典深度学习模型，对比它们在中文新闻分类任务上的性能表现，探索不同模型架构的优劣。

通过这个项目，我希望：
- 掌握完整的NLP项目流程（数据处理→模型训练→评估部署）
- 深入理解TextCNN和LSTM的核心原理和适用场景
- 积累文本分类项目经验，为后续的大模型学习打下基础

### 主要功能
- 中文新闻多分类（14个类别）
- 实现TextCNN、LSTM、BERT三种模型
- 完整的训练和评估流程
- 详细的对比分析
- 可视化训练过程和结果

### 技术特点
-  对比了TextCNN、LSTM、BERT三种模型
-  完整的数据预处理流程（jieba分词、词表构建、文本转ID）
-  详细的对比分析（混淆矩阵、各类别准确率、推理速度）
-  可视化训练过程和结果
-  代码模块化、可复用

---

## 📊 实验结果

### 模型对比

| 模型 | 验证集准确率 | 测试集准确率 | 参数量 |
|------|------------|------------|--------|
| TextCNN | 94.83%  | 95.39%     | 35.6M   | 
| LSTM    | 95.39%  | 93.47%     | 37.9M   |
| BERT    | 96.58% | 96.99%      |  102.3M|
<img width="1485" height="884" alt="three_models_accuracy_comparison" src="https://github.com/user-attachments/assets/1e21c2ff-a674-4a4a-815f-a98e44061f15" />
<img width="2927" height="884" alt="three_models_confusion_matrix" src="https://github.com/user-attachments/assets/02b68414-7550-46fb-9ac6-51c7596bdeae" />


**关键发现**：
- BERT在测试集上表现最好（96.99%），预训练模型优势明显
- TextCNN参数量少、速度快，更适合部署
- LSTM参数量更大，但在这个任务上优势不明显

### 训练曲线
<img width="2100" height="750" alt="textcnn_training_curve" src="https://github.com/user-attachments/assets/d83bf162-20c4-4997-b6be-1623856f8ac6" />

<img width="2100" height="750" alt="lstm_training_curve" src="https://github.com/user-attachments/assets/98d4d4fa-47c8-4e40-8a4e-bcfd884ce060" />

<img width="2085" height="734" alt="bert_training_curve" src="https://github.com/user-attachments/assets/94d3a774-45f3-401f-93bc-259f0f01f21a" />



---

## 🚀 快速开始
### 必装软件
```bash
# Python环境
Anaconda/Miniconda
Python 3.8+
PyTorch 2.0+
# 深度学习框架
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft

# 常用库
pip install numpy pandas matplotlib seaborn scikit-learn
pip install jupyter notebook
pip install tensorboard wandb

# Web开发
pip install fastapi gradio streamlit uvicorn

# 其他工具
Git/GitHub Desktop
VSCode + Python插件
Typora (Markdown编辑器)
```
### 安装依赖
```bash
# 1. 克隆仓库
git clone https://github.com/ysysn315/text-classification-zh.git
cd text-classification-zh

# 2. 创建虚拟环境
conda create -n dl python=3.10
conda activate dl

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 数据准备
1. 下载THUCNews数据集：http://thuctc.thunlp.org/
2. 解压到 `data/THUCNews/THUCNews/`
3. 运行数据预处理notebook：
   - `notebooks/01_jieba_test.ipynb` (Cell 7-10)
   - `notebooks/2_data_preprocessing.ipynb` (所有cells)

### 训练模型
```bash
# 在Jupyter Notebook中运行：

# 训练TextCNN
notebooks/03_train_textcnn.ipynb

# 训练LSTM  
notebooks/05_train_lstm.ipynb

# 查看模型对比
notebooks/06_model_comparison.ipynb
```

---

## 📁 项目结构
```
text-classification-zh/
├── data/                       # 数据目录（不上传Git）
│   ├── THUCNews/              # THUCNews原始数据
│   └── processed/             # 预处理后的数据
│       ├── train.csv          # 训练集
│       ├── val.csv            # 验证集
│       ├── test.csv           # 测试集
│       ├── vocab.pkl          # 词表
│       └── *.pkl              # 处理后的数据
├── src/                       # 源代码
│   ├── models/
│   │   ├── textcnn.py         # TextCNN模型
│   │   └── lstm.py            # LSTM模型
│   └── utils/
│       └── dataset.py         # PyTorch Dataset类
├── notebooks/                 # Jupyter notebooks
│   ├── 01_jieba_test.ipynb    # jieba分词和数据读取
│   ├── 2_data_preprocessing.ipynb  # 数据预处理
│   ├── 03_train_textcnn.ipynb # TextCNN训练
│   ├── 05_train_lstm.ipynb    # LSTM训练
│   └── 06_model_comparison.ipynb # 模型对比分析
├── output/                    # 输出（模型、图表）
│   ├── textcnn_best.pth       # TextCNN最佳模型
│   ├── lstm_best.pth          # LSTM最佳模型
│   └── *.png                  # 对比图表
├── requirements.txt           # Python依赖
├── .gitignore                 # Git忽略文件
└── README.md                  # 项目说明
```


---

## 💡 技术细节

### 数据集
- **名称**：THUCNews（清华大学中文新闻数据集）
- **规模**：约70,000篇完整新闻（每个类别5,000篇）
- **类别**：14类（体育、娱乐、家居、彩票、房产、教育、时尚、时政、星座、游戏、社会、科技、股票、财经）
- **文本长度**：平均600-800字（完整新闻正文，非标题）
- **数据来源**：http://thuctc.thunlp.org/

### TextCNN
- 核心思想：通过卷积来得到文本的局部特征，再将其组合
- 主要参数：input,output,stride,kernel_size,padding,
- 优点：容易抓住局部的特征
- 缺点：如果特征的范围很大，卷积不能完全发现这个特征

### LSTM
- 核心思想：使用遗忘门，输入门等来实现长短期的记忆，既包含某些过去的信息
- 主要参数：num_input,num_hidden
- 优点：平衡了长期记忆保存和短期输入
- 缺点：相对于GRU,结构更复杂，训练成本高

### BERT
- **核心思想**：
  - **预训练+微调**：先在大量文本上预训练（学习语言表示），再在下游任务上微调
  - **Masked Language Model (MLM)**：随机遮盖15%的词，让模型预测被遮盖的词，学习上下文理解
  - **双向编码**：使用Transformer Encoder，能同时看到前后文（不像LSTM只能从左到右）
  - 本项目中：使用预训练的bert-base-chinese，在THUCNews数据集上微调用于新闻分类
  
- **主要参数**：
  - 模型：bert-base-chinese（中文预训练模型）
  - 隐藏层：768维
  - Transformer层数：12层
  - 注意力头：12个
  - 最大长度：128（本项目，标准512）
  - 学习率：2e-5（微调专用，比从零训练小50倍）
  
- **优点**：
  - 预训练模型效果好，本项目达到96.99%准确率（最高）
  - 不需要构建词表，直接用预训练的tokenizer
  - 泛化能力强，理解上下文语义
  - 迁移学习，适用各种NLP任务
  
- **缺点**：
  - 参数量大（102M，是TextCNN的26倍）
  - 训练时间长（即使是微调也需要45分钟）
  - 推理速度慢（实时系统可能不适用）
  - 显存占用大（batch_size只能16-32，TextCNN可以64+）
---

## 📝 实验记录

### 遇到的问题

1. **TNEWS数据集文本太短**
   - 问题：最初使用TNEWS数据集（新闻标题），平均只有11个词
   - 影响：TextCNN和LSTM准确率都只有47-50%
   - 解决：换用THUCNews完整新闻数据集（平均600-800字）
   - 结果：准确率提升到93-95%

2. **类别不平衡（初期）**
   - 问题：某些类别样本很少（如股票类只有211条）
   - 尝试：使用加权CrossEntropyLoss
   - 发现：THUCNews相对均衡，不需要加权

### 优化尝试
-  数据集选择：从TNEWS短文本 → THUCNews长文本
-  词表构建：包含训练集和验证集，降低<UNK>比例
-  数据划分：使用stratify保证类别均衡
-  梯度裁剪：LSTM训练时使用，防止梯度爆炸
-  后续尝试：数据增强、模型集成等 

---

