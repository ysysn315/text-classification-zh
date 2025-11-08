"""
中文新闻分类 Gradio Demo
加载训练好的BERT模型，实现Web界面预测
"""

import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys
sys.path.append('../..')

# 配置
MODEL_PATH = '../../output/bert_best.pth'
MODEL_NAME = 'bert-base-chinese'

# 类别映射
label_map = {
    0: '体育', 1: '娱乐', 2: '家居', 3: '彩票', 4: '房产', 5: '教育',
    6: '时尚', 7: '时政', 8: '星座', 9: '游戏', 10: '社会', 11: '科技',
    12: '股票', 13: '财经'
}

# 加载模型
print("加载BERT模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=14
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print(" 模型加载完成")


def predict_news(text):
    """
    预测新闻类别

    Args:
        text: 新闻文本

    Returns:
        预测结果字典 {类别: 概率}
    """
    if not text.strip():
        return {"错误": "请输入新闻文本"}

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 移到GPU/CPU
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    # 转为字典 {类别: 概率}
    results = {}
    for i, prob in enumerate(probs.cpu().numpy()):
        category = label_map[i]
        results[category] = float(prob)

    return results


# ===== Gradio界面 =====

# 示例文本
examples = [
    ["中国男篮在世界杯上取得优异成绩，球迷欢呼雀跃"],
    ["股市今日大涨，上证指数突破3000点"],
    ["人工智能技术突破，深度学习应用广泛"],
    ["最新电影上映，票房大卖"]
]

# 创建界面
demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        label="输入新闻文本",
        placeholder="请输入一段新闻...",
        lines=5
    ),
    outputs=gr.Label(
        label="分类结果",
        num_top_classes=5  # 显示Top 5类别
    ),
    title="🗞️ 中文新闻分类系统",
    description="""
    基于BERT的中文新闻分类（14类）
    - 模型：bert-base-chinese微调
    - 准确率：96.99%
    - 数据集：THUCNews
    """,
    examples=examples,
    theme="default"
)

# 启动
if __name__ == "__main__":
    demo.launch(
        share=False,  # True会生成公开链接
        server_port=7860
    )