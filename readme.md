# 学习神经网络

## 学习路径
1. 学习神经网络基础 - 全连接、CNN和变种、RNN和变种
2. 学习序列模型 - seq2seq、transformer，完成翻译任务
3. 把手写的transformer模型改造成LLM模型（还未完成）
4. 为这个LLM模型实现一个简单的推理引擎
5. 优化这个推理引擎，支持主流模型（比如Qwen 0.6B）

## Quick Start
环境准备：
```bash
uv sync
```

执行模型
```bash
# 跑seq2seq模型
uv run seq2seq

# 跑transformer模型（还未完成）
uv run timformer
```