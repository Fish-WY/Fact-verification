# Fact-verification
Bert based FEVER project
## lupyne_create_index.py 
将 wiki 转化为 index

## convert_traindev.py 
对训练集插入句子内容以备 bert 训练  
检索测试集相关句子

## lupyne_retrieval_predict.py
包含检索相关函数  
和使用 AllenNLP 预训练模型预测 label的方法（旧版本）
## run_bert.ipython
使用 Bert 训练和预测结果
