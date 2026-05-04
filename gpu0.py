from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np

# 设置模型缓存路径
os.environ['MODELSCOPE_CACHE'] = '/workspace/patent/model'

# 检查是否有CUDA可用
cuda_available = torch.cuda.is_available()
device = 'cuda:0' if cuda_available else 'cpu'

# 加载模型
model_id = "iic/nlp_gte_sentence-embedding_chinese-large"
pipeline_se = pipeline(Tasks.sentence_embedding,
                      model=model_id,
                      sequence_length=512,
                      device=device)

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    if not data or 'texts' not in data:
        return jsonify({"error": "请提供'texts'字段，包含要嵌入的文本列表"}), 400
    
    texts = data['texts']
    try:
        if isinstance(texts, list):
            inputs = {
                "source_sentence": texts,
            }
            result = pipeline_se(input=inputs)
            
            # 将numpy数组转换为列表以便JSON序列化
            embeddings = result['text_embedding'].tolist()
            return jsonify({"embeddings": embeddings})
        else:
            return jsonify({"error": "texts必须是文本列表"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.json
    if not data or 'source' not in data or 'targets' not in data:
        return jsonify({"error": "请提供'source'和'targets'字段"}), 400
    
    source = data['source']
    targets = data['targets']
    
    try:
        inputs = {
            "source_sentence": [source],
            "sentences_to_compare": targets
        }
        result = pipeline_se(input=inputs)
        
        # 获取相似度得分
        scores = result['scores']
        
        return jsonify({
            "source": source,
            "targets": targets,
            "scores": scores
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"模型已加载，运行在{device}上")
    app.run(host='0.0.0.0', port=5000, debug=False)