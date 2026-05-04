import argparse
import os
import numpy as np
from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch

from config import MODEL_CACHE_DIR

os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE_DIR


def create_app(pipeline_se):
    app = Flask(__name__)

    @app.route('/embed', methods=['POST'])
    def embed():
        data = request.json
        if not data or 'texts' not in data:
            return jsonify({"error": "请提供'texts'字段，包含要嵌入的文本列表"}), 400
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": "texts必须是文本列表"}), 400
        try:
            result = pipeline_se(input={"source_sentence": texts})
            return jsonify({"embeddings": result['text_embedding'].tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/similarity', methods=['POST'])
    def similarity():
        data = request.json
        if not data or 'source' not in data or 'targets' not in data:
            return jsonify({"error": "请提供'source'和'targets'字段"}), 400
        try:
            result = pipeline_se(input={
                "source_sentence": [data['source']],
                "sentences_to_compare": data['targets'],
            })
            return jsonify({"source": data['source'], "targets": data['targets'], "scores": result['scores']})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='专利向量化服务')
    parser.add_argument('--port', type=int, default=5000, help='监听端口')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"加载模型，运行在 {device}，端口 {args.port}")

    pipeline_se = pipeline(
        Tasks.sentence_embedding,
        model="iic/nlp_gte_sentence-embedding_chinese-large",
        sequence_length=512,
        device=device,
    )

    app = create_app(pipeline_se)
    app.run(host='0.0.0.0', port=args.port, debug=False)
