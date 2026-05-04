import requests
import json
import pandas as pd
import time

# 服务器地址
base_url = "http://localhost:5000"

# 1. 文本嵌入请求示例
def test_embed(text_list):
    url = f"{base_url}/embed"
    
    # 准备要嵌入的文本列表
    payload = {
        "texts": text_list
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    # 发送POST请求
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    end_time = time.time()
    print(f"嵌入耗时: {end_time - start_time:.4f}秒")
    
    # 打印结果
    if response.status_code == 200:
        result = response.json()
        print("嵌入成功:")
        for i, embedding in enumerate(result["embeddings"]):
            print(f"句子 {i+1} 嵌入向量维度: {len(embedding)}")
            print(f"前10个维度值: {embedding[:100]}")
            print("-" * 50)
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)

# 2. 文本相似度计算请求示例
def test_similarity():
    url = f"{base_url}/similarity"
    
    # 准备源文本和目标文本
    payload = {
        "source": "专利相似度计算方法",
        "targets": [
            "一种基于词向量的专利相似度计算方法",
            "数据库查询方法与系统",
            "基于神经网络的专利检索技术",
            "一种计算文本相似度的方法"
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 发送POST请求
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # 打印结果
    if response.status_code == 200:
        result = response.json()
        print("相似度计算结果:")
        # 创建结果列表并按相似度排序
        similarity_results = []
        for i, target in enumerate(result["targets"]):
            similarity_results.append({
                "文本": target,
                "相似度": result["scores"][i]
            })
        
        # 按相似度降序排序
        similarity_results.sort(key=lambda x: x["相似度"], reverse=True)
        
        # 打印排序后的结果
        for item in similarity_results:
            print(f"文本: {item['文本']}")
            print(f"相似度: {item['相似度']:.4f}")
            print("-" * 50)
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)

# 3. 批量相似度计算（多个源文本与多个目标文本）
def batch_similarity_calculation():
    url = f"{base_url}/embed"
    
    # 准备所有文本
    source_texts = [
        "专利相似度计算方法",
        "自动驾驶技术"
    ]
    
    target_texts = [
        "一种基于词向量的专利相似度计算方法",
        "数据库查询方法与系统",
        "基于神经网络的专利检索技术",
        "一种计算文本相似度的方法",
        "无人驾驶汽车导航系统"
    ]
    
    # 所有文本合并后一起获取嵌入
    all_texts = source_texts + target_texts
    
    payload = {
        "texts": all_texts
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 发送POST请求获取所有嵌入
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        embeddings = result["embeddings"]
        
        # 获取源文本和目标文本的嵌入
        source_embeddings = embeddings[:len(source_texts)]
        target_embeddings = embeddings[len(source_texts):]
        
        # 计算相似度矩阵
        print("批量相似度计算结果:")
        for s_idx, source in enumerate(source_texts):
            print(f"源文本: {source}")
            source_embedding = source_embeddings[s_idx]
            
            # 计算与每个目标文本的相似度
            similarities = []
            for t_idx, target in enumerate(target_texts):
                target_embedding = target_embeddings[t_idx]
                
                # 计算余弦相似度 (简化计算)
                dot_product = sum(a*b for a, b in zip(source_embedding, target_embedding))
                magnitude_s = sum(a*a for a in source_embedding) ** 0.5
                magnitude_t = sum(b*b for b in target_embedding) ** 0.5
                similarity = dot_product / (magnitude_s * magnitude_t) if magnitude_s * magnitude_t != 0 else 0
                
                similarities.append({
                    "文本": target,
                    "相似度": similarity
                })
            
            # 按相似度降序排序
            similarities.sort(key=lambda x: x["相似度"], reverse=True)
            
            # 打印排序后的结果
            for item in similarities:
                print(f"  目标: {item['文本']}")
                print(f"  相似度: {item['相似度']:.4f}")
                print("  " + "-" * 48)
            print("=" * 50)
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # 读取专利数据
    # patent_df = pd.read_csv('patent_data_D_cleaned.csv')
    # patent_df = patent_df.fillna("")
    # patent_breif_1000 = patent_df['brief'][500:1200].values.tolist()
    patent_breif_1000 = ['今天天气真不错，真的真的很不错','']


    print("测试文本嵌入功能:")
    test_embed(patent_breif_1000)
    
    # print("\n" + "=" * 80 + "\n")
    

    # print("测试文本相似度计算功能:")
    # test_similarity()
    
    # print("\n" + "=" * 80 + "\n")
    
    # print("测试批量相似度计算:")
    # batch_similarity_calculation()
