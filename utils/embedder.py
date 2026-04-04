import asyncio
import aiohttp
from openai import AsyncOpenAI
from typing import List, Dict

class SiliconFlowEmbedder:
    def __init__(self, api_key: str, embedding_model: str = "BAAI/bge-m3", rerank_model: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化硅基流动 (SiliconFlow) 向量化与重排引擎
        """
        self.api_key = api_key
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        # 🌟 动态接收配置中的模型名称，彻底告别硬编码
        self.embedding_model = embedding_model 
        self.rerank_model = rerank_model
        
    # 🌟 优化：调大 batch_size 到 64，极大加快后台归一化时大量短词的计算速度
    async def get_embeddings(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        批量获取文本的向量表示 (带防超限保护)
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"[Embedder] 🧠 开始硅基流动向量化加工，共 {len(texts)} 个文本块，分为 {total_batches} 批次...")

        for i in range(0, len(texts), batch_size):
            # 🌟 优化：强制将空字符串替换为占位符，防止 API 因空输入报 400 错误直接崩溃
            batch_texts = [
                t[:2500] if isinstance(t, str) and t.strip() else "空白" 
                for t in texts[i : i + batch_size]
            ]
            current_batch_num = (i // batch_size) + 1
            
            try:
                response = await self.client.embeddings.create(
                    model=self.embedding_model, 
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # 严格按照传入文本的顺序排序
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                
                all_embeddings.extend(batch_embeddings)
                
                # 由于 batch_size 调大了，这里的提示只会稍微出现几次，避免刷屏
                if total_batches > 1:
                    print(f"[Embedder] ⏳ 批次 {current_batch_num}/{total_batches} 向量化完成。")
                
                if current_batch_num < total_batches:
                    await asyncio.sleep(0.5) 
                    
            except Exception as e:
                print(f"[Embedder] ❌ 批次 {current_batch_num} 向量化失败: {str(e)}")
                raise e

        print(f"[Embedder] ✅ 向量化加工完毕！成功生成 {len(all_embeddings)} 条向量。")
        return all_embeddings

    async def get_query_embedding(self, query: str) -> list[float]:
        """获取单个查询句子的向量"""
        try:
            # 强制将 query 包裹在列表中并加上长度保护
            safe_query = query[:2500] if query.strip() else "空白"
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=[safe_query],  
                encoding_format="float"
            )
            
            if response and response.data:
                return response.data[0].embedding
            return []
            
        except Exception as e:
            err_msg = f"[Embedder Error] {str(e)}"
            print(err_msg)
            raise Exception(err_msg)

    async def rerank(self, query: str, texts: List[str], top_n: int = 3) -> List[Dict]:
        """
        交叉注意力重排 (Rerank)
        """
        # 如果备选文本为空，直接返回，避免浪费网络请求
        if not texts or not query.strip():
            return []
            
        url = "https://api.siliconflow.cn/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.rerank_model,
            "query": query,
            "texts": texts,
            "top_n": top_n
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('results', [])
                    else:
                        # 🌟 优化：打印出真实的错误原因，比如“余额不足”、“模型不存在”等
                        error_text = await resp.text()
                        print(f"[Reranker Error] API 返回异常状态码 {resp.status}: {error_text}")
                        return []
        except Exception as e:
            print(f"[Reranker Error] 网络请求异常: {e}")
            return []