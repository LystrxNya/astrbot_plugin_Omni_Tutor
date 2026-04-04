import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_dir: str = "data/omnitutor_vdb"):
        """
        初始化双路并发架构的 ChromaDB 向量数据库
        """
        os.makedirs(persist_dir, exist_ok=True)
        
        # 初始化持久化客户端
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 🌟 集合 A：纯文本切片池 (Chunk Collection)
        self.chunks_collection = self.client.get_or_create_collection(
            name="tutor_chunks",
            metadata={"hnsw:space": "cosine"} 
        )
        
        # 🌟 集合 B：独立概念标签池 (Tag Collection)
        self.tags_collection = self.client.get_or_create_collection(
            name="tutor_tags",
            metadata={"hnsw:space": "cosine"} 
        )

    # ================= 集合 A: Chunk 存取操作 =================

    def add_chunks(self, chunks_data: List[Dict], source_file: str, ids: List[str], embeddings: List[List[float]] = None):
        """
        极简落盘：剥离所有复杂标签，仅保存向量、原文及溯源 ID
        """
        if not chunks_data: return

        documents = []
        metadatas = []

        for chunk in chunks_data:
            documents.append(chunk.get("text", ""))
            # metadata 暴瘦：只留一个 source 用于后续的按文件删除
            metadatas.append({"source": source_file})

        if embeddings:
            self.chunks_collection.upsert(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.chunks_collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def search_chunks(self, query: str, top_k: int = 15, query_embedding: List[float] = None) -> List[str]:
        """
        🌟 第二路召回 (盲捞)：在 Chunk 集合中做纯语义检索，直接返回匹配的 chunk_ids
        """
        query_params = {"n_results": top_k}
        
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        else:
            query_params["query_texts"] = [query]

        results = self.chunks_collection.query(**query_params)
        
        if results and results.get('ids') and results['ids'][0]:
            return results['ids'][0]
        return []

    # ================= 集合 B: Tag 存取操作 =================

    def add_tags(self, tags: List[str], embeddings: List[List[float]] = None):
        """
        填充概念池：将独立的标签加入 Tag 集合
        使用 tag 文本本身作为 ID，确保唯一性。upsert 会自动忽略已存在的标签或更新向量。
        """
        if not tags: return
        
        if embeddings:
            self.tags_collection.upsert(documents=tags, embeddings=embeddings, ids=tags)
        else:
            self.tags_collection.upsert(documents=tags, ids=tags)

    def search_tags(self, query: str, top_k: int = 3, query_embedding: List[float] = None) -> List[str]:
        """
        🌟 第一路召回 (精准路由)：拿用户问题去撞击 Tag 集合，找出语义最匹配的几个核心概念
        """
        query_params = {"n_results": top_k}
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        else:
            query_params["query_texts"] = [query]
            
        # 防止库里标签不够 top_k 导致报错，先获取当前标签总数
        tag_count = self.tags_collection.count()
        if tag_count == 0: return []
        query_params["n_results"] = min(top_k, tag_count)

        results = self.tags_collection.query(**query_params)
        
        if results and results.get('ids') and results['ids'][0]:
            return results['ids'][0] # ID 就是 Tag 的名字
        return []

    # ================= 运维管理与数据同步 =================

    def delete_by_source(self, source_file: str):
        """
        清理原文档对应的 Chunk 向量（Tag 向量因为是公共知识库的一部分，不随单一文件删除）
        """
        self.chunks_collection.delete(where={"source": source_file})

    def delete_tags(self, tags: List[str]):
        """
        从独立概念池中物理删除废弃的标签向量
        """
        if not tags: return
        try:
            self.tags_collection.delete(ids=tags)
        except Exception:
            pass

    def apply_normalization_mapping(self, course_map: dict, concept_map: dict) -> int:
        """
        🌟 史诗级减负：后台洗牌逻辑
        由于 Chunk 不再强绑定 Tag，这里只需要从概念池把旧的废弃 Tag 删掉即可！
        新的 Tag 会在后续对话/录入中自动被填入向量库。
        """
        if not concept_map: return 0
        
        old_tags = list(concept_map.keys())
        try:
            # 将被合并掉的历史词汇从独立概念池中抹除
            self.tags_collection.delete(ids=old_tags)
            return len(old_tags)
        except Exception:
            return 0