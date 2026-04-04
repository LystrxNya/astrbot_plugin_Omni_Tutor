import os
import asyncio
import uuid
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
from astrbot.api.all import logger  # 🌟 导入 AstrBot 官方日志器

try:
    import fitz  # PyMuPDF
except ImportError:
    pass # 依赖安装已在 ocr_engine 中处理

# 导入内部模块
from ..utils.ocr_engine import QwenOCREngine
from ..utils.embedder import SiliconFlowEmbedder
from .doc_processor import DocumentProcessor
from ..database.vector_store import VectorStore
from ..database.sqlite_manager import SQLiteManager
from ..utils.office_parser import parse_office_file
from .concept_tagger import ConceptTagger


class TutorBrain:
    def __init__(self, dashscope_key: str, siliconflow_key: str, 
                 ocr_model: str = "qwen-vl-ocr-2025-11-20", 
                 vision_model: str = "qwen-vl-max", 
                 fast_model: str = "qwen-plus",            # 快脑
                 reasoning_model: str = "deepseek-v3",     # 慢脑
                 embedding_model: str = "BAAI/bge-m3",                
                 rerank_model: str = "BAAI/bge-reranker-v2-m3",
                 persist_dir: str = "data/omnitutor_vdb", 
                 db_path: str = "data/omnitutor_sqlite.db"):
        """
        初始化导师中枢大脑，装配倒排索引双路召回架构的智能化零部件。
        """
        print("\n" + "="*50)
        print(f"[TutorBrain] 🧠 正在启动中枢系统 (VDB: {persist_dir})...")
        
        self.siliconflow_key = siliconflow_key
        
        # 将模型配置精细分发给底层引擎
        self.ocr = QwenOCREngine(api_key=dashscope_key, ocr_model=ocr_model, vision_model=vision_model)
        self.embedder = SiliconFlowEmbedder(api_key=siliconflow_key, embedding_model=embedding_model, rerank_model=rerank_model)
        self.processor = DocumentProcessor(chunk_size=800, overlap_size=150)
        
        # 将双脑模型参数透传给思考引擎
        self.tagger = ConceptTagger(
            dashscope_key=dashscope_key, 
            fast_model=fast_model, 
            reasoning_model=reasoning_model
        )
        
        # 存储中枢 (双集合 ChromaDB + 倒排索引 SQLite)
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.sql_manager = SQLiteManager(db_path=db_path)

        print(f"  ├── 👁️  双核视觉引擎: {ocr_model} + {vision_model} [就绪]")
        print(f"  ├── 🧬 向量与重排引擎: {embedding_model} + {rerank_model} [就绪]")
        print(f"  ├── 🏷️  认知思考双引擎: {fast_model} (快) + {reasoning_model} (慢) [就绪]")
        print(f"  └── 💾 存储中枢: 双路 ChromaDB + SQLite [就绪]")
        print("[TutorBrain] ✅ 系统环境装配完毕。")
        print("="*50 + "\n")

    # =========================================================================
    # 🌟 深度精读与思考闭环流水线
    # =========================================================================

    async def deep_read_and_summarize(self, files: list, raw_text: str, zoom: float = 3.0, progress_callback=None) -> str:
        """多文件解析提取 -> 思考引擎深度总结"""
        async def notify(msg):
            if progress_callback: await progress_callback(msg)

        combined_text = raw_text + "\n"
        
        for file_path, file_name in files:
            ext = file_path.lower().split('.')[-1]
            await notify(f"👁️ 可琳正在通读 [{file_name}] 的每一行字...")
            
            if ext in ['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'webp']:
                is_pdf = ext == 'pdf'
                total_pages = 1
                if is_pdf:
                    doc = fitz.open(file_path)
                    total_pages = min(len(doc), 15) # 限制精读页数防止超限
                    doc.close()
                
                for i in range(total_pages):
                    page_text = await self.ocr.process_file(file_path, page_num=i, zoom=zoom, enable_vision=False)
                    combined_text += f"\n{page_text}"
            else:
                text = await asyncio.to_thread(parse_office_file, file_path, ext)
                combined_text += f"\n{text}"
                
        if not combined_text.strip():
            return "⚠️ 资料里好像没有可以提取的有效文字呢..."
            
        await notify("🤔 资料研读完毕！可琳正在开启【深度思考模式】重构知识网络。大概需要十几秒，请主人耐心等待哦~")
        summary = await self.tagger.deep_summarize_with_thinking(combined_text)
        return summary

    # =========================================================================
    # 🌟 核心检索：双路召回 (Dual-Route Retrieval) + 倒排图谱路由
    # =========================================================================

    async def retrieve_knowledge(self, query: str, uid: str, top_k: int = 5) -> str:
        """
        新纪元检索流：意图裂变 -> 标签向量池秒寻 -> 第一路图谱召回 -> 第二路语义盲捞 -> 智能安全截断 -> 重排缝合
        """
        try:
            logger.info(f"🚀 [TutorBrain] 启动双脑智能倒排检索流...")
            
            q_emb = await self.embedder.get_query_embedding(query)
            all_recall_results = []
            seen_ids = set()

            # 🌟 步 1: 慢脑动态意图裂变 (裂变出可能的 Concepts 和 Pedagogy Type)
            intents = await self.tagger.analyze_multi_intent(query)
            
            # 🌟 步 2: 第一路召回 (图谱路由 - 核心：由于这是一个 Concept 集合，往往会召回大量切片)
            if intents:
                all_guessed_concepts = list(set([c for intent in intents for c in intent.get("concepts", [])]))
                matched_tags = []
                
                for gc in all_guessed_concepts:
                    g_emb = await self.embedder.get_query_embedding(gc)
                    found_tags = self.vector_store.search_tags(query=gc, top_k=2, query_embedding=g_emb)
                    matched_tags.extend(found_tags)
                
                matched_tags = list(set(matched_tags))
                if matched_tags:
                    logger.info(f"🎯 [TutorBrain] 第一路(概念路由)命中图谱标签: {matched_tags}")
                    route_1_chunks = self.sql_manager.get_chunks_by_tags(matched_tags)
                    all_recall_results.extend(route_1_chunks) # ⚠️ 此处可能引入数百个切片
            
            # 🌟 步 3: 第二路召回 (纯文本语义盲捞 - 确保语义相关性底座)
            logger.info(f"🔍 [TutorBrain] 第二路(全库盲捞)启动语义深潜...")
            blind_chunk_ids = self.vector_store.search_chunks(query, top_k=15, query_embedding=q_emb)
            if blind_chunk_ids:
                route_2_chunks = self.sql_manager.get_chunks_by_ids(blind_chunk_ids)
                all_recall_results.extend(route_2_chunks) # 🌟 此时 all_recall_results 最后 15 个是语义盲捞结果

            # 物理去重合并
            unique_results = []
            for r in all_recall_results:
                if r['chunk_id'] not in seen_ids:
                    unique_results.append(r)
                    seen_ids.add(r['chunk_id'])

            if not unique_results:
                logger.warning(f"⚠️ [TutorBrain] 双路召回均为空。")
                return "【无匹配资料】系统未在知识库中找到高相关度的学术切片，请基于你的内置知识进行解答。"

            # 🌟 步 3.5: 教学类型 (Pedagogy Type) 智能软过滤
            allowed_types = set([t for intent in intents for t in intent.get("pedagogy_type", [])])
            if allowed_types:
                type_filtered_results = []
                for r in unique_results:
                    chunk_type_str = r['metadata'].get('pedagogy_type', "")
                    if any(at in chunk_type_str for at in allowed_types):
                        type_filtered_results.append(r)
                
                if type_filtered_results:
                    unique_results = type_filtered_results
                else:
                    logger.warning(f"⚠️ [TutorBrain] 教学类型过滤导致结果为空，已自动降级放宽限制。")

            # ==========================================
            # 🌟 核心修复：智能安全截断保护阀 (Safety Valve)
            # ==========================================
            MAX_CANDIDATES = 60 # 设定 BGE-Reranker 的负载极限
            if len(unique_results) > MAX_CANDIDATES:
                logger.warning(f"⚠️ [TutorBrain] 候选池爆炸 ({len(unique_results)} 个)，触发安全截断保护！")
                
                # 策略：强制保送 Route 2 的 15 个语义最相关结果
                # 因为 Route 2 是最后加入全集的，所以通过索引后段截取
                route_2_count = min(15, len(unique_results))
                route_2_safe = unique_results[-route_2_count:]
                
                # 剩余名额从庞大的图谱库(Route 1)中随机采样，保证知识广度
                route_1_pool = unique_results[:-route_2_count]
                
                import random
                sample_size = MAX_CANDIDATES - route_2_count
                if len(route_1_pool) > sample_size:
                    route_1_safe = random.sample(route_1_pool, sample_size)
                else:
                    route_1_safe = route_1_pool
                
                unique_results = route_1_safe + route_2_safe
                logger.info(f"✅ [TutorBrain] 截断成功。保留 15 个语义保底 + {len(route_1_safe)} 个图谱采样切片。")

            # ==========================================
            # 🌟 步 4: BGE 交叉注意力重排 (Rerank)
            # ==========================================
            logger.info(f"秤 [TutorBrain] 准备对 {len(unique_results)} 个精选候选切片进行重排打分...")
            texts = [res['text'] for res in unique_results]
            rerank_results = await self.embedder.rerank(query, texts, top_n=top_k)
            
            final_chunks = []
            if rerank_results:
                for r in rerank_results:
                    # 设定相关度阈值，防止把完全无关的内容喂给 LLM
                    if r['relevance_score'] > 0.05: 
                        chunk = unique_results[r['index']]
                        chunk['score'] = r['relevance_score']
                        final_chunks.append(chunk)
            else: 
                final_chunks = unique_results[:top_k]

            # 🌟 步 5: 动态智能拼图 (Stitching)
            context_output = "【全知导师专属理论回放 (包含无缝拼图的完整上下文)】\n"
            for res in final_chunks:
                chunk_id = res['chunk_id']
                meta = res['metadata']
                score = f"{res.get('score', 0):.1%}"
                text_to_show = res['text']
                
                # 检查边界，如果切片不完整，回表查询前后各1个切片进行缝合
                if meta.get("boundary_status") in ["无头无尾", "章节转换"] or meta.get("context_loss") == 1:
                    logger.info(f"🧩 [TutorBrain] 触发智能拼图 [Chunk ID: {chunk_id[-6:]}]")
                    stitched_text = self.sql_manager.get_surrounding_context_by_id(chunk_id, window=1)
                    if stitched_text:
                        text_to_show = stitched_text

                context_output += f"\n📍 [来源: {meta.get('source')} | 学科: {meta.get('course')}] (匹配度: {score})\n"
                context_output += f"--- 知识切片 ---\n{text_to_show}\n"

            return context_output

        except Exception as e:
            import traceback
            logger.error(f"❌ [TutorBrain] 检索流崩溃: {traceback.format_exc()}")
            return f"❌ 检索失败: {e}"

    # =========================================================================
    # 🌟 录入学习流 (实时双池分路落盘 + 洗牌归一化版)
    # =========================================================================

    async def _run_background_normalization(self, new_courses: set, new_concepts: set, callback):
        """后台静默洗牌与归一化守护任务"""
        try:
            existing_courses = self.sql_manager.get_all_unique_courses()
            existing_concepts = self.sql_manager.get_all_unique_concepts()
            
            course_map = await self.tagger.cluster_words(
                list(new_courses), existing_courses, self.embedder, 
                direct_threshold=0.95, ai_threshold=0.90
            )
            
            concept_map = await self.tagger.cluster_words(
                list(new_concepts), existing_concepts, self.embedder, 
                direct_threshold=0.92, ai_threshold=0.80
            )

            course_map = {k: v for k, v in course_map.items() if k != v}
            concept_map = {k: v for k, v in concept_map.items() if k != v}

            # 双轨修改
            sqlite_merged = self.sql_manager.apply_normalization_mapping(course_map, concept_map)
            vdb_updated = self.vector_store.apply_normalization_mapping(course_map, concept_map)
            
            merged_count = len(course_map) + len(concept_map)
            if merged_count > 0:
                logger.info(f"🎉 [OmniTutor-洗牌] 后台图谱归一化完成！合并了 {merged_count} 个概念。从向量图谱摘除了 {vdb_updated} 个废弃节点。")
                if callback:
                    await callback(
                        f"🎉 【后台图谱归一化完成】\n"
                        f"可琳已将 {merged_count} 个同义词合并！图谱网络更加纯净了呢~"
                    )
            else:
                logger.info(f"♻️ [OmniTutor-洗牌] 后台图谱检查完毕，本次无新概念合并。")
        except Exception as e:
            import traceback
            logger.error(f"❌ [后台洗牌异常] {traceback.format_exc()}")

    async def learn_from_file(self, file_path: str, source_name: str, zoom: float = 3.0, progress_callback=None) -> Tuple[str, str]:
        """全模态流式极速录入流水线"""
        start_time = time.time()
        logger.info(f"[TutorBrain] 📚 开始流式研读与分轨入库: [{source_name}]")
        
        async def notify(msg):
            if progress_callback: await progress_callback(msg)

        try:
            full_markdown_text = ""
            ext = file_path.lower().split('.')[-1]
            visual_exts = ['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'webp']
            office_exts = ['txt', 'md', 'csv', 'xlsx', 'xls', 'docx', 'pptx']
            
            # ================= 1. 物理切片提取 =================
            if ext in visual_exts or ext == 'pdf':
                is_pdf = ext == 'pdf'
                total_pages = 1
                if is_pdf:
                    pdf_doc = fitz.open(file_path)
                    total_pages = len(pdf_doc)
                    pdf_doc.close()
                
                await notify(f"👁️ 正在启动多线程视觉引擎识别 ({total_pages} 页)...")
                ocr_sem = asyncio.Semaphore(3)
                
                async def fetch_page_ocr(page_idx):
                    await asyncio.sleep(0.1 * (page_idx % 3)) 
                    async with ocr_sem:
                        return page_idx, await self.ocr.process_file(file_path, page_num=page_idx, zoom=zoom, enable_vision=False)
                
                ocr_tasks = [fetch_page_ocr(i) for i in range(total_pages)]
                pages_results = await asyncio.gather(*ocr_tasks)
                pages_results.sort(key=lambda x: x[0])

                current_cache = ""
                for idx, page_text in pages_results:
                    if not page_text: continue
                    if not current_cache: current_cache = page_text; continue
                        
                    if current_cache in page_text: current_cache = page_text
                    elif page_text in current_cache: continue
                    else:
                        if SequenceMatcher(None, current_cache, page_text).ratio() > 0.85:
                            if len(page_text) > len(current_cache): current_cache = page_text
                        else:
                            full_markdown_text += "\n\n" + current_cache
                            current_cache = page_text
                if current_cache: full_markdown_text += "\n\n" + current_cache

            elif ext in office_exts:
                await notify(f"📄 正在解析 Office 文档...")
                full_markdown_text = await asyncio.to_thread(parse_office_file, file_path, ext)

            if not full_markdown_text.strip():
                return f"❌ 研读失败：未能在 [{source_name}] 中识别到有效内容。", ""

            chunks_dict_list = self.processor.process(full_markdown_text)
            pure_texts = [c["text"] for c in chunks_dict_list]
            chunk_ids = [uuid.uuid4().hex for _ in pure_texts]
            
            await notify(f"✂️ 物理分块完成，共计 {len(pure_texts)} 个片段。")

            # ================= 2. 极速提取原生标签 =================
            await notify(f"🏷️ 正在开启满血并发打标，提取原生概念...")
            
            final_chunks_data = []
            new_courses_set = set()
            new_concepts_set = set()
            
            tag_sem = asyncio.Semaphore(200)

            async def process_single_chunk(text_chunk):
                async with tag_sem:
                    return await self.tagger.extract_chunk_tags(text_chunk)

            tasks = [process_single_chunk(t) for t in pure_texts]
            raw_data_list = await asyncio.gather(*tasks)

            for text, raw_data in zip(pure_texts, raw_data_list):
                course = raw_data.get("tags", {}).get("course", "通用")
                concepts = raw_data.get("tags", {}).get("concepts", [])

                self.sql_manager.update_concept_pool(course, concepts)
                new_courses_set.add(course)
                new_concepts_set.update(concepts)
                
                raw_data["tags"]["course"] = course
                raw_data["tags"]["concepts"] = concepts
                raw_data["text"] = text
                final_chunks_data.append(raw_data)
            
            logger.info(f" ├── 🏷️ 打标完成！处理了 {len(pure_texts)} 个文本块！收集到 {len(new_concepts_set)} 个新概念。")
        
            # ================= 3. 双轨落盘分离 (图谱与切片) =================
            await notify(f"🧬 标签提取完成，正在生成向量池并建立倒排索引...")
            
            # 3.1 独立写入 Tags 向量池
            if new_concepts_set:
                tags_list = list(new_concepts_set)
                tag_embeddings = await self.embedder.get_embeddings(tags_list)
                self.vector_store.add_tags(tags_list, tag_embeddings)

            # 3.2 独立写入 Chunk 文本向量池
            embeddings = await self.embedder.get_embeddings(pure_texts)
            self.vector_store.add_chunks(final_chunks_data, source_name, chunk_ids, embeddings)
            
            # 3.3 写入 SQLite 关系库，自动建立多对多桥梁映射
            self.sql_manager.save_document(source_name, full_markdown_text, final_chunks_data, chunk_ids)

            # ================= 4. 启动后台异步清洗任务 =================
            asyncio.create_task(self._run_background_normalization(new_courses_set, new_concepts_set, progress_callback))
            
            cost = round(time.time() - start_time, 1)
            msg = (
                f"✅ 资料已极速物理入库！\n"
                f"📄 来源：{source_name}\n"
                f"🎯 提炼切片：{len(final_chunks_data)} 个\n"
                f"⏱️ 耗时：{cost} 秒\n\n"
                f"(可琳正在后台为您洗牌与归一化，您可以继续发送下一份资料啦~)"
            )
            return msg, full_markdown_text

        except Exception as e:
            import traceback
            logger.error(f"[TutorBrain Error] {traceback.format_exc()}")
            return f"❌ 学习失败: {str(e)}", ""

    async def learn_from_text(self, raw_text: str, instruction: str, source_label: str, progress_callback=None) -> str:
        """对话回流学习"""
        start_time = time.time()
        async def notify(msg):
            if progress_callback: await progress_callback(msg)

        try:
            await notify("🔍 助理模型正在提取纯净学术内容...")
            clean_text = await self.tagger.extract_academic_content(raw_text, instruction)
            
            if not clean_text or len(clean_text) < 10:
                return "⚠️ 提取内容过少或未发现有效学术信息，已取消存入。"

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_stamped_text = f"【记录时间：{current_time}】\n{clean_text}"
            source_name = f"{source_label}_{current_time}"

            chunks_dict_list = self.processor.process(time_stamped_text)
            pure_texts = [c["text"] for c in chunks_dict_list]
            chunk_ids = [uuid.uuid4().hex for _ in pure_texts]
            
            await notify(f"🏷️ 提取完成，正在进行极速打标...")
            
            final_chunks_data = []
            new_courses_set = set()
            new_concepts_set = set()

            for text in pure_texts:
                raw_data = await self.tagger.extract_chat_tags(text)
                course = raw_data.get("tags", {}).get("course", "通用")
                concepts = raw_data.get("tags", {}).get("concepts", [])

                self.sql_manager.update_concept_pool(course, concepts)
                new_courses_set.add(course)
                new_concepts_set.update(concepts)

                raw_data["tags"]["course"] = course
                raw_data["tags"]["concepts"] = concepts
                raw_data["text"] = text
                final_chunks_data.append(raw_data)

            # 同步分离双落盘
            if new_concepts_set:
                tags_list = list(new_concepts_set)
                tag_embeddings = await self.embedder.get_embeddings(tags_list)
                self.vector_store.add_tags(tags_list, tag_embeddings)

            embeddings = await self.embedder.get_embeddings(pure_texts)
            self.vector_store.add_chunks(final_chunks_data, source_name, chunk_ids, embeddings)
            self.sql_manager.save_document(source_name, time_stamped_text, final_chunks_data, chunk_ids)
            
            asyncio.create_task(self._run_background_normalization(new_courses_set, new_concepts_set, progress_callback))
            
            return f"✅ 回话内容已极速内化至知识库！\n🎯 记忆锚点：{len(final_chunks_data)} 个\n⏱️ 耗时：{round(time.time()-start_time, 1)}s\n(后台正在静默归一化图谱)"
            
        except Exception as e:
            import traceback
            logger.error(f"[TutorBrain Text-Learn Error] {traceback.format_exc()}")
            return f"❌ 存入失败: {e}"

    # =========================================================================
    # 副脑 MIU 碎片操作
    # =========================================================================

    async def ingest_important_info(self, raw_text: str = "", file_path: str = "", source_label: str = "手动录入", zoom: float = 3.0, progress_callback=None) -> Tuple[str, str]:
        """极速碎片化录入"""
        start_time = time.time()
        async def notify(msg):
            if progress_callback: await progress_callback(msg)

        try:
            full_text = raw_text
            if file_path:
                ext = file_path.lower().split('.')[-1]
                if ext in ['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'webp']:
                    await notify(f"👁️ 正在启动视觉引擎识别文件...")
                    full_text = await self.ocr.process_file(file_path, page_num=0, zoom=zoom, enable_vision=False)
                else:
                    await notify(f"📄 正在解析办公文档...")
                    full_text = await asyncio.to_thread(parse_office_file, file_path, ext)
                    
            if not full_text.strip(): return "⚠️ 未能提取到有效文本内容。",""

            await notify("✂️ 助理模型正在提取最小信息单元 (MIU) 并生成概括条目...")
            mius = await self.tagger.extract_miu(full_text)
            
            if not mius: return "⚠️ 模型未能成功提取出任何信息单元。",""

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            source_name = f"{source_label}_{current_time}"
            
            titles = [m['title'] for m in mius]
            contents = [m['content'] for m in mius]
            chunk_ids = [uuid.uuid4().hex for _ in mius]
            
            await notify(f"🧬 正在对 {len(titles)} 个条目进行高纯度语义向量化...")
            embeddings = await self.embedder.get_embeddings(titles)
            
            mock_chunks = [{"text": t, "tags": {}, "pedagogy": {}, "boundary": {}} for t in titles]
            self.vector_store.add_chunks(mock_chunks, source_name, chunk_ids, embeddings)
            
            for cid, title, content in zip(chunk_ids, titles, contents):
                self.sql_manager.save_important_miu(cid, title, content, source_name)
                logger.info(f"  ├── 📦 已落盘碎片: [{title}]")

            cost = round(time.time() - start_time, 1)
            return f"✅ 碎片化内化完毕！\n🎯 提炼条目：{len(titles)} 个\n⏱️ 耗时：{cost}s", full_text
            
        except Exception as e:
            logger.error(f"❌ 碎片入库失败: {e}")
            return f"❌ 碎片入库失败: {e}", ""

    async def query_important_miu(self, query: str, top_k: int = 5) -> str:
        """父子文档检索：向量标题 -> SQLite 完整文"""
        try:
            logger.info(f"\n[ImportantBrain] 🚀 启动 MIU 碎片查询...")
            q_emb = await self.embedder.get_query_embedding(query)
            if not q_emb: return "❌ 向量化查询失败。"
            
            # 🌟 修正：调用分离后的 search_chunks 并直接获取 IDs
            chunk_ids = self.vector_store.search_chunks(query, top_k=top_k, query_embedding=q_emb)
            if not chunk_ids: return "📭 重要信息库中未检索到相关内容。"
                
            context = ""
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for cid in chunk_ids:
                miu_data = self.sql_manager.get_important_miu(cid)
                if miu_data:
                    context += f"\n--- [条目: {miu_data['title']} | 来源: {miu_data['source']}] ---\n{miu_data['content']}\n"
                    
            prompt = f"""
            你是一个严格的文件查阅助手。请完全且仅基于参考资料回答问题。
            【当前时间】：{current_time}
            【规则】：客观提取。没有就说没有，严禁编造。不要闲聊。
            【用户问题】：{query}
            【参考资料】：{context}
            """
            
            answer = await self.tagger._call_llm_text(prompt, target_model=self.tagger.fast_model)
            return f"🌟 【碎片库检索结果】\n\n{answer}"
            
        except Exception as e:
            logger.error(f"❌ 碎片查询失败: {e}")
            return f"❌ 碎片查询失败: {e}"

    # =========================================================================
    # 杂项方法与删除路由
    # =========================================================================

    def forget_file(self, keyword: str) -> str:
        """清空主脑记忆，并触发图谱网络自适应缩容"""
        all_sources = self.sql_manager.get_all_source_names()
        target = None
        
        if keyword in all_sources:
            target = keyword
        else:
            matched = [s for s in all_sources if keyword in s]
            if not matched: return f"⚠️ 找不到包含【{keyword}】的记录。"
            if len(matched) > 1:
                return "⚠️ 触发安全拦截！匹配多条记录，请提供更精确的名称：\n" + "\n".join([f"🔸 {s}" for s in matched])
            target = matched[0]
            
        # 1. 触发级联删除（物理切片 + 映射桥梁自动消亡）
        self.vector_store.delete_by_source(target)
        self.sql_manager.delete_document(target)
        
        # 2. 🌟 触发扫地僧机制：清理孤立图谱节点并重新计算全局热度
        orphans = self.sql_manager.clean_and_recalc_concepts()
        if orphans:
            self.vector_store.delete_tags(orphans)
            
        msg = f"🗑️ 成功精确删除了文件：[{target}]\n"
        if orphans:
            msg += f"🧹 图谱已自动缩容，摘除了 {len(orphans)} 个失效概念节点。"
        else:
            msg += f"🔄 图谱词频已完成重新校准。"
            
        return msg
    
    def forget_miu(self, keyword: str) -> str:
        """清空副脑记忆"""
        all_mius = self.sql_manager.get_all_mius()
        if not all_mius: return "📭 碎片库为空。"
        
        titles = [m['title'] for m in all_mius]
        
        if keyword in titles:
            chunk_ids = self.sql_manager.delete_miu_by_title(keyword)
            if chunk_ids: self.vector_store.chunks_collection.delete(ids=chunk_ids) # 🌟 修正集合调用
            return f"🗑️ 成功精准删除了条目：[{keyword}]"
            
        matched_titles = list(set([t for t in titles if keyword in t]))
        if len(matched_titles) == 1:
            target = matched_titles[0]
            chunk_ids = self.sql_manager.delete_miu_by_title(target)
            if chunk_ids: self.vector_store.chunks_collection.delete(ids=chunk_ids) # 🌟 修正集合调用
            return f"🗑️ 模糊匹配成功，已删除唯一条目：[{target}]"
        elif len(matched_titles) > 1:
            return "⚠️ 触发安全拦截！请提供更精确的条目名：\n" + "\n".join([f"🔸 {t}" for t in matched_titles])
                    
        sources = list(set([m['source'] for m in all_mius]))
        matched_sources = [s for s in sources if keyword in s]
        if len(matched_sources) == 1:
            target_source = matched_sources[0]
            self.vector_store.delete_by_source(target_source)
            self.sql_manager.delete_document(target_source)
            return f"🗑️ 已连锅端删除整个批次记录：[{target_source}]"
        elif len(matched_sources) > 1:
            return f"⚠️ 匹配到了多个批次，请提供更准确的批次时间戳。"
            
        return f"⚠️ 找不到包含关键词【{keyword}】的记录。"
    
    async def clean_entire_graph(self, progress_callback=None) -> str:
        """全库手动深度自清洗 (全盘碎片整理)"""
        try:
            if progress_callback:
                await progress_callback("🔍 开始全库扫描，拉取所有图谱节点...")

            all_courses = self.sql_manager.get_all_unique_courses()
            all_concepts = self.sql_manager.get_all_unique_concepts()

            if not all_concepts:
                return "📭 当前知识图谱为空，无需清洗。"

            if progress_callback:
                await progress_callback(f"🧬 共发现 {len(all_courses)} 个学科和 {len(all_concepts)} 个概念节点，准备启动折叠算法进行全量去重...")

            final_course_map = {}
            final_concept_map = {}

            # ==========================================
            # 🌟 核心坍缩逻辑：对全库概念进行递进式归一化
            # ==========================================
            current_base = []
            for idx, c in enumerate(all_concepts):
                if not current_base:
                    current_base.append(c)
                    continue
                
                # 拿着这一个词，去和目前已经确立的新基准比较
                cmap = await self.tagger.cluster_words(
                    [c], current_base, self.embedder, 
                    direct_threshold=0.92, ai_threshold=0.80
                )
                target = cmap.get(c, c)
                if target != c:
                    final_concept_map[c] = target # 找到了大哥，记录融合映射
                else:
                    current_base.append(c) # 开山立派，成为新的参照基准
                
                # 进度播报 (每处理 30 个播报一次，防止刷屏)
                if progress_callback and (idx + 1) % 30 == 0:
                    await progress_callback(f"⏳ 正在深度清洗中... (已扫描 {idx + 1}/{len(all_concepts)} 个节点)")
                
                await asyncio.sleep(0.05) # 微小延迟，保护 API 不被熔断

            # ==========================================
            # 🌟 对学科 (Course) 也进行一次洗牌
            # ==========================================
            current_course_base = []
            for c in all_courses:
                if not current_course_base:
                    current_course_base.append(c)
                    continue
                cmap = await self.tagger.cluster_words(
                    [c], current_course_base, self.embedder, 
                    direct_threshold=0.95, ai_threshold=0.90
                )
                target = cmap.get(c, c)
                if target != c:
                    final_course_map[c] = target
                else:
                    current_course_base.append(c)

            # ==========================================
            # 🌟 落地修改与数据重构
            # ==========================================
            if not final_concept_map and not final_course_map:
                return "✨ 全库深度扫描完毕！图谱非常纯净，没有发现需要合并的同义词呢~"

            if progress_callback:
                await progress_callback("💾 洗牌计算完成，正在执行底层数据库重构与映射重写...")

            # 双轨落盘修改
            self.sql_manager.apply_normalization_mapping(final_course_map, final_concept_map)
            vdb_updated = self.vector_store.apply_normalization_mapping(final_course_map, final_concept_map)

            # 🌟 强制批量执行重连桥梁，拯救大批孤儿切片！
            self._force_reconnect_mappings(final_concept_map)

            merged_count = len(final_course_map) + len(final_concept_map)
            return f"🎉 【全库自清洗完成】\n可琳成功将 {merged_count} 个同义词进行了融合坍缩！\n(已重连全部底层切片羁绊，并摘除了 {vdb_updated} 个冗余向量，知识网络更干练了哦~)"

        except Exception as e:
            import traceback
            from astrbot.api.all import logger
            logger.error(f"❌ [全库自清洗异常] {traceback.format_exc()}")
            return f"❌ 清洗过程发生错误: {e}"
        
    # =========================================================================
    # 🌟 知识图谱“手术刀”指令集
    # =========================================================================

    async def manual_merge_concepts(self, old_name: str, new_name: str) -> str:
        """【手术刀 1】强行将 A 概念合并到 B 概念 (手动对齐)"""
        if old_name == new_name: return "⚠️ 名字一样，不需要合并哦。"
        
        # 1. 构造映射字典
        concept_map = {old_name: new_name}
        
        # 2. 调用现有的归一化落盘逻辑
        sqlite_updated = self.sql_manager.apply_normalization_mapping({}, concept_map)
        vdb_updated = self.vector_store.apply_normalization_mapping({}, concept_map)
        
        if sqlite_updated > 0:
            return f"✅ 合并成功！已将 【{old_name}】 彻底并入 【{new_name}】。\n影响了 {sqlite_updated} 个关联切片，删除了 {vdb_updated} 个冗余向量节点。"
        else:
            return f"❌ 合并失败：在库中没找到名为 【{old_name}】 的概念。"

    async def clean_specific_concept(self, concept_name: str, progress_callback=None) -> str:
        """【手术刀 2】针对性清洗：让指定概念重新去库里找一遍“大哥”"""
        all_concepts = self.sql_manager.get_all_unique_concepts()
        if concept_name not in all_concepts:
            return f"❌ 库里没有 【{concept_name}】 这个概念。"
            
        # 剔除自己作为参照物
        base_concepts = list(set(all_concepts) - {concept_name})
        if not base_concepts: return "📭 库里只有这一个概念，没法清洗呢。"

        if progress_callback: await progress_callback(f"🔍 正在为 【{concept_name}】 寻找最匹配的归宿...")
        
        # 调用聚类引擎
        cmap = await self.tagger.cluster_words([concept_name], base_concepts, self.embedder)
        target = cmap.get(concept_name, concept_name)
        
        if target != concept_name:
            return await self.manual_merge_concepts(concept_name, target)
        else:
            return f"✨ 清洗完毕：【{concept_name}】 表现很正统，目前不需要合并到其他概念。"

    async def reprocess_unsuitable_concept(self, concept_name: str, progress_callback=None) -> str:
        """【手术刀 3 满血版】重塑映射：并发呼叫大模型重写，批量安全重连图谱"""
        # 1. 拉取该概念下的所有物理切片
        chunks = self.sql_manager.get_chunks_by_tags([concept_name])
        if not chunks: return f"❌ 找不到属于 【{concept_name}】 的切片数据。"
        
        if progress_callback: 
            await progress_callback(f"☢️ 警告：检测到错乱概念 【{concept_name}】，提取出 {len(chunks)} 个切片。")
            await progress_callback(f"🚀 正在启动满血并发引擎，全火力请求大模型进行重写 (最大并发: 100)，请稍候...")

        # ==========================================
        # 🌟 阶段一：高并发请求 LLM (利用 Semaphore 控制最大 100 并发)
        # ==========================================
        sem = asyncio.Semaphore(100)
        
        async def process_single_chunk(idx, chunk):
            old_text = chunk['text']
            old_id = chunk['chunk_id']
            old_source = chunk.get('metadata', {}).get('source', f"重塑归档_{concept_name}")
            
            async with sem:
                # 并发呼叫专属重塑方法
                reprocessed_data = await self.tagger.reprocess_and_tag_chunk(old_text, bad_concept=concept_name)
            
            return idx, chunk, old_id, old_source, reprocessed_data

        # 一次性把所有任务扔进事件循环，等待它们同时跑完
        tasks = [process_single_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # 按照原来的顺序排好队
        results.sort(key=lambda x: x[0])

        if progress_callback: 
            await progress_callback(f"💾 大模型全并发重写完毕！正在进行底层数据库的高速安全重构...")

        # ==========================================
        # 🌟 阶段二：内存极速处理与顺序落盘 (保护 SQLite 免受死锁)
        # ==========================================
        success_count = 0
        new_courses_set = set()
        new_concepts_set = set()

        for idx, chunk, old_id, old_source, reprocessed_data in results:
            old_text = chunk['text']
            clean_text = reprocessed_data.get("rewritten_text", old_text)
            course = reprocessed_data.get("tags", {}).get("course", "通用")
            concepts = reprocessed_data.get("tags", {}).get("concepts", [])

            # 保险锁：过滤旧错词
            concepts = [c for c in concepts if c != concept_name]
            if not concepts: concepts = ["待分类知识"]
            reprocessed_data["tags"]["concepts"] = concepts

            # 更新集合用于后续的后台洗牌
            self.sql_manager.update_concept_pool(course, concepts)
            new_courses_set.add(course)
            new_concepts_set.update(concepts)

            # 斩断旧桥梁 (🌟 这里已经用上了你刚才修复的 chunk_id)
            with self.sql_manager._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM chunk_tag_mapping WHERE chunk_id = ?", (old_id,))
                cur.execute("DELETE FROM chunks WHERE chunk_id = ?", (old_id,))
                conn.commit()
                
            self.vector_store.chunks_collection.delete(ids=[old_id])
            
            # 注入新生命
            new_id = uuid.uuid4().hex
            reprocessed_data["text"] = clean_text
            
            # 安全顺序写入 SQLite
            self.sql_manager.save_document(old_source, f"【重写归档】\n{clean_text}", [reprocessed_data], [new_id])
            
            # 向量化并加入 ChromaDB
            embeddings = await self.embedder.get_embeddings([clean_text])
            self.vector_store.add_chunks([reprocessed_data], old_source, [new_id], embeddings)

            success_count += 1
            
        # ==========================================
        # 🌟 收尾：同步大图谱与后台归一化
        # ==========================================
        if new_concepts_set:
            tags_list = list(new_concepts_set)
            tag_embeddings = await self.embedder.get_embeddings(tags_list)
            self.vector_store.add_tags(tags_list, tag_embeddings)
            
        # 彻底从世界树上抹除那个罪恶的旧概念节点
        self.sql_manager.delete_concept_by_name(concept_name)
        self.vector_store.delete_tags([concept_name])
        
        # 触发后台洗牌，让新生词汇融入网络
        asyncio.create_task(self._run_background_normalization(new_courses_set, new_concepts_set, None))
        
        return f"🔥 并发重塑任务圆满完成！\n已将属于 【{concept_name}】 的 {success_count} 个切片以全并发模式重写完毕。它们已经重连回原本的档案中，旧概念节点彻底抹除！"
    
    def _force_reconnect_mappings(self, concept_map: dict):
        """🌟 核心修复：强制重连底层的多对多桥接表，确保物理切片不迷路"""
        if not concept_map: return
        try:
            with self.sql_manager._get_conn() as conn:
                cur = conn.cursor()
                
                # 动态探查你的表结构（兼容字段名叫 concept 还是 tag 还是 tag_name）
                cur.execute("PRAGMA table_info(chunk_tag_mapping)")
                columns = [col[1] for col in cur.fetchall()]
                if not columns: return # 表不存在则跳过
                tag_col = "concept" if "concept" in columns else "tag" if "tag" in columns else "tag_name"
                
                for old_name, new_name in concept_map.items():
                    # 1. 拷贝继承：把拥有老概念的切片，全部复制一份映射给新概念
                    # (使用 INSERT OR IGNORE 防止该切片本来就拥有新概念，导致主键冲突)
                    cur.execute(f"""
                        INSERT OR IGNORE INTO chunk_tag_mapping (chunk_id, {tag_col})
                        SELECT chunk_id, ? FROM chunk_tag_mapping WHERE {tag_col} = ?
                    """, (new_name, old_name))
                    
                    # 2. 斩草除根：彻底抹除老概念的旧桥梁
                    cur.execute(f"DELETE FROM chunk_tag_mapping WHERE {tag_col} = ?", (old_name,))
                
                conn.commit()
        except Exception as e:
            from astrbot.api.all import logger
            logger.error(f"❌ 强制重连图谱桥梁失败: {e}")