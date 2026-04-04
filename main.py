import os
import asyncio
import sys
import math
import uuid
from datetime import datetime
import subprocess

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import astrbot.core.message.components as Comp
from astrbot.api.provider import ProviderRequest
from astrbot.api.all import logger

# 导入内部引擎
from .core.renderer import MarkdownRenderer
from .core.tutor_brain import TutorBrain

@register("omni_tutor", "YourName", "全知导师 - 终极认知版", "4.0.0")
class OmniTutorPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        
        # 1. 基础与模型配置加载 (彻底解耦)
        # 做了兼容处理，防止新旧配置项键名不一致
        self.dashscope_key = config.get("dashscope_key", config.get("dashscope_api_key", ""))
        self.siliconflow_key = config.get("siliconflow_key", config.get("siliconflow_api_key", ""))
        self.zoom = config.get("ocr_zoom", 3.0)
        
        self.ocr_model = config.get("ocr_model", "qwen-vl-ocr-2025-11-20")
        self.vision_model = config.get("vision_model", "qwen-vl-max")
        # 🌟 核心替换：将原先的 assistant_model 升级为快慢双脑
        self.fast_model = config.get("fast_model", "qwen-turbo")
        self.reasoning_model = config.get("reasoning_model", "deepseek-v3")
        self.embedding_model = config.get("embedding_model", "BAAI/bge-m3")
        self.rerank_model = config.get("rerank_model", "BAAI/bge-reranker-v2-m3")
        
        # 2. 状态队列初始化
        self.waiting_users = set()           # 主库录入
        self.important_waiting_users = set() # 独立库录入
        self.deep_read_waiting_users = set() # 精读模式
        self.file_buffer = {}                # 防抖等待购物车
        self.processed_msg_ids = set()       # 消息去重池
        self.aborted_users = set()           # 🌟 新增：紧急刹车拦截池
        self.ocr_waiting_users = set()       # 🌟 新增：OCR 打包提取专属等待队列
        
        self.enable_md_reply = False
        self.last_responses = {} 
        self.background_tasks = set()  # 🌟 修复：真正初始化这个集合，防止静默任务被杀
        # 🌟 一级学科强制收敛黑洞 (将各种别名吸入正统名称)
        self.course_alias_map = {
        "高等数学": "微积分",
        "高数": "微积分",
        "数学分析": "微积分",
        "线代": "线性代数",
        "大物": "大学物理",
        "普通物理": "大学物理",
        "C++": "C语言",
        "编程基础": "C语言",
        "闲聊": "日常对话",
        "数学": "微积分" # 兜底过于宽泛的词
        }

        # 3. 交互与渲染引擎初始化
        self.renderer = MarkdownRenderer()
        print("[OmniTutor] 🔄 正在检查 Chromium 浏览器内核...")
        try:
            print("[OmniTutor] ✅ 浏览器内核准备就绪，长图渲染火力全开！")
        except Exception as e:
            print(f"[OmniTutor] ❌ 内核异常: {e}")

        # 4. 智能双脑初始化并透传全部模型参数
        keys_valid = (
            self.dashscope_key and "填写" not in self.dashscope_key and 
            self.siliconflow_key and "填写" not in self.siliconflow_key
        )

        if keys_valid:
            self.brain = TutorBrain(
                dashscope_key=self.dashscope_key, 
                siliconflow_key=self.siliconflow_key,
                fast_model=self.fast_model,             # 🌟 传入快脑 (用于高并发打标)
                reasoning_model=self.reasoning_model,   # 🌟 传入慢脑 (用于意图裂变)
                ocr_model=self.ocr_model,
                vision_model=self.vision_model,
                embedding_model=self.embedding_model,
                rerank_model=self.rerank_model
            )
            # 物理隔离的独立副脑 (同样装配双脑)
            self.important_brain = TutorBrain(
                dashscope_key=self.dashscope_key, 
                siliconflow_key=self.siliconflow_key,
                fast_model=self.fast_model,             # 🌟 传入快脑
                reasoning_model=self.reasoning_model,   # 🌟 传入慢脑
                ocr_model=self.ocr_model,
                vision_model=self.vision_model,
                embedding_model=self.embedding_model,
                rerank_model=self.rerank_model,
                persist_dir="data/important_vdb",       
                db_path="data/important_sqlite.db"      
            )
        else:
            self.brain = None
            self.important_brain = None
            print("[OmniTutor] ⚠️ 警告：检测到 API Key 缺失。")

    

    def _debug_log(self, msg: str):
        """专门用于排查死机的本地日志记录器"""
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug.log")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        except: pass

    # =========================================================================
    # 第一部分：指令控制与文件拦截防抖流水线
    # =========================================================================

    @filter.command("学习")
    async def learn_command(self, event: AstrMessageEvent):
        """手动投喂学术资料至主库：/学习"""
        if not self.brain: 
            yield event.plain_result("❌ 导师未唤醒，请检查 API Key。")
            return
        self.waiting_users.add(event.get_sender_id())
        yield event.plain_result("📝 学习模式已就绪！👉 请直接发送图片、PDF或Office文档（支持一次发送多个文件）。")

    @filter.command("重要录入")
    async def import_important_file(self, event: AstrMessageEvent):
        """独立重要库录入：支持多行长文本或文件"""
        if not self.important_brain: return
        full_text = event.message_obj.message_str.replace("/重要录入", "").strip()
        
        if full_text: # 直接输入文字
            async def progress(msg): await event.send(event.plain_result(msg))
            yield event.plain_result("🌟 [独立重要库] 正在切割处理文字信息...")
            res, _ = await self.important_brain.ingest_important_info(raw_text=full_text, source_label="手动录入", progress_callback=progress)
            yield event.plain_result(res)
            return

        self.important_waiting_users.add(event.get_sender_id())
        yield event.plain_result("🌟 [独立重要库] 极速录入就绪！👉 请直接发送含零碎信息的图片或文档。")

    @filter.command("精读")
    async def deep_read_command(self, event: AstrMessageEvent):
        """【新增】进入深度精读与思考总结模式"""
        if not self.brain: return
        self.deep_read_waiting_users.add(event.get_sender_id())
        yield event.plain_result("📖 【深度精读模式】已开启！\n👉 请发送文件或图片，可琳将开启深度思考模式为您整理最详尽的复习资料（默认下发 MD 文件）。")

    @filter.command("ocr")
    async def manual_ocr_command(self, event: AstrMessageEvent):
        """手动触发 OCR：进入防抖打包提取模式"""
        if not self.brain or not hasattr(self.brain, 'ocr'):
            yield event.plain_result("❌ OCR 引擎未初始化。")
            return
            
        uid = event.get_sender_id()
        if not hasattr(self, 'ocr_waiting_users'): self.ocr_waiting_users = set()
        self.ocr_waiting_users.add(uid)
        
        yield event.plain_result("🔎 [纯文本提取模式] 已就绪！\n👉 请直接发送需要提取的图片（支持连发多图，5秒内无新文件将自动打包提取）。")

    @filter.regex(r".*")
    async def handle_waiting_file(self, event: AstrMessageEvent):
        """统一拦截：支持带时间窗口（防抖）的多模式打包收集"""
        msg_id = getattr(event.message_obj, 'message_id', None)
        if msg_id:
            if msg_id in getattr(self, 'processed_msg_ids', set()): return
            self.processed_msg_ids.add(msg_id)
            if len(self.processed_msg_ids) > 1000: self.processed_msg_ids.clear()

        plain_text = "".join([c.text for c in event.message_obj.message if isinstance(c, Comp.Plain)]).strip()
        
        # 放行所有唤醒指令
        if plain_text in ["/学习", "/重要录入", "/碎片录入", "/精读", "/ocr", "/提取文字"]: return

        uid = event.get_sender_id()
        is_main = uid in getattr(self, 'waiting_users', set())
        is_miu = uid in getattr(self, 'important_waiting_users', set())
        is_reading = uid in getattr(self, 'deep_read_waiting_users', set())
        is_ocr = uid in getattr(self, 'ocr_waiting_users', set())

        # 如果不属于任何收集模式，立刻放行
        if not any([is_main, is_miu, is_reading, is_ocr]): return 

        # 如果在收集中途遇到其他指令，安全退出当前收集状态
        if plain_text.startswith("/"): 
            self.waiting_users.discard(uid)
            self.important_waiting_users.discard(uid)
            self.deep_read_waiting_users.discard(uid)
            if hasattr(self, 'ocr_waiting_users'): self.ocr_waiting_users.discard(uid)
            if uid in getattr(self, 'file_buffer', {}) and self.file_buffer[uid]['timer']:
                self.file_buffer[uid]['timer'].cancel()
            self.file_buffer.pop(uid, None)
            return

        # 确认是目标文件后，停止事件传播，防止大模型插嘴
        event.stop_event()

        files = await self._extract_files_info(event)
        
        if not hasattr(self, 'file_buffer'): self.file_buffer = {}
        
        # 🛒 初始化购物车
        if uid not in self.file_buffer:
            if is_reading: prefix = "📖 [深度精读]"
            elif is_miu: prefix = "🌟 [独立重要库]"
            elif is_ocr: prefix = "🔎 [纯文本提取]"
            else: prefix = "🧠 [全知主库]"
            
            self.file_buffer[uid] = {
                'files': [], 'text': [], 'timer': None,
                'target_brain': self.important_brain if is_miu else self.brain,
                'prefix': prefix,
                'is_miu': is_miu,
                'is_reading': is_reading,
                'is_ocr': is_ocr
            }
            await event.send(event.plain_result(f"⏳ 收到资料，正在等待后续文件... (5秒内无动作将自动打包处理)"))

        if files: self.file_buffer[uid]['files'].extend(files)
        if plain_text: self.file_buffer[uid]['text'].append(plain_text)

        # 🔄 防抖倒计时重置
        if self.file_buffer[uid]['timer']:
            self.file_buffer[uid]['timer'].cancel()
            
        self.file_buffer[uid]['timer'] = asyncio.create_task(self._process_buffered_data(uid, event))

    async def _process_buffered_data(self, uid: str, event: AstrMessageEvent):
        """倒计时结束后的终极处理分支"""
        try: await asyncio.sleep(5)
        except asyncio.CancelledError: return

        data = self.file_buffer.pop(uid, None)
        if not data: return

        # 卸载标志位
        getattr(self, 'waiting_users', set()).discard(uid)
        getattr(self, 'important_waiting_users', set()).discard(uid)
        getattr(self, 'deep_read_waiting_users', set()).discard(uid)
        if hasattr(self, 'ocr_waiting_users'): self.ocr_waiting_users.discard(uid)

        files = data['files']
        text = "\n".join(data['text']).strip()
        target_brain = data['target_brain']
        prefix = data['prefix']
        is_miu = data['is_miu']
        is_reading = data['is_reading']
        is_ocr = data.get('is_ocr', False)

        async def progress(msg): 
            await event.send(event.plain_result(msg))
            await asyncio.sleep(0.6) 

        try:
            # ==========================================
            # 🌟 分支 0：OCR 批量提取模式
            # ==========================================
            if is_ocr:
                if not files:
                    await progress("❌ 未检测到有效图片，文字提取已自动退出。")
                    return
                await progress(f"{prefix} 收集完毕！正在启动高精度引擎批量扫描 {len(files)} 张图片...")
                results = []
                for idx, (target_path, file_name) in enumerate(files):
                    try:
                        extracted_text = await self.brain.ocr.process_file(target_path, 0, self.zoom, enable_vision=True) 
                        if extracted_text:
                            results.append(f"【图 {idx+1}】\n{extracted_text}")
                        else:
                            results.append(f"【图 {idx+1}】未能识别出有效文字。")
                    except Exception as e:
                        results.append(f"【图 {idx+1}】解析失败: {e}")
                
                final_text = "\n\n".join(results)
                # 如果附带了文字说明，也一并呈现在提取结果中
                if text: final_text += f"\n\n【附加文本】：{text}"
                
                await event.send(event.plain_result(f"✅ 提取完成：\n\n{final_text}"))
                return

            # ==========================================
            # 🌟 分支 1：深度精读模式 (下发 MD 文件)
            # ==========================================
            if is_reading:
                await progress(f"{prefix} 收到资料！准备启动百炼思考引擎进行重构...")
                summary = await target_brain.deep_read_and_summarize(files, text, self.zoom, progress_callback=progress)
                await progress(f"📝 思考与重构完毕！可琳正在为您生成详尽的 Markdown 笔记文件...")
                out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")
                os.makedirs(out_dir, exist_ok=True)
                md_filename = f"deepread_{uuid.uuid4().hex[:6]}.md"
                md_path = os.path.join(out_dir, md_filename)
                with open(md_path, "w", encoding="utf-8") as f: f.write(summary)
                await event.send(event.chain_result([Comp.File(file=md_path, name=md_filename)]))
                return

            # ==========================================
            # 🌟 分支 2：知识库录入模式
            # ==========================================
            combined_ocr_text = "" 
            if files:
                await progress(f"📦 收集完毕！共收到 {len(files)} 个文件，即将开启批量流水线...")
                for target_path, file_name in files:
                    await progress(f"{prefix} 正在深度解析 [{file_name}]...")
                    if is_miu:
                        result = await target_brain.ingest_important_info(file_path=target_path, source_label=file_name, zoom=self.zoom, progress_callback=progress)
                    else:
                        result = await target_brain.learn_from_file(target_path, source_name=file_name, zoom=self.zoom, progress_callback=progress)
                        
                    res_msg = result[0] if isinstance(result, tuple) else str(result)
                    ocr_text = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
                    await progress(res_msg)
                    if ocr_text: combined_ocr_text += f"\n\n# 【文件提取】{file_name}\n\n{ocr_text}"
                if len(files) > 1:
                    await progress(f"🎉 批量录入完成，共完美处理了 {len(files)} 个文件！")
                    
            elif text and is_miu:
                await progress(f"{prefix} 收集完毕，正在切割处理多段文字信息...")
                result = await target_brain.ingest_important_info(raw_text=text, source_label="手动录入", progress_callback=progress)
                res_msg = result[0] if isinstance(result, tuple) else str(result)
                await progress(res_msg)
            else:
                await progress("❌ 未检测到有效文件，录入模式已自动退出。")
                return

            if getattr(self, 'enable_md_reply', False) and combined_ocr_text.strip():
                out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")
                os.makedirs(out_dir, exist_ok=True)
                md_filename = f"ocr_batch_{uuid.uuid4().hex[:6]}.md"
                md_path = os.path.join(out_dir, md_filename)
                with open(md_path, "w", encoding="utf-8") as f: f.write(combined_ocr_text.strip())
                await event.send(event.chain_result([Comp.File(file=md_path, name=md_filename)]))
                
        except Exception as e:
            import traceback
            print(f"[Process Error] {traceback.format_exc()}")
            await event.send(event.plain_result(f"❌ 批量处理任务发生异常: {str(e)}"))

    @filter.command("取消")
    async def cancel_action(self, event: AstrMessageEvent):
        """紧急刹车：中止录入状态，并拦截大模型当前正在生成的对话回复"""
        uid = event.get_sender_id()
        canceled_file = False

        if uid in getattr(self, 'waiting_users', set()): self.waiting_users.discard(uid); canceled_file = True
        if uid in getattr(self, 'important_waiting_users', set()): self.important_waiting_users.discard(uid); canceled_file = True
        if uid in getattr(self, 'deep_read_waiting_users', set()): self.deep_read_waiting_users.discard(uid); canceled_file = True
        if uid in getattr(self, 'ocr_waiting_users', set()): self.ocr_waiting_users.discard(uid); canceled_file = True

        if uid in getattr(self, 'file_buffer', {}):
            if self.file_buffer[uid]['timer']: self.file_buffer[uid]['timer'].cancel()
            self.file_buffer.pop(uid, None)
            canceled_file = True

        # ==========================================
        # 🌟 必须补上这块：拦截挂起的 8秒 纯图片智能缝合状态
        # ==========================================
        if hasattr(self, "pending_image_tasks") and uid in self.pending_image_tasks:
            del self.pending_image_tasks[uid]
            canceled_file = True

        self.aborted_users.add(uid)

        if canceled_file:
            yield event.plain_result("🛑 紧急刹车成功！已取消当前收集任务，并拦截潜在的对话回复。")
        else:
            yield event.plain_result("🛑 已收到中断指令！如果可琳正在为您生成回复，该回复将被立刻丢弃闭嘴。")
    # =========================================================================
    # 第二部分：全自动拦截器 (视觉截获与知识注入)
    # =========================================================================

    # =========================================================================
    # 🌟 全局消息防抖与图文缝合器 (Smart Bundle)
    # =========================================================================
    @filter.regex(r".*")
    async def global_smart_bundle(self, event: AstrMessageEvent):
        """在进入大模型前，从源头拦截纯图片并等待文字缝合"""
        uid = event.get_sender_id()
        msg_str = event.message_obj.message_str.strip()

        # 🌟 遇到所有 / 指令放行
        if msg_str.startswith("/"): return 
        
        # 🌟 核心修复：如果用户已经处于“收集状态”（学习、精读、OCR），全局拦截器立刻让步！
        if any(uid in getattr(self, attr, set()) for attr in ['waiting_users', 'important_waiting_users', 'deep_read_waiting_users', 'ocr_waiting_users']):
            return
        
        # ==========================================
        # 🌟 修复 Bug 1 & 2：遇到所有指令（如 /取消, /学习），直接放行，绝对不能参与图文缝合！
        # ==========================================
        if msg_str.startswith("/"):
            return 

        has_img = any(isinstance(c, Comp.Image) for c in event.message_obj.message)
        
        if not hasattr(self, "pending_image_tasks"):
            self.pending_image_tasks = {}

        # 1. 场景 A：收到纯图片
        if has_img and not msg_str:
            self._debug_log("📸 收到纯图片，挂起当前消息流 8 秒等待文字补充...")
            await event.send(event.plain_result("🖼️ 可琳收到图片啦~ 正在乖乖等待主人补充文字说明哦 (8秒内有效)。如果不需要补充，可琳稍后会直接开始分析图片呢♪"))
            
            # 把图片暂存到池子里
            self.pending_image_tasks[uid] = [c for c in event.message_obj.message if isinstance(c, Comp.Image)]
            
            # 异步阻塞当前图片事件的传播
            for _ in range(80):  
                await asyncio.sleep(0.1)
                if uid not in self.pending_image_tasks:
                    event.stop_event()
                    return
            
            if uid in self.pending_image_tasks:
                del self.pending_image_tasks[uid]
                self._debug_log("⏳ 8秒已过，未收到文字，单独放行纯图片。")
            return

        # 2. 场景 B：收到文字消息，且池子里有等待的图片
        elif msg_str and uid in self.pending_image_tasks:
            self._debug_log("✅ 收到文字，将挂起的图片缝合到当前消息中！")
            cached_images = self.pending_image_tasks.pop(uid)
            event.message_obj.message = cached_images + event.message_obj.message

    # =========================================================================
    # 🌟 知识注入与检索拦截 (剥离了等待逻辑，更加纯粹)
    # =========================================================================
    @filter.on_llm_request()
    async def auto_inject_knowledge(self, event: AstrMessageEvent, req: ProviderRequest):
        """只负责 RAG 检索和人设注入"""
        if not self.brain: return
        uid = event.get_sender_id()
        
        if uid in self.aborted_users:
            self.aborted_users.discard(uid)
            
        current_query = event.message_str.strip()
        files = await self._extract_files_info(event)

        vision_hint = ""
        if files:
            self._debug_log("👁️ 图像已进入主脑视觉处理通道。")
            vision_hint = "\n(System: 用户发送了图片，请直接通过多模态视觉能力进行观察和解析。)\n"

        if not current_query and not files: return

        # 只要有文字，必然触发 RAG 检索 (现在图文缝合后一定有文字)
        if current_query:
            await event.send(event.plain_result("🔍 正在检索全知知识库与认知图谱..."))
            try:
                search_result = await self.brain.retrieve_knowledge(query=current_query, uid=uid, top_k=3)
                priority_instruction = (
                    f"\n\n[SYSTEM INSTRUCTION: 专属记忆回放与最高优先级约束]{vision_hint}\n"
                    "【人格铁律】：请绝对保持你【治愈系女仆导师·可琳】的核心设定。严禁使用干巴巴的机械列表！\n"
                    "【知识采信铁律】：以下是从【专属私有记忆】中检索到的资料。你必须优先、深度依赖以下内容进行解答！\n"
                    "--- 专属私有记忆检索结果 ---\n"
                )
                req.system_prompt = (req.system_prompt or "") + priority_instruction + search_result
            except Exception as e:
                self._debug_log(f"❌ 检索拦截崩溃: {e}")
        else:
            req.system_prompt = (req.system_prompt or "") + f"\n\n[SYSTEM INSTRUCTION]\n保持【治愈系女仆导师·可琳】人格。{vision_hint}"

    @filter.on_llm_response()
    async def render_llm_response(self, event: AstrMessageEvent, resp):
        """核心回复拦截：长图渲染 + 认知追踪"""
        uid = event.get_sender_id()
        
        # 🌟 终极拦截点：如果在生成期间用户喊了 /取消
        if uid in self.aborted_users:
            self.aborted_users.discard(uid)  # 摘除标记
            resp.completion_text = ""        # 强行清空大模型的回复
            event.stop_event()               # 彻底阻断事件传播
            print(f"[OmniTutor] 🛑 成功拦截并丢弃了 {uid} 的一次作废回复。")
            return                           # 直接退出，连静默认知追踪都不做了！
            
        if not hasattr(resp, 'completion_text') or not resp.completion_text:
            resp.completion_text = "⚠️ 导师的大模型大脑突然短路了（返回了空回复），请检查 API 状态。"
            return
            
        await event.send(event.plain_result("💡 模型思考完毕，正在生成最终专属图文回复..."))

        original_text = resp.completion_text
        uid = event.get_sender_id()
        self.last_responses[uid] = original_text 

        # 🌟 启动静默认知追踪器
        # 🌟 启动静默认知追踪器
        if self.brain:
            user_query = event.message_obj.message_str
            # 🌟 修复：强引用保存 task，防止被 GC 清理，并在执行完毕后自动移除
            task = asyncio.create_task(self._silent_track(uid, user_query, original_text))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

        try:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")
            os.makedirs(out_dir, exist_ok=True)  
            self.renderer.output_dir = out_dir
            
            success, result = await self.renderer.render_to_image(original_text)
            
            if success:
                if getattr(self, 'enable_md_reply', False):
                    md_filename = f"reply_{uuid.uuid4().hex[:6]}.md"
                    md_path = os.path.join(out_dir, md_filename)
                    with open(md_path, "w", encoding="utf-8") as f: f.write(original_text)
                    await event.send(event.chain_result([Comp.File(file=md_path, name=md_filename)]))
                else:
                    await event.send(event.chain_result([Comp.Image(file=result)]))
                resp.completion_text = "" 
            else:
                resp.completion_text = f"❌ 渲染引擎崩溃啦！核心报错如下：\n\n{result[:500]}"
                
        except Exception as e:
            resp.completion_text = f"❌ 拦截器严重错误：{e}"

    import re
    import json

    # 在类初始化 __init__ 里加上
    # self.background_tasks = set() 

    async def _silent_track(self, uid: str, query: str, answer: str):
        """带埋点的追踪流程，包含一级学科强力纠偏收敛"""
        try:
            import time
            task_id = f"{uid}_{int(time.time())}"
            logger.info(f"🚩 [CP1-触发] 任务 {task_id} 启动。用户: {uid}")
            
            if not answer or len(answer) < 5:
                logger.info(f"❌ [流程中断] 答案太短或为空，不触发追踪。")
                return

            logger.info(f"🚩 [CP2-打标中] 正在呼叫慢脑进行认知分析...")
            analysis = await self.brain.tagger.analyze_user_struggle(query, answer)
            
            if not analysis:
                logger.warning(f"❌ [CP3-失败] 慢脑返回为空，可能是 API 超时或 JSON 解析失败。")
                return
            
            logger.info(f"🚩 [CP4-分析成功] 慢脑判定结果: {analysis}")
            
            # 获取原始打标结果
            raw_course = analysis.get('course', '通用')
            concepts = analysis.get('concepts', [])
            mastery_delta = analysis.get('mastery_delta', 0.05)
            
            # 物理清洗逻辑：去除斜杠等多余符号
            if "/" in raw_course: 
                raw_course = raw_course.split("/")[0].strip()
            
            # 如果是闲聊或没有提取到概念，直接终止
            if not concepts or raw_course == "闲聊":
                logger.info(f"ℹ️ [流程结束] 判定为闲聊或无概念，不计入图谱。")
                return
            
            # ==========================================
            # 🌟 核心修复：一级学科强制收敛器 (拦截并纠偏)
            # ==========================================
            final_course = raw_course
            
            # 1. 查字典绝对匹配
            if raw_course in self.course_alias_map:
                final_course = self.course_alias_map[raw_course]
                self._debug_log(f"🔧 [图谱收敛] 字典精准命中：将 '{raw_course}' 纠偏为 -> '{final_course}'")
            else:
                # 2. 模糊包含匹配兜底 (比如模型输出"大学微积分")
                for std_name in ["微积分", "线性代数", "大学物理", "人工智能基础", "C语言", "Python"]:
                    if std_name in raw_course:
                        final_course = std_name
                        self._debug_log(f"🔧 [图谱收敛] 模糊包含命中：将 '{raw_course}' 提取为 -> '{final_course}'")
                        break
            # ==========================================

            logger.info(f"🚩 [CP5-入库中] 准备写入数据库: {final_course} -> {concepts}")
            
            # 执行写入 (🌟 必须使用清洗后的 final_course)
            self.brain.sql_manager.update_concept_pool(final_course, concepts)
            self.brain.sql_manager.update_user_cognition(uid, final_course, concepts, mastery_delta, 0.0)

            import asyncio
            asyncio.create_task(self.brain._run_background_normalization({final_course}, set(concepts), None))
            
            logger.info(f"✅ [CP6-完结] 任务 {task_id} 成功闭环！")

        except Exception as e:
            logger.error(f"💥 [流程崩溃] 任务发生异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _extract_files_info(self, event: AstrMessageEvent):
        """提取消息中的文件信息"""
        files = []
        for c in event.message_obj.message:
            if isinstance(c, (Comp.Image, Comp.File)):
                try:
                    file_path = await c.get_file() if hasattr(c, 'get_file') else c.file
                    if not file_path or not os.path.exists(file_path): continue
                    
                    file_name = getattr(c, 'name', os.path.basename(file_path))
                    # 避免无限循环提取自己生成的文件
                    if file_name.startswith("ocr_batch_") or file_name.startswith("reply_") or file_name.startswith("deepread_"):
                        continue
                    files.append((file_path, file_name))
                except: continue
        return files

    # =========================================================================
    # 第三部分：杂项管理指令 (档案 / 诊断 / 清除等)
    # =========================================================================
    
    @filter.command("档案")
    async def list_main_sources(self, event: AstrMessageEvent):
        if not self.brain: return
        sources = self.brain.sql_manager.get_all_source_names()
        if not sources:
            yield event.plain_result("📭 主库目前没有任何记录。")
            return
        res = "📂 【主库学术档案列表】\n" + "\n".join([f"🔸 {s}" for s in sources]) 
        res += "\n\n💡 提示：使用 /遗忘 [名称/关键词] 即可模糊删除对应记录。"
        yield event.plain_result(res)

    @filter.command("遗忘")
    async def forget_command(self, event: AstrMessageEvent, *, filename: str = ""):
        if not self.brain: return
        if not filename: yield event.plain_result("❓ 请输入要删除的文件名。"); return
        yield event.plain_result(self.brain.forget_file(filename))

    @filter.command("检索")
    async def ask_command(self, event: AstrMessageEvent, *, question: str = ""):
        if not self.brain: return
        if not question: yield event.plain_result("❓ 请附带问题。"); return
        
        yield event.plain_result("🔍 正在从主库启动 Agentic 检索前瞻...")
        
        # 1. 调用大脑进行多维检索
        res = await self.brain.retrieve_knowledge(query=question, uid=event.get_sender_id(), top_k=3)
        
        # 2. 如果没搜到，直接返回文本提示
        if "【无匹配资料】" in res:
            yield event.plain_result(res)
            return

        # 3. 🌟 核心补丁：调用渲染引擎将检索结果转为长图
        from astrbot.api.all import logger
        logger.info(f"🎨 [OmniTutor] 正在为检索结果渲染高清长图...")
        
        try:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")
            os.makedirs(out_dir, exist_ok=True)
            self.renderer.output_dir = out_dir
            
            # 使用渲染器处理检索出的 Markdown 文本
            success, result = await self.renderer.render_to_image(res)
            
            if success:
                # 发送图片结果
                yield event.chain_result([Comp.Image(file=result)])
            else:
                # 渲染失败则降级返回纯文本
                logger.error(f"❌ 检索结果渲染失败: {result}")
                yield event.plain_result(res)
                
        except Exception as e:
            logger.error(f"❌ 渲染流程异常: {e}")
            yield event.plain_result(res)


    @filter.command("存入")
    async def save_last_reply(self, event: AstrMessageEvent, *, instruction: str = ""):
        if not self.brain: return
        last_text = self.last_responses.get(event.get_sender_id())
        if not last_text: yield event.plain_result("⚠️ 找不到上次回答记录。"); return
        async def progress(text): await event.send(event.plain_result(text))
        yield event.plain_result("🚀 启动助理模型进行学术提纯回流...")
        res = await self.brain.learn_from_text(raw_text=last_text, instruction=instruction, source_label="问答回流", progress_callback=progress)
        yield event.plain_result(res)

    @filter.command("诊断")
    async def chapter_report(self, event: AstrMessageEvent):
        """🌟 全面适配新扁平化结构"""
        if not self.brain: return
        raw_data = self.brain.sql_manager.get_user_cognition_data(event.get_sender_id())
        if not raw_data: yield event.plain_result("📊 暂无认知数据。"); return
        
        processed_data = []
        for row in raw_data:
            try: last_time = datetime.strptime(row['last_interact'], "%Y-%m-%d %H:%M:%S")
            except: last_time = datetime.now()
            days_passed = (datetime.now() - last_time).total_seconds() / 86400.0
            
            real_mastery = row['mastery']
            if days_passed > 0:
                decay_factor = math.exp(-math.log(2) / 7.0 * days_passed)
                real_mastery = (row['mastery'] - 0.2 * row['mastery']) * decay_factor + 0.2 * row['mastery']
                
            processed_data.append({**row, "real_mastery": real_mastery, "days": days_passed})
            
        processed_data.sort(key=lambda x: (x['real_mastery'], -x['struggle']))
        md_text = "# 📊 动态认知内窥镜 (遗忘曲线版)\n\n| 课程模块 | 核心概念 | 真实掌握度 | 距今 | 阻力 |\n|---|---|---|---|---|\n"
        for row in processed_data[:15]:
            bar = "🟩" * int(row['real_mastery'] * 5) + "⬜" * (5 - int(row['real_mastery'] * 5))
            md_text += f"| {row['course']} | **{row['concept']}** | {bar} {row['real_mastery']*100:.0f}% | {row['days']:.1f}d | {row['struggle']:.1f} |\n"
            
        yield event.plain_result("🎨 正在绘制诊断图谱...")
        success, result = await self.renderer.render_to_image(md_text)
        if success: await event.send(event.chain_result([Comp.Image(file=result)]))
        else: yield event.plain_result(md_text)

    @filter.command("图谱")
    async def knowledge_tree(self, event: AstrMessageEvent):
        """🌟 全面适配新扁平化概念池"""
        if not self.brain: return
        concepts = self.brain.sql_manager.get_all_knowledge_concepts()
        if not concepts: yield event.plain_result("🌳 当前概念池为空。"); return
            
        tree_dict = {}
        for node in concepts:
            course = node['course']
            concept = node['concept']
            freq = node['frequency']
            
            hot_icon = "🔥🔥🔥" if freq >= 10 else "🔥🔥" if freq >= 5 else "🔥" if freq >= 2 else "🧊"
            display = f"{concept} {hot_icon} *(频次: {freq})*"
            
            if course not in tree_dict: tree_dict[course] = []
            tree_dict[course].append((freq, display))
            
        md_text = "# 🌳 扁平化概念图谱概览\n\n"
        for course, c_list in tree_dict.items():
            md_text += f"## 📘 {course}\n"
            c_list.sort(key=lambda x: x[0], reverse=True)
            # 限制每个学科只显示前30个高频词
            md_text += f"- **核心概念**: {', '.join([item[1] for item in c_list[:100]])}\n"
                
        yield event.plain_result("🎨 正在渲染概念网络...")
        success, result = await self.renderer.render_to_image(md_text)
        if success: await event.send(event.chain_result([Comp.Image(file=result)]))
        else: yield event.plain_result(md_text)

    @filter.command("清空主库")
    async def clear_brain(self, event: AstrMessageEvent):
        if not self.brain: return
        try:
            # 1. 彻底重置 ChromaDB 的双路图谱集合
            client = self.brain.vector_store.client
            for col_name in ["tutor_chunks", "tutor_tags"]:
                try: 
                    client.delete_collection(col_name)
                except Exception: 
                    pass
            
            # 重新建立全新的空集合
            self.brain.vector_store.chunks_collection = client.get_or_create_collection(
                name="tutor_chunks", metadata={"hnsw:space": "cosine"}
            )
            self.brain.vector_store.tags_collection = client.get_or_create_collection(
                name="tutor_tags", metadata={"hnsw:space": "cosine"}
            )
            
            # 2. 彻底清空 SQLite 并重置自增 ID (包含全新的倒排映射表)
            with self.brain.sql_manager._get_conn() as conn:
                cur = conn.cursor()
                # 倒排桥梁表 chunk_tag_mapping 也必须清空
                for table in ["concept_pool", "user_cognition", "chunk_tag_mapping", "chunks", "documents"]:
                    try: cur.execute(f"DELETE FROM {table}")
                    except: pass
                
                # 重置所有的 AUTOINCREMENT 序列
                cur.execute("DELETE FROM sqlite_sequence")
                conn.commit()
                
            yield event.plain_result("💥 脑白金生效！全知主脑的【双路图谱】已彻底清空并完全重置。")
        except Exception as e: 
            yield event.plain_result(f"❌ 擦除失败: {e}")

    @filter.command("重要清空")
    async def clear_important_database(self, event: AstrMessageEvent):
        if not self.important_brain: return
        try:
            # 1. 彻底重置重要库的双集合
            client = self.important_brain.vector_store.client
            for col_name in ["tutor_chunks", "tutor_tags"]:
                try: 
                    client.delete_collection(col_name)
                except Exception: 
                    pass

            self.important_brain.vector_store.chunks_collection = client.get_or_create_collection(
                name="tutor_chunks", metadata={"hnsw:space": "cosine"}
            )
            self.important_brain.vector_store.tags_collection = client.get_or_create_collection(
                name="tutor_tags", metadata={"hnsw:space": "cosine"}
            )

            # 2. 彻底清空 SQLite
            with self.important_brain.sql_manager._get_conn() as conn:
                cur = conn.cursor()
                for table in ["important_miu", "chunk_tag_mapping", "documents", "chunks", "concept_pool"]: 
                    try: cur.execute(f"DELETE FROM {table}")
                    except: pass
                    
                cur.execute("DELETE FROM sqlite_sequence")
                conn.commit()
                
            yield event.plain_result("💥 [重要库] 碎片及图谱数据已抹除并完全重置。")
        except Exception as e: 
            yield event.plain_result(f"❌ 抹除失败: {e}")

    @filter.command("删节点")
    async def delete_kb_node(self, event: AstrMessageEvent, *, node_name: str = ""):
        if not self.brain: return
        if not node_name: yield event.plain_result("⚠️ 请输入概念名称。"); return
        count = self.brain.sql_manager.delete_concept_by_name(node_name)
        yield event.plain_result(f"✅ 已删除概念：【{node_name}】" if count > 0 else "❌ 未找到该概念。")

    @filter.command("重置节点")
    async def reset_kb_mastery(self, event: AstrMessageEvent, *, node_name: str = ""):
        if not self.brain: return
        if not node_name: yield event.plain_result("⚠️ 请输入概念名称。"); return
        count = self.brain.sql_manager.reset_mastery_by_name(node_name)
        yield event.plain_result(f"🔄 已复原【{node_name}】的掌握度。" if count > 0 else "❌ 未找到该概念。")

    @filter.command("md开关")
    async def toggle_md_reply(self, event: AstrMessageEvent):
        self.enable_md_reply = not self.enable_md_reply
        yield event.plain_result(f"⚙️ MD 回复已【{'开启' if self.enable_md_reply else '关闭'}】。")

    @filter.command("重要查询")
    async def query_important(self, event: AstrMessageEvent, *, question: str = ""):
        if not self.important_brain: return
        if not question: yield event.plain_result("❓ 请输入查询问题。"); return
        yield event.plain_result("🔍 正在检索 MIU 碎片库...")
        yield event.plain_result(await self.important_brain.query_important_miu(question))

    @filter.command("重要档案")
    async def list_important_sources(self, event: AstrMessageEvent):
        if not self.important_brain: return
        mius = self.important_brain.sql_manager.get_all_mius()
        if not mius: yield event.plain_result("📭 重要库为空。"); return
        res = "📂 【重要库条目】\n" + "\n".join([f"🔸 {m['title']} *(批次:{m['source'].split('_')[-1] if '_' in m['source'] else m['source']})*" for m in mius[:20]])
        yield event.plain_result(res)
            
    @filter.command("重要遗忘")
    async def forget_important(self, event: AstrMessageEvent, *, keyword: str = ""):
        if not self.important_brain: return
        if not keyword: yield event.plain_result("❓ 请输入要删除的名称。"); return
        yield event.plain_result(self.important_brain.forget_miu(keyword))

    
    @filter.command("撤销")
    async def undo_last_insert(self, event: AstrMessageEvent):
        """撤销最后一次对数据库的知识录入（对话回流或文件存入）"""
        if not self.brain: 
            return
            
        # 1. 揪出最后一次录入的足迹
        latest_source = self.brain.sql_manager.get_latest_source_name()
        if not latest_source:
            yield event.plain_result("📭 数据库目前是空的，没有可以撤销的记录哦。")
            return
            
        yield event.plain_result(f"⏳ 锁定目标！正在撤回最后一次录入：[{latest_source}]...")
        
        try:
            # 2. 直接呼叫带有扫地僧机制的主脑遗忘逻辑！
            # 它会自动级联删除 SQLite 数据、ChromaDB 向量，并重算图谱词频
            result_msg = self.brain.forget_file(latest_source)
            
            yield event.plain_result(f"↩️ 撤销完成！\n{result_msg}")
        except Exception as e:
            from astrbot.api.all import logger
            logger.error(f"撤销失败: {e}")
            yield event.plain_result(f"❌ 撤销失败: {e}")

    @filter.command("自清洗")
    async def manual_clean_graph(self, event: AstrMessageEvent):
        """手动触发全库图谱清洗与坍缩"""
        if not self.brain: 
            yield event.plain_result("❌ 导师大脑未唤醒。")
            return
            
        yield event.plain_result("🚀 收到指令！正在为全知主库启动【深度自清洗】流水线，这可能需要一些时间，请稍候...")
        
        # 封装一个实时的进度播报回调
        async def progress(text):
            await event.send(event.plain_result(text))
            
        try:
            # 呼叫大脑执行全库深层清洗
            result_msg = await self.brain.clean_entire_graph(progress_callback=progress)
            yield event.plain_result(result_msg)
        except Exception as e:
            yield event.plain_result(f"❌ 清洗任务崩溃: {e}")

    @filter.command("合并概念")
    async def cmd_merge_concept(self, event: AstrMessageEvent, old_name: str, new_name: str):
        """指令：/合并概念 [旧名] [新名]"""
        if not self.brain: return
        yield event.plain_result(f"🪄 正在施展合并魔法...")
        res = await self.brain.manual_merge_concepts(old_name, new_name)
        yield event.plain_result(res)

    @filter.command("清洗概念")
    async def cmd_clean_concept(self, event: AstrMessageEvent, name: str):
        """指令：/清洗概念 [名称]"""
        if not self.brain: return
        async def progress(t): await event.send(event.plain_result(t))
        res = await self.brain.clean_specific_concept(name, progress_callback=progress)
        yield event.plain_result(res)

    @filter.command("重塑概念")
    async def cmd_reprocess_concept(self, event: AstrMessageEvent, name: str):
        """指令：/重塑概念 [名称] - 极度消耗 Token，慎用"""
        if not self.brain: return
        yield event.plain_result(f"🛠️ 启动重塑手术，正在呼叫慢脑重新审视 【{name}】 的所有切片...")
        async def progress(t): await event.send(event.plain_result(t))
        res = await self.brain.reprocess_unsuitable_concept(name, progress_callback=progress)
        yield event.plain_result(res)