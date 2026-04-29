import json
import logging
import asyncio
import base64
import os
import numpy as np
from typing import List, Dict, Optional, Tuple

# 🌟 抛弃充满 Bug 的 dashscope 原生库，改用稳定无敌的 openai 库
from openai import AsyncOpenAI 

logger = logging.getLogger(__name__)

class ConceptTagger:
    def __init__(self, dashscope_key: str, fast_model: str = "qwen-flash", reasoning_model: str = "qwen3.5-plus",vision_model: str = "qwen-vl-max"):
        # 🌟 直接使用 OpenAI 兼容模式连接阿里云百炼！
        self.client = AsyncOpenAI(
            api_key=dashscope_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.fast_model = fast_model 
        self.reasoning_model = reasoning_model
        self.vision_model = vision_model 

    async def _call_llm(self, prompt: str, target_model: str, retries: int = 3, force_json: bool = True, images: List[str] = None):
        """🌟 升级版：支持 OpenAI 标准的多模态图文输入"""
        
        if force_json and "json" not in prompt.lower():
            prompt = "【强制指令：你必须严格以 JSON 格式输出结果】\n" + prompt

        for attempt in range(retries):
            try:
                # 组装消息体
                content = [{"type": "text", "text": prompt}]
                
                if images:
                    for img_path in images:
                        if not os.path.exists(img_path): continue
                        with open(img_path, "rb") as f:
                            base64_img = base64.b64encode(f.read()).decode("utf-8")
                        # 识别 mime 类型
                        mime = "image/jpeg" if img_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{base64_img}"}
                        })

                kwargs = {
                    "model": target_model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.1,
                }
                
                if force_json:
                    kwargs["response_format"] = {"type": "json_object"}

                completion = await self.client.chat.completions.create(**kwargs)
                res_content = completion.choices[0].message.content
                return json.loads(res_content) if force_json else res_content

            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                    continue
                logger.error(f"❌ LLM 意图识别请求异常: {e}")
                return {} if force_json else ""
        
    async def _call_llm_text(self, prompt: str, target_model: str) -> str:
        """🌟 极其稳定的 OpenAI 兼容模式 - 纯文本/Markdown 深度输出"""
        try:
            # 去掉了那些乱七八糟的 enable_thinking 和 stream 参数，让大模型自由输出即可
            completion = await self.client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ LLM 纯文本请求抛出异常: {e}")
            return ""

    # =========================================================================
    # 🌟 阶段一：流式录入与实时规整 (Data Ingestion & Normalization)
    # =========================================================================

    async def extract_chunk_tags(self, chunk_content: str, course_pool: List[str] = None) -> Dict:
        """步 1：极速原生信息提取。优化概念提取的提示词策略。"""
        if not course_pool:
            course_pool = ["微积分", "常微分方程", "工程图学", "人工智能基础", "大学物理", "C语言", "Python"]
        course_str = ", ".join(course_pool)

        prompt = f"""
        角色：学术知识图谱构建专家
        任务：深入分析提供的学术文本切片，提取最精确的分类和核心概念。必须严格输出符合规范的 JSON 格式。

        【一级学科池参考】：[{course_str}]

        【🌟 核心概念提取准则】(极其重要):
        1. 概念(concepts)必须是明确的【学术专有名词、定理名称、公式名或专有技术词汇】。
        2. 绝对禁止使用泛泛而谈的词汇（如：“公式”、“介绍”、“第一章”、“代码”、“解题思路”）。
        3. 尽量简短，但必须是名词性短语（如：“牛顿第二定律”、“傅里叶级数”、“面向对象编程”）。

        【✅ 正面提取示例】：
        原文：“通过分部积分法，我们可以求解形如 x*e^x 的积分...” -> 概念：["分部积分法", "指数函数积分"]
        原文：“当且仅当行列式不为0时，矩阵可逆...” -> 概念：["矩阵可逆", "行列式"]

        【❌ 反面提取示例（禁止出现）】：
        概念：["积分求解", "性质", "数学"，"点乘","一般式"] (太泛了，毫无检索价值！)

        【切片内容】：
        {chunk_content[:1200]}

        【要求输出的 JSON 结构】：
        {{
            "tags": {{
                "course": "必须严格、一字不差地从给出的【一级学科池参考】中选择一个！绝对禁止自己创造如'高等数学'、'高数'、'数学'等同义词！只有当内容完全不属于这些学科时，才能输出新词。",
                "course_explanation": "一句话判断学科的理由",
                "concepts": ["学术概念1", "学术概念2", "学术概念3"], // 提取1-4个最高度概括的学术名词
                "concepts_explanation": "简述为何这几个词最能代表切片核心"
            }},
            "pedagogy": {{
                "type": ["定义定理"], // 从 [定义定理, 实际应用, 练习, 拓展] 中限选1-2项
                "type_explanation": "判断教学性质的理由"
            }},
            "boundary": {{
                "completeness": "相对完整", // 无头无尾 / 章节转换 / 相对完整
                "completeness_explanation": "判断边界状态的理由",
                "context_loss": false,
                "chapter_transition": false
            }}
        }}
        """
        try:
            return await self._call_llm(prompt, target_model=self.fast_model)
        except Exception as e:
            import logging
            logging.error(f"提取 Chunk Tags 失败: {e}")
            return {
                "tags": {"course": "通用", "concepts": []},
                "pedagogy": {"type": ["拓展"]},
                "boundary": {"completeness": "相对完整", "context_loss": False, "chapter_transition": False}
            }

    # =========================================================================
    # 🌟 阶段二：多维意图检索与答疑 (Retrieval & Synthesis)
    # =========================================================================

    async def analyze_multi_intent(self, query: str, images: List[str] = None) -> List[Dict]:
        """
        步 1：多重意图裂变 (🌟 视觉增强 + CoT 广度扩展版)
        """
        
        prompt = f"""
        你是一个顶级的学术意图路由专家与检索词扩展专家。
        请深度分析用户的提问（及图片题目,若有），进行多维度的“意图裂变”，提取出能够最大化知识召回率的关键词。
        
        【用户提问】：{query if query else "（见图片内容）"}
        
        【裂变规则】：
        1. **思维链先行 (CoT)**：在提取关键词前，必须先在 `analysis` 字段中简要推演解题需要的完整知识链路。严格限制三句话以内。
        2. **全维度概念提取**：基于你的推演，在 `concepts` 数组中**必须提取 5 到 10 个**精准的学术名词。必须涵盖以下维度：
           - 粗粒度大类：提取所属粗粒度知识点（如：三重积分、电路分析、数列收敛）。
           - 微观考点：直接考察的公式/定理（如：洛必达法则、节点电压法）。
           - 隐含前置知识：推导必备的基础概念（如：等价无穷小、受力分析）。
           - 解题工具/方法：实际计算需要的数学工具（如：分部积分、极坐标变换）。
        3. **类型判定** (pedagogy_type)：必须从 [定义定理, 实际应用, 练习, 拓展] 中选择。若图片是题目，必须包含“练习”。
        
        【输出 JSON】：
        {{
            "intents": [
                {{
                    "analysis": "这里简要写出你的分析过程，严格限制三句话以后，拆解这道题具体考察了什么，隐含了什么前置知识...",
                    "concepts": ["粗粒度知识点", "微观考点1", "隐含知识2", "解题方法3", "易错点4", "工具5"], 
                    "pedagogy_type": ["你判定的教学类型，取值严格限定为：定义定理, 实际应用, 练习, 拓展 中的一个或多个"]
                }}
            ]
        }}
        """
        try:
            # 如果有图片，强制使用视觉模型进行裂变
            target = self.vision_model if images else self.reasoning_model
            res = await self._call_llm(prompt, target_model=target, images=images, force_json=True)
            return res.get("intents", [])
        except Exception as e:
            from astrbot.api.all import logger
            logger.error(f"意图裂变失败: {e}")
            return [{"concepts": [], "pedagogy_type": []}]

    # =========================================================================
    # 🌟 其他工具方法 (静默认知诊断与精读整理)
    # =========================================================================

    async def extract_miu(self, text: str) -> List[Dict[str, str]]:
        """MIU 高内聚信息提取 (副脑碎片化入库专用)"""
        prompt = f"""
        将文本切分为高内聚信息单元(MIU)。必须输出 JSON 格式。
        切分原则：紧密相关信息打包，不可过度碎片化（如账号密码IP必须在一起）。
        【文本】：{text}
        
        【要求】：必须输出如下结构的 JSON 字符串：
        {{
            "data": [
                {{"title":"概括名词15字内", "content":"完整原文"}}
            ]
        }}
        """
        try:
            res = await self._call_llm(prompt, target_model=self.fast_model)
            return res.get("data", [])
        except: 
            return [{"title": text[:15]+"...", "content": text}]


    async def analyze_user_struggle(self, query: str, answer: str) -> Optional[Dict]:
        """适配扁平化标签的认知诊断 (开启深度思考 + 严格 JSON 约束)"""
        prompt = f"""
        你是一位顶级的教育心理学测量专家。请根据师生问答，进行知识掌握度测量。必须严格输出 JSON 格式。

        【问答数据】：
        用户问：{query}
        导师答：{answer}

        【量表】：
        维度一：提问深度
        - L1 识记/理解 (+0.05)
        - L2 应用/分析 (+0.15)
        - L3 综合/评价 (+0.25)
        维度二：受阻与错误归因
        - Fatal 概念崩塌 (-0.2)
        - Stuck 逻辑卡壳 (+0.0)
        - Slip 粗心 (-0.05)
        - None 无错误 (+0.0)

        🌟【强制计分公式】（绝对铁律）：
        mastery_delta = (维度一得分) + (维度二得分)
        你必须严格、死板地只使用上述量表中的固定数值进行加法计算！绝对禁止自己发明微小数！
        计算示例：判定为 L1(+0.05) 且 None(+0.0)，则 mastery_delta 必须精确填入 0.05。

        🌟【词汇提取铁律】（极其重要）：
        1. 提取的 concepts 数量必须严格限制在 1~2 个最核心的词汇！绝对不要贪多！
        2. concepts 必须是极度精简的纯学术名词，绝对禁止包含括号、解释说明或任何标点符号！
        （✅正确示例：["向量叉乘", "混合积"] ； ❌错误示例：["向量叉乘（向量积）", "向量混合积（标量三重积）"]）

        🌟【重要特殊规则】：如果判断用户的提问属于日常聊天、问候或感谢，请务必将 "concepts" 设为空数组 []，且将 "course" 设为 "闲聊"，并将 mastery_delta 设为 0.0。

        【要求严格按照以下根节点结构输出 JSON】（不要随意嵌套 analysis 字段！）：
        {{
            "course": "推测涉及的课程名",
            "concepts": ["核心概念1", "核心概念2"], // 最多2个，绝对不能有括号！
            "cognitive_state": "EXPLORING", // 从 CONFUSED, EXPLORING, APPLYING, MASTERED 中选
            "mastery_delta": 0.05, // 填入计算出的最终浮点数得分
            "diagnosis_reason": "一句话简述（🌟绝对限制在30个字以内！严禁复述思考过程！只写结论！）"
        }}
        """
        try: 
            # 🌟 核心修改：追加 enable_thinking=True 开启深度推理
            res = await self._call_llm(
                prompt, 
                target_model=self.reasoning_model, 
                force_json=True, 
            )
            if not isinstance(res, dict): return None
            return res
        except Exception as e: 
            from astrbot.api.all import logger
            logger.error(f"❌ 认知诊断失败: {e}")
            return None
    
    async def extract_academic_content(self, text: str, instruction: str) -> str:
        """提取对话回流的学术原文"""
        prompt = f"""
        从下方【文本】中，根据【指令】提取纯学术内容。
        强制：必须截取原文，绝对禁止修改、润色或总结。剔除对话废话。
        【指令】：{instruction if instruction else "提取全部学术内容"}
        【文本】：{text}
        """
        return await self._call_llm_text(prompt, target_model=self.fast_model)
    
    async def extract_chat_tags(self, chat_content: str, course_pool: List[str] = None) -> Dict:
        """
        专门用于日常对话/问答回流的打标。
        🌟 启用慢脑 (reasoning_model) + 深度思考 (enable_thinking=True)，精准提炼口语中的学术概念。
        """
        if not course_pool:
            course_pool = ["微积分", "常微分方程", "工程图学", "人工智能基础", "大学物理", "C语言", "Python"]
        course_str = ", ".join(course_pool)

        prompt = f"""
        角色：学科知识图谱标注工程师
        任务：分析输入的【师生对话/问答片段】，提取课程、概念、内容类型及边界特征。必须输出 JSON 格式。

        【一级学科池参考】：["微积分", "常微分方程", "工程图学", "人工智能基础", "大学物理", "C语言", "Python"]

        标签规则：
        - tags.course：单一课程名称，注意绝对唯一！🌟【绝对强制】：高优先从【一级学科池参考】中选取最贴切的一个！。如果不匹配，输出2-4个字的标准学科名词。
        - tags.concepts：1-3个核心概念。🌟【绝对强制】：必须是极致精简的【专有名词】。每个词必须在 8 个字以内！绝对禁止输出描述性短句！
        - pedagogy.type：1-2个内容类型，取值严格限定为：定义定理 / 实际应用 / 练习 / 拓展
        
        【对话内容】：
        {chat_content[:1200]}
        
        【要求输出的 JSON 结构】：
        {{
            "tags": {{
                "course": "课程名",
                "course_explanation": "一句话推断理由",
                "concepts": ["概念1", "概念2"],
                "concepts_explanation": "提炼理由"
            }},
            "pedagogy": {{
                "type": ["类型1"],
                "type_explanation": "一句话推断理由"
            }},
            "boundary": {{
                "completeness": "相对完整",
                "completeness_explanation": "这是问答切片，默认完整",
                "context_loss": false,
                "chapter_transition": false
            }}
        }}

        【样例】：
        {{
            "tags": {{
                "course": "微积分",
                "course_explanation": "对话核心围绕二重积分的计算技巧，属于多元微积分内容",
                "concepts": ["二重积分", "富比尼定理", "积分次序"],
                "concepts_explanation": "学生明确提及二重积分，老师引入富比尼定理说明交换条件，并讨论积分次序选择策略"
            }},
            "pedagogy": {{
                "type": ["定义定理", "实际应用"],
                "type_explanation": "老师解释了富比尼定理的适用条件（定义定理），同时指导学生如何根据区域形状选择积分顺序（实际应用）"
            }},
            "boundary": {{
                "completeness": "相对完整",
                "completeness_explanation": "这是问答切片，默认完整",
                "context_loss": false,
                "chapter_transition": false
            }}
        }}
        """
        try:
            # 🌟 核心：换用 reasoning_model，并开启 enable_thinking=True
            return await self._call_llm(prompt, target_model=self.reasoning_model)
        except Exception as e:
            import logging
            logging.error(f"提取对话 Tags 失败: {e}")
            return {
                "tags": {"course": "通用", "concepts": []},
                "pedagogy": {"type": ["拓展"]},
                "boundary": {"completeness": "相对完整", "context_loss": False, "chapter_transition": False}
            }
    # =========================================================================
    # 🌟 混合架构归一化 (向量直出 + AI 边缘裁决)
    # =========================================================================

    async def _edge_case_judge(self, word1: str, word2: str) -> bool:
        """边缘情况调用快脑进行判定 (严格 JSON 约束)"""
        prompt = f"""
        请判断“{word1}”和“{word2}”是否在学术上指代同一个核心概念或学科。
        你必须输出 JSON 格式。
        
        【要求输出的 JSON 结构】：
        {{
            "is_same": true或false
        }}
        """
        try:
            # 🌟 直接调用 _call_llm，该方法内部已经配置了 response_format={'type': 'json_object'}
            res = await self._call_llm(prompt, target_model=self.fast_model)
            # 解析出布尔值，兜底为 False
            return bool(res.get("is_same", False))
        except Exception as e:
            # 万一发生异常，为了安全起见，判定为不相似（保留为新词）
            logger.warning(f"AI 边缘裁决异常 ({word1} vs {word2}): {e}")
            return False

    async def cluster_words(self, new_words: List[str], pool_words: List[str], embedder, 
                            direct_threshold: float, ai_threshold: float) -> Dict[str, str]:
        """
        通用聚类逻辑。
        返回 {新词: 标准词} 的映射字典。不会修改原本就是新词的内容。
        """
        if not new_words or not pool_words: return {}
        
        # 1. 批量获取向量
        new_embs = await embedder.get_embeddings(new_words)
        pool_embs = await embedder.get_embeddings(pool_words)
        
        mapping = {}
        for nw, nw_emb_raw in zip(new_words, new_embs):
            # 🌟 优化：确保参与运算的是 numpy array，防止列表引发运算错误
            nw_emb = np.array(nw_emb_raw)
            best_match, max_sim = None, -1.0
            
            # 2. 余弦相似度比对
            for pw, pw_emb_raw in zip(pool_words, pool_embs):
                pw_emb = np.array(pw_emb_raw)
                
                # 计算余弦相似度，增加分母防零除保护
                norm_nw = np.linalg.norm(nw_emb)
                norm_pw = np.linalg.norm(pw_emb)
                if norm_nw == 0 or norm_pw == 0:
                    continue
                    
                sim = np.dot(nw_emb, pw_emb) / (norm_nw * norm_pw)
                if sim > max_sim:
                    max_sim, best_match = sim, pw
            
            # 3. 混合裁决流
            if max_sim >= direct_threshold:
                mapping[nw] = best_match  # 极高相似度，直接归一化
            elif max_sim >= ai_threshold:
                if await self._edge_case_judge(nw, best_match):
                    mapping[nw] = best_match # 模棱两可，AI 拍板通过
            # 低于 ai_threshold 的情况直接放弃，作为全新词汇存在
            
        return mapping
    
    async def deep_summarize_with_thinking(self, text: str) -> str:
        """
        🌟 深度精读模式：调用慢脑 (开启思考过程) 对长文档进行深度结构化总结与重构
        """
        prompt = f"""
        你是一位顶级的学术导师和知识重构专家。请对下方提供的【长篇学术/学习资料】进行深度研读，并重构为一份极其详尽、结构清晰的 Markdown 格式复习笔记。

        【资料内容】：
        {text}

        【重构与排版要求】：
        1. **极致清晰的结构**：必须使用多级标题 (#, ##, ###)、有序/无序列表、加粗等 Markdown 语法进行优美排版。
        2. **硬核知识提取**：绝对不要遗漏原资料中的任何核心定义、定理推导、重要结论及关键例题的解题逻辑。
        3. **严格公式保护**：资料中原有的数学、物理公式必须严格保留，并保持 LaTeX 语法不变（行间公式必须用 $$ 包裹，行内公式必须用 $ 包裹）。
        4. **智能语义缝合**：输入资料可能是由多张 OCR 识别的图片碎片拼凑而成，可能存在换行错乱或语句断裂。请你运用推理能力，将其缝合为逻辑通顺连贯的完整讲义。
        5. **纯净输出**：直接输出 Markdown 笔记正文！绝对不要包含任何诸如“好的，这是您的笔记”、“重构完成”之类的客套废话。
        """
        
        try:
            # 🌟 核心逻辑：明确调用 reasoning_model (慢脑)，并强制开启思考模式
            logger.info(f"🧠 正在调用慢脑进行深度精读总结 (文本长度: {len(text)} 字符)...")
            summary = await self._call_llm_text(
                prompt=prompt, 
                target_model=self.reasoning_model, 
            )
            
            if not summary or len(summary) < 20:
                logger.warning("⚠️ 慢脑精读返回内容过短或为空。")
                return "⚠️ 导师虽然进行了深度思考，但似乎未能从这份资料中提取出足够丰富的结构化笔记。请检查原资料是否清晰或包含有效的学术文字。"
                
            # 去除大模型可能自带的 markdown 代码块包裹 (防呆设计)
            import re
            cleaned_summary = re.sub(r"^```markdown\s*", "", summary, flags=re.IGNORECASE)
            cleaned_summary = re.sub(r"^```\s*", "", cleaned_summary)
            cleaned_summary = re.sub(r"\s*```$", "", cleaned_summary)
            
            return cleaned_summary.strip()
            
        except Exception as e:
            logger.error(f"深度精读归纳崩溃: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"❌ 深度精读任务发生异常，慢脑短路了：{e}"
        
    async def reprocess_and_tag_chunk(self, chunk_content: str, bad_concept: str, course_pool: List[str] = None) -> Dict:
        """【手术刀专属】带反向提示词的切片重塑与重新打标引擎"""
        if not course_pool:
            course_pool = ["微积分", "线性代数", "大学物理", "人工智能基础", "C语言", "Python", "常微分方程"]
        course_str = ", ".join(course_pool)

        prompt = f"""
        角色：极其严厉的学术知识图谱打标专家
        任务：你收到了一段被“污染”的学术文本切片。这段文本之前被错误地打上了【{bad_concept}】这个不恰当、太宽泛或完全错误的学术标签。
        你的任务是：
        1. 剥离【{bad_concept}】这个标签，用严谨的教科书级语言重写这段文本的核心干货。
        2. 重新审视重写后的文本，提取出它真正属于的细分学术概念，严格输出 JSON 格式。

        【⚠️ 重塑手术警告/反向提示】(最高优先级)：
        提取出来的新概念里，绝对、绝对、绝对禁止再次出现【{bad_concept}】这个词！必须去寻找更精确、更具体的学术名词！

        【一级学科池参考】：[{course_str}]

        【原始污染切片内容】：
        {chunk_content[:1500]}

        【要求输出的 JSON 结构】：
        {{
            "rewritten_text": "用严谨的学术语言重写并提纯后的切片文本，务必保留所有公式和推导过程。",
            "tags": {{
                "course": "必须严格从给出的【一级学科池参考】中选择一个",
                "course_explanation": "判断学科的理由",
                "concepts": ["更精确的学术概念1", "更精确的学术概念2"], // 提取1-4个最高度概括的学术名词，绝不能包含原来的错词
                "concepts_explanation": "简述为何这几个词最能代表切片核心"
            }},
            "pedagogy": {{
                "type": ["定义定理"], // 从 [定义定理, 实际应用, 练习, 拓展] 中限选1-2项
                "type_explanation": "判断教学性质的理由"
            }},
            "boundary": {{
                "completeness": "相对完整",
                "completeness_explanation": "判断边界状态的理由",
                "context_loss": false,
                "chapter_transition": false
            }}
        }}
        """
        try:
            # 🌟 换回快脑 (fast_model)：拒绝思维链啰嗦，追求极致 JSON 并发速度与稳定性！
            return await self._call_llm(prompt, target_model=self.fast_model)
        except Exception as e:
            import logging
            logging.error(f"重塑 Chunk Tags 失败: {e}")
            return {
                "rewritten_text": chunk_content, # 兜底：如果崩了，就用原文本
                "tags": {"course": "通用", "concepts": ["重塑兜底概念"]},
                "pedagogy": {"type": ["拓展"]},
                "boundary": {"completeness": "相对完整", "context_loss": False, "chapter_transition": False}
            }
        
    async def analyze_chunk_strategy(self, sample_text: str) -> Dict:
        """
        【新增】工程级文档探视：根据样例文本的知识密度、文本风格，动态推荐最佳切片参数
        """
        prompt = f"""
        你是一个顶级的 RAG 向量检索架构师。
        请分析以下从长文档中随机抽样的两段切片，评估其“知识密度”和“文本风格”，并给出最适合该文档的物理切片参数。
        
        【抽样切片】：
        {sample_text}
        
        【评估与参数设定法则】：
        1. 知识密度极高（如：满篇公式推导、密集名词定义、理科题目）：
           -> 建议减小切片以提高向量纯度 (chunk_size: 300~600)，知识密度越高切片越小；增加重叠度防止推导逻辑断裂 (overlap_size: 150~200)。
        2. 知识密度中等（如：标准教科书、科普文章）：
           -> 建议使用标准均衡参数 (chunk_size: 800, overlap_size: 150)。
        3. 知识密度较低/偏叙事（如：小说、历史传记、日常对话）：
           -> 建议大块切分以保留宏观上下文 (chunk_size: 1000~1200)；减小重叠度降低冗余 (overlap_size: 100)。
           
        【强制 JSON 输出】：
        {{
            "chunk_size": 800, // 整数值
            "overlap_size": 150, // 整数值
            "reason": "请用一句话(20字内)简述判定理由，例如：'包含大量硬核公式推导，需小块高重叠以防断裂。'"
        }}
        """
        try:
            # 🌟 使用推理模型 (慢脑) 进行探视
            res = await self._call_llm(prompt, target_model=self.reasoning_model, force_json=True)
            return res
        except Exception as e:
            from astrbot.api.all import logger
            logger.warning(f"⚠️ 动态切片策略推演失败，将使用默认参数: {e}")
            return {"chunk_size": 800, "overlap_size": 150, "reason": "探视失败，采用保守标准策略"}