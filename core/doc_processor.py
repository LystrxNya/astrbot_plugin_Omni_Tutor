import re

class DocumentProcessor:
    def __init__(self, chunk_size=800, overlap_size=150):
        """
        :param chunk_size: 每个文本块的最大字符数（尽量贴合大模型的最佳理解窗口）
        :param overlap_size: 相邻文本块的重叠字符数（防止段落交界处的语义断裂）
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def is_exercise_chunk(self, text: str) -> bool:
        """
        🌟 零 AI 正则匹配：判断当前文本块是否包含题目/例题特征
        """
        # 1. 强特征词汇匹配（位于行首或独立成段）
        strong_pattern = r"(?:^|\n)(?:【?例题?】?\s*\d*|已知|求证|求解|证明|计算|解答|解析|答：|解：|习题\s*\d+|练习\s*\d+)"
        if re.search(strong_pattern, text):
            return True
            
        # 2. 弱特征组合匹配（包含问号，且包含理科疑问词）
        if "?" in text or "？" in text:
            if any(kw in text for kw in ["求", "多少", "何值", "证明", "化简"]):
                return True
                
        return False

    def process(self, text: str, dynamic_chunk_size: int = None, dynamic_overlap_size: int = None) -> list[dict]:
        """
        主入口：将长篇 Markdown 文本切分为安全的 Chunk 列表
        🌟 返回格式已升级为附带元数据的字典列表：[{"text": "内容", "is_exercise": True/False}, ...]
        🌟 新增支持：可动态传入探视器推荐的切片策略，若无则使用默认参数
        """
        if not text:
            return []
            
        # 🌟 采用动态参数，若未传入则使用初始化时的默认值
        target_chunk_size = dynamic_chunk_size or self.chunk_size
        target_overlap_size = dynamic_overlap_size or self.overlap_size

        # 1. 统一换行符
        text = text.replace('\r\n', '\n')

        # 2. 按照双换行符（段落）进行初步切分
        raw_blocks = re.split(r'\n\n+', text)
        
        # 3. 合并碎片：修复被切断的 LaTeX 公式块
        safe_blocks = self._merge_math_blocks(raw_blocks)

        # 4. 组装最终的 Chunks
        chunks = []
        current_chunk = ""

        for block in safe_blocks:
            if not current_chunk:
                current_chunk = block
                continue

            # 预测如果加入当前 block，长度是否超标
            predicted_length = len(current_chunk) + 2 + len(block)

            # 🌟 使用动态的 target_chunk_size 进行边界判断
            if predicted_length <= target_chunk_size:
                current_chunk += "\n\n" + block
            else:
                # 🌟 超标了！保存当前 chunk 并打上是否为题目的 Tag
                chunks.append({
                    "text": current_chunk,
                    "is_exercise": self.is_exercise_chunk(current_chunk)
                })
                
                # 处理 Overlap（滑动窗口重叠）：从刚才保存的块尾部提取安全的段落，作为新块的开头
                # 🌟 透传 target_overlap_size
                overlap_text = self._extract_safe_overlap(current_chunk, target_overlap_size)
                current_chunk = overlap_text + "\n\n" + block if overlap_text else block

        # 将最后剩下的部分保存
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "is_exercise": self.is_exercise_chunk(current_chunk)
            })

        return chunks

    def _merge_math_blocks(self, blocks: list[str]) -> list[str]:
        """
        核心力场：确保 $$ $$ 公式块绝对不会被 \n\n 切断
        """
        safe_blocks = []
        buffer = []
        in_math = False

        for block in blocks:
            buffer.append(block)
            
            # 统计当前段落里 $$ 出现的次数
            math_count = block.count('$$')
            
            # 如果出现奇数次，说明我们正在跨越公式的边界（进入或离开）
            if math_count % 2 != 0:
                in_math = not in_math

            # 如果当前不在公式块内部，说明 buffer 积攒的内容是一个安全的完整逻辑块
            if not in_math:
                safe_blocks.append("\n\n".join(buffer))
                buffer = []

        # 兜底防爆：如果解析到最后，发现 in_math 还是 True 
        # (通常是因为 OCR 偶尔抽风漏识别了结尾的 $$)，强行闭合保存，防止数据丢失
        if buffer:
            safe_blocks.append("\n\n".join(buffer))

        return safe_blocks

    def _extract_safe_overlap(self, chunk_text: str, overlap_size: int = None) -> str:
        """
        从块尾部提取重叠文本，且保证是按“段落”提取，不会切碎句子
        """
        # 🌟 兜底：如果没传 overlap_size，使用类初始化时的默认值
        target_overlap = overlap_size if overlap_size is not None else self.overlap_size
        
        blocks = chunk_text.split("\n\n")
        overlap_length = 0
        overlap_blocks = []

        # 从后往前取段落
        for block in reversed(blocks):
            # 如果加入这个段落会导致重叠区过长，且我们已经拿到了一些重叠文本，就停止
            if overlap_length + len(block) > target_overlap and overlap_blocks:
                break
            
            overlap_blocks.insert(0, block)
            overlap_length += len(block) + 2

        return "\n\n".join(overlap_blocks)