import base64
import re
import asyncio
import fitz  # PyMuPDF
from openai import AsyncOpenAI

class QwenOCREngine:
    def __init__(self, api_key: str, ocr_model: str = "qwen-vl-ocr-2025-11-20", vision_model: str = "qwen-vl-max"):
        """
        初始化双核视觉引擎 (纯异步升级版)
        """
        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.ocr_model = ocr_model
        self.vision_model = vision_model
        
        # ================= 引擎 1：纯文本与公式提取 =================
        self.ocr_sys_prompt = "你是一个严谨的数学排版专家和 OCR 引擎。你的任务是将提供的图像完美无损地转换为包含 LaTeX 公式的 Markdown 文本。"
        self.ocr_user_prompt = (
            "请精确转录图片内容。规则：\n"
            "1. 强制使用 $ 包裹行内公式，$$ 包裹行间公式。禁止使用 \\( 或 \\[。\n"
            "2. 严谨区分 f 和 r，区分 alpha 和 a。\n"
            "3. 直接输出结果，不要包含 ```markdown 代码块，忽略水印和页码。"
        )

        # ================= 引擎 2：几何/物理图形逻辑分析 =================
        self.vision_sys_prompt = "你是一个严谨的理科助教。你的任务是观察图片中的数理图形，并用文字详细描述其结构，以便没有任何视觉能力的大模型也能‘听懂’这张图。"
        self.vision_user_prompt = (
            "请仔细观察图片中是否包含几何图形、函数图像、物理受力图、电路图或坐标系。\n"
            "【任务】：如果有图形，请用极度严谨的文字描述它的结构、点线面关系、已知角度/长度、运动方向等所有视觉可见的变量标注。\n"
            "【注意】：\n"
            "1. 不要转录图片里大段的题目文本，你的眼睛只盯着『图』！\n"
            "2. 如果图片只是纯文本题目，完全没有任何几何/物理图形，请只回复四个字：'无逻辑图形'。"
        )

    def _clean_text(self, raw_text: str) -> str:
        """清洗大模型的冗余输出"""
        if not raw_text: return ""
        cleaned = re.sub(r"^```markdown\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"(?m)^.*@.*$\n?", "", cleaned) 
        return cleaned.strip()

    async def _call_model(self, model_name: str, sys_prompt: str, user_prompt: str, base64_image: str) -> str:
        """底层纯异步 API 调用器"""
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
            completion = await self.client.chat.completions.create(
                model=model_name, 
                messages=messages, 
                temperature=0.01,
                top_p=0.01
            )
            return self._clean_text(completion.choices[0].message.content)
        except Exception as e:
            return f"[调用 {model_name} 失败: {str(e)}]"

    async def _process_single_base64(self, base64_image: str, enable_vision: bool) -> str:
        """处理单张 base64 图片的核心逻辑"""
        if enable_vision:
            ocr_task = self._call_model(self.ocr_model, self.ocr_sys_prompt, self.ocr_user_prompt, base64_image)
            vision_task = self._call_model(self.vision_model, self.vision_sys_prompt, self.vision_user_prompt, base64_image)
            
            ocr_result, vision_result = await asyncio.gather(ocr_task, vision_task)

            final_output = f"[原文与公式提取]:\n{ocr_result}\n"
            if "无逻辑图形" not in vision_result and len(vision_result) > 10:
                final_output += f"\n[辅助图形描述 (仅供无视觉模型参考)]:\n{vision_result}\n"
            return final_output
        else:
            ocr_result = await self._call_model(self.ocr_model, self.ocr_sys_prompt, self.ocr_user_prompt, base64_image)
            return f"[原文与公式提取]:\n{ocr_result}\n"

    async def process_file(self, file_path: str, page_num: int = 0, zoom: float = 3.0, enable_vision: bool = True) -> str:
        """
        🚀 修复版：精准物理切片，每次只处理指定的一页，不再自作主张遍历全书！
        """
        file_ext = file_path.lower()
        
        # 🌟 PDF 分支：只提取被要求的那一页 (page_num)
        if file_ext.endswith('.pdf'):
            doc = fitz.open(file_path)
            
            # 防御性判断：防止外部传入了越界的页码
            if page_num >= len(doc):
                doc.close()
                return ""
                
            page = doc.load_page(page_num)
            mat = fitz.Matrix(zoom, zoom) 
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("jpg")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            doc.close()
            
            return await self._process_single_base64(base64_image, enable_vision)
            
        # 普通图片分支
        with open(file_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        return await self._process_single_base64(base64_image, enable_vision)

    # 🗑️ _process_full_pdf 方法已被完全删除！
    # 因为在你的 tutor_brain.py 中，已经写了非常完美的 Semaphore(3) 和 asyncio.gather 来安全地调度并发了！