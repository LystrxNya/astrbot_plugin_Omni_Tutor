import os
import time
import re
import html
import markdown
from playwright.async_api import async_playwright

class MarkdownRenderer:
    def __init__(self, output_dir="data/output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 🌟 HTML 模板：大字号、宽字距、KaTeX 公式增强 + 🌟 Chromium 原生 Zoom 自适应缩放
        self.html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
                    line-height: 1.8;       
                    font-size: 30px;        
                    letter-spacing: 0.05em; 
                    color: #24292e; 
                    padding: 30px;          
                    max-width: 700px;       
                    margin: 0 auto; 
                    background-color: #ffffff; 
                    word-wrap: break-word;
                }
                h1 { font-size: 1.4em; margin-top: 40px; border-bottom: 3px solid #eee; padding-bottom: 0.3em; }
                h2 { font-size: 1.25em; margin-top: 35px; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }
                h3 { font-size: 1.15em; margin-top: 30px; }
                
                /* 代码块样式 */
                code { font-size: 0.85em; background: #f6f8fa; padding: 4px 8px; border-radius: 6px; font-family: monospace; }
                pre { background: #f6f8fa; padding: 25px; border-radius: 12px; line-height: 1.5; overflow: auto; }
                
                /* 表格样式 */
                table { border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.8em; }
                table th, table td { border: 1px solid #dfe2e5; padding: 10px 15px; }
                table tr:nth-child(2n) { background-color: #f6f8fa; }
                
                /* 🌟 KaTeX 公式显著放大 */
                .katex { font-size: 1.25em !important; } 
                .katex-display { 
                    margin: 1.2em 0; 
                    padding: 10px 0; 
                    text-align: center;
                }
            </style>
            
            <link rel="stylesheet" href="https://cdn.staticfile.net/KaTeX/0.16.9/katex.min.css">
            <script defer src="https://cdn.staticfile.net/KaTeX/0.16.9/katex.min.js"></script>
            <script defer src="https://cdn.staticfile.net/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
        </head>
        <body>
            <div id="content">
                {{ content }}
            </div>
            
            <script>
                document.addEventListener("DOMContentLoaded", function() {
                    // 兜底超时设置，防止阻塞 Playwright
                    setTimeout(() => { window.mathRendered = true; }, 3000);
                    
                    if (typeof renderMathInElement !== 'undefined') {
                        // 1. 先进行标准渲染
                        renderMathInElement(document.body, {
                            delimiters: [
                                {left: '$$', right: '$$', display: true},
                                {left: '$', right: '$', display: false}
                            ],
                            throwOnError : false
                        });
                        
                        // 2. 🌟 核心魔法：使用 Chromium 原生 zoom 进行超宽自适应缩放
                        document.querySelectorAll('.katex-display').forEach(function(el) {
                            var katexElement = el.querySelector('.katex');
                            if (katexElement) {
                                var containerWidth = el.clientWidth;
                                var mathWidth = katexElement.scrollWidth;
                                
                                // 如果公式撑爆了屏幕
                                if (mathWidth > containerWidth) {
                                    // 计算缩放比例 (留出 5% 的左右安全边距)
                                    var scale = (containerWidth / mathWidth) * 0.95; 
                                    
                                    // 运用浏览器原生的 zoom 属性，完美重算排版宽度，杜绝右侧截断
                                    el.style.zoom = scale; 
                                }
                            }
                        });
                    }
                    // 通知 Playwright 渲染和缩放彻底完成，可以截图了
                    window.mathRendered = true; 
                });
            </script>
        </body>
        </html>
        """

    def _protect_math(self, md_text):
        """保护公式 (终极无敌版：自动修复缺漏的 $$，并兼容 KaTeX 的 aligned)"""
        math_blocks = []
        
        def convert_to_aligned(text):
            """将不被 KaTeX 支持的独立 align 环境强转为 aligned"""
            text = text.replace(r"\begin{align*}", r"\begin{aligned}")
            text = text.replace(r"\begin{align}", r"\begin{aligned}")
            text = text.replace(r"\end{align*}", r"\end{aligned}")
            text = text.replace(r"\end{align}", r"\end{aligned}")
            return text

        def save_math(match):
            content = convert_to_aligned(match.group(0))
            math_blocks.append(content)
            return f"@@MATH_BLOCK_{len(math_blocks)-1}@@"

        def save_and_wrap_env(match):
            content = convert_to_aligned(match.group(0))
            # LLM 裸写的公式环境，强行包上 $$
            math_blocks.append("$$\n" + content + "\n$$")
            return f"@@MATH_BLOCK_{len(math_blocks)-1}@@"
            
        def save_bracket_math(match):
            # 兼容大模型有时会输出 \[ ... \] 这种块级公式格式
            content = match.group(1)
            math_blocks.append("$$\n" + content + "\n$$")
            return f"@@MATH_BLOCK_{len(math_blocks)-1}@@"

        # 1. 抓取标准听话的 $$...$$ 
        text = re.sub(r'\$\$.*?\$\$', save_math, md_text, flags=re.DOTALL)
        
        # 2. 抓取裸露的 \begin{}...\end{} (模型漏加了 $$，强行补上)
        text = re.sub(r'\\begin\{([a-zA-Z*]+)\}.*?\\end\{\1\}', save_and_wrap_env, text, flags=re.DOTALL)
        
        # 3. 抓取标准的 \[ ... \] 块公式
        text = re.sub(r'\\\[(.*?)\\\]', save_bracket_math, text, flags=re.DOTALL)
        
        # 4. 最后抓取行内公式 $...$
        text = re.sub(r'\$.*?\$', save_math, text)
        
        return text, math_blocks

    def _restore_math(self, html_text, math_blocks):
        """还原公式"""
        for i, block in enumerate(math_blocks):
            safe_block = html.escape(block)
            html_text = html_text.replace(f"@@MATH_BLOCK_{i}@@", safe_block)
        return html_text

    def _strip_emojis(self, text):
        """🌟 终极防御：在 Markdown 解析前过滤掉会导致方框的特殊 Emoji 字符"""
        if not text: return text
        try:
            # 采用高精度正则表达式，过滤绝大部分 Emoji 和符号
            emoji_pattern = re.compile(
                '['
                '\U00010000-\U0010ffff'  # Unicode emoji 核心区域
                ']+', 
                flags=re.UNICODE
            )
            return emoji_pattern.sub(r'', text)
        except Exception as e:
            print(f"[Renderer] Emoji过滤失败: {e}")
            return text

    async def render_to_image(self, md_text: str):
        """主渲染流程"""
        try:
            # 🌟 第0步：Emoji 净化
            clean_md = self._strip_emojis(md_text)

            # 1. 公式保护 (使用净化后的 Markdown，自动修正残缺 LaTeX)
            safe_md, math_blocks = self._protect_math(clean_md)
            
            # 2. 转换为 HTML
            html_body = markdown.markdown(safe_md, extensions=['fenced_code', 'tables'])
            
            # 3. 还原公式
            html_body = self._restore_math(html_body, math_blocks)
            
            # 4. 填充模板
            full_html = self.html_template.replace("{{ content }}", html_body)
            
            output_path = os.path.join(self.output_dir, f"render_{int(time.time()*100)}.png")
            
            async with async_playwright() as p:
                browser = None
                try:
                    browser = await p.chromium.launch(headless=True, channel="msedge", args=['--no-sandbox', '--disable-setuid-sandbox'])
                except Exception:
                    browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
                
                page = await browser.new_page(viewport={"width": 700, "height": 600}) 
                await page.set_content(full_html, wait_until="domcontentloaded", timeout=4000)
                
                try:
                    # 等待我们的自适应缩放引擎完成工作
                    await page.wait_for_function("window.mathRendered === true", timeout=1500)
                except:
                    print("[Renderer] 公式渲染脚本超时。")
                    
                await page.locator("body").screenshot(path=output_path)
                await browser.close()
                
            return True, output_path
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return False, error_trace