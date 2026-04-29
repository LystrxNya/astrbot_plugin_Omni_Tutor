import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Union

class SQLiteManager:
    def __init__(self, db_path: str = "data/omnitutor_sqlite.db"):
        """
        初始化支持倒排索引（Inverted Index）、动态概念池及 MIU 碎片的 SQLite 数据库
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;") # 强制开启 SQLite 外键级联删除
        return conn

    def _init_db(self):
        """初始化表结构：全新解耦倒排索引架构"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # ================= 1. 主脑：全知导师系统表 =================
            # --- 基础文档存储表 ---
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT UNIQUE NOT NULL,
                    full_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 🌟 恢复 pedagogy_type 用于意图过滤，同时保留 course 供宏观管理和洗牌
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    course TEXT,            -- 保留学科分类
                    pedagogy_type TEXT,     -- 🌟 存 JSON 数组字符串，用于教学意图过滤
                    boundary_status TEXT,   
                    context_loss INTEGER DEFAULT 0,
                    chapter_transition INTEGER DEFAULT 0,
                    FOREIGN KEY (source_name) REFERENCES documents(source_name) ON DELETE CASCADE
                )
            ''')
            
            # 🌟 架构灵魂：倒排索引桥梁表 (多对多映射)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunk_tag_mapping (
                    chunk_id TEXT NOT NULL,
                    tag_name TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, tag_name),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                )
            ''')
            # 建立索引，加速基于 tag 的并发拉取
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag_name ON chunk_tag_mapping(tag_name)')

            # --- 全局动态概念池 (Tags) ---
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concept_pool (
                    course TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (course, concept)
                )
            ''')

            # --- 动态认知追踪表 ---
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_cognition (
                    uid TEXT NOT NULL,
                    course TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    query_count INTEGER DEFAULT 1,
                    mastery_level REAL DEFAULT 0.1,
                    struggle_index REAL DEFAULT 0.0,
                    last_interact TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(uid, course, concept)
                )
            ''')

            # ================= 2. 第二脑：独立私人助理表 =================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS important_miu (
                    chunk_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

    # ================= 倒排索引：数据存取方法 (核心重构区) =================

    def save_document(self, source_name: str, full_text: str, chunks_data: List[Dict], chunk_ids: List[str]):
        """
        保存长篇原文档，物理切片落盘，并建立倒排索引映射
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents (source_name, full_text, created_at)
                VALUES (?, ?, ?)
            ''', (source_name, full_text, datetime.now()))
            
            for i, (chunk_id, data) in enumerate(zip(chunk_ids, chunks_data)):
                content = data.get("text", "")
                tags = data.get("tags", {})
                course = tags.get("course", "通用")
                concepts = tags.get("concepts", [])  # 这里是切片关联的所有 tags 数组
                
                pedagogy = data.get("pedagogy", {})
                pedagogy_type = pedagogy.get("type", ["拓展"])
                
                boundary = data.get("boundary", {})
                boundary_status = boundary.get("completeness", "相对完整")
                context_loss = 1 if boundary.get("context_loss", False) else 0
                chapter_transition = 1 if boundary.get("chapter_transition", False) else 0
                
                # 1. 物理切片落库 (🌟 新增 pedagogy_type 字段插入)
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, source_name, chunk_index, content, 
                        course, pedagogy_type, boundary_status, context_loss, chapter_transition
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk_id, source_name, i, content, 
                    course, json.dumps(pedagogy_type, ensure_ascii=False),
                    boundary_status, context_loss, chapter_transition
                ))

                # 2. 建立倒排索引桥梁！(解耦概念绑定)
                for concept in concepts:
                    cursor.execute('''
                        INSERT OR IGNORE INTO chunk_tag_mapping (chunk_id, tag_name)
                        VALUES (?, ?)
                    ''', (chunk_id, concept))
            conn.commit()

    def get_chunks_by_tags(self, tag_names: List[str]) -> List[Dict]:
        """
        🌟 概念路由召回 (第一路召回)：
        传入命中的 Tag 列表，直接拉出所有绑定了这些 Tag 的切片原文及元数据
        """
        if not tag_names:
            return []
            
        placeholders = ','.join(['?'] * len(tag_names))
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 🌟 拉取数据时携带 course 和 pedagogy_type
            cursor.execute(f'''
                SELECT DISTINCT c.chunk_id, c.content, c.source_name, c.course, c.pedagogy_type, c.boundary_status, c.context_loss, c.chapter_transition
                FROM chunks c
                JOIN chunk_tag_mapping m ON c.chunk_id = m.chunk_id
                WHERE m.tag_name IN ({placeholders})
            ''', tag_names)
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "chunk_id": r[0],
                    "text": r[1],
                    "metadata": {
                        "source": r[2],
                        "course": r[3],
                        "pedagogy_type": r[4], # 🌟 暴露给 TutorBrain 做意图过滤
                        "boundary_status": r[5],
                        "context_loss": r[6],
                        "chapter_transition": r[7]
                    }
                })
            return results

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """
        🌟 纯语义向量召回 (第二路召回)：
        根据向量库盲捞的 chunk_id，回表拉取明文
        """
        if not chunk_ids:
            return []
            
        placeholders = ','.join(['?'] * len(chunk_ids))
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 🌟 同样带上 pedagogy_type
            cursor.execute(f'''
                SELECT chunk_id, content, source_name, course, pedagogy_type, boundary_status, context_loss, chapter_transition
                FROM chunks
                WHERE chunk_id IN ({placeholders})
            ''', chunk_ids)
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "chunk_id": r[0],
                    "text": r[1],
                    "metadata": {
                        "source": r[2],
                        "course": r[3],
                        "pedagogy_type": r[4], # 🌟 暴露给 TutorBrain 做意图过滤
                        "boundary_status": r[5],
                        "context_loss": r[6],
                        "chapter_transition": r[7]
                    }
                })
            return results

    def get_surrounding_context_by_id(self, chunk_id: str, window: int = 1) -> str:
        """根据 chunk_id 查找物理上相邻的前文和后文进行无缝缝合"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT source_name, chunk_index FROM chunks WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            if not row: return ""
            
            source_name, chunk_index = row
            cursor.execute('''
                SELECT content FROM chunks 
                WHERE source_name = ? AND chunk_index >= ? AND chunk_index <= ?
                ORDER BY chunk_index ASC
            ''', (source_name, chunk_index - window, chunk_index + window))
            rows = cursor.fetchall()
            return "\n\n".join([r[0] for r in rows]) if rows else ""

    # ================= 动态概念池与认知管理 =================

    def update_concept_pool(self, course: str, concepts: List[str]):
        """将新提取的概念加入概念池或增加词频"""
        if not course or not concepts: return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for concept in concepts:
                cursor.execute('''
                    INSERT INTO concept_pool (course, concept, frequency, last_seen)
                    VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(course, concept) DO UPDATE SET
                        frequency = frequency + 1,
                        last_seen = CURRENT_TIMESTAMP
                ''', (course, concept))
            conn.commit()

    def get_concept_pool(self, course: str, limit: int = 50) -> List[str]:
        if not course: return []
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT concept FROM concept_pool 
                WHERE course = ? ORDER BY frequency DESC LIMIT ?
            ''', (course, limit))
            return [row[0] for row in cursor.fetchall()]

    def update_user_cognition(self, uid: str, course: str, concepts: List[str], mastery_delta: float, struggle_delta: float):
        if not concepts: return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for concept in concepts:
                cursor.execute('''
                    INSERT INTO user_cognition (uid, course, concept, query_count, mastery_level, struggle_index, last_interact)
                    VALUES (?, ?, ?, 1, MAX(0.0, MIN(1.0, 0.1 + ?)), MAX(0.0, ?), CURRENT_TIMESTAMP)
                    ON CONFLICT(uid, course, concept) DO UPDATE SET
                        query_count = query_count + 1,
                        mastery_level = MAX(0.0, MIN(1.0, mastery_level + ?)),
                        struggle_index = MAX(0.0, struggle_index + ?),
                        last_interact = CURRENT_TIMESTAMP
                ''', (uid, course, concept, mastery_delta, struggle_delta, mastery_delta, struggle_delta))
            conn.commit()

    # ================= 其他运维方法与 MIU =================
    
    def delete_document(self, source_name: str):
        """级联清理：删除 documents 会自动销毁对应的 chunks 和 chunk_tag_mapping！"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            if cursor.fetchone():
                cursor.execute('DELETE FROM documents WHERE source_name = ?', (source_name,))
                
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='important_miu'")
            if cursor.fetchone():
                cursor.execute('DELETE FROM important_miu WHERE source_name = ?', (source_name,))
            conn.commit()

    def get_user_cognition_data(self, uid: str) -> List[Dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT course, concept, query_count, mastery_level, struggle_index, last_interact FROM user_cognition WHERE uid = ?', (uid,))
            return [{"course": r[0], "concept": r[1], "count": r[2], "mastery": r[3], "struggle": r[4], "last_interact": r[5]} for r in cursor.fetchall()]

    def get_all_knowledge_concepts(self) -> List[Dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT course, concept, frequency FROM concept_pool ORDER BY course, frequency DESC')
            return [{"course": r[0], "concept": r[1], "frequency": r[2]} for r in cursor.fetchall()]

    def delete_concept_by_name(self, concept: str) -> int:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM concept_pool WHERE concept = ?', (concept,))
            count = cursor.rowcount
            cursor.execute('DELETE FROM user_cognition WHERE concept = ?', (concept,))
            # 顺便清洗一遍桥梁表
            cursor.execute('DELETE FROM chunk_tag_mapping WHERE tag_name = ?', (concept,))
            conn.commit()
            return count

    def reset_mastery_by_name(self, concept: str) -> int:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE user_cognition SET mastery_level = 0.1, struggle_index = 0.0, query_count = 0 WHERE concept = ?', (concept,))
            count = cursor.rowcount
            conn.commit()
            return count
            
    def get_all_source_names(self) -> list:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            sources = set()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            if cursor.fetchone():
                cursor.execute('SELECT DISTINCT source_name FROM documents')
                sources.update([row[0] for row in cursor.fetchall()])
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='important_miu'")
            if cursor.fetchone():
                cursor.execute('SELECT DISTINCT source_name FROM important_miu')
                sources.update([row[0] for row in cursor.fetchall()])
            return sorted(list(sources), reverse=True)

    def save_important_miu(self, chunk_id: str, title: str, content: str, source_name: str):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO important_miu (chunk_id, title, content, source_name) VALUES (?, ?, ?, ?)', (chunk_id, title, content, source_name))
            conn.commit()

    def get_important_miu(self, chunk_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT title, content, source_name FROM important_miu WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            if row: return {"title": row[0], "content": row[1], "source": row[2]}
            return None
            
    def get_all_mius(self) -> list:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='important_miu'")
            if not cursor.fetchone(): return []
            cursor.execute('SELECT title, source_name FROM important_miu ORDER BY created_at DESC')
            return [{"title": r[0], "source": r[1]} for r in cursor.fetchall()]
        
    def delete_miu_by_title(self, title: str) -> list:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT chunk_id FROM important_miu WHERE title = ?', (title,))
            chunk_ids = [r[0] for r in cursor.fetchall()]
            if chunk_ids:
                cursor.execute('DELETE FROM important_miu WHERE title = ?', (title,))
                conn.commit()
            return chunk_ids

    # ================= 后台图谱归一化 (史诗级优化) =================
    
    def get_all_unique_courses(self) -> List[str]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT course FROM concept_pool")
            return [r[0] for r in cursor.fetchall()]

    def get_all_unique_concepts(self) -> List[str]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT concept FROM concept_pool")
            return [r[0] for r in cursor.fetchall()]

    def apply_normalization_mapping(self, course_map: dict, concept_map: dict) -> int:
        """
        🚀 重构后的光速洗牌：利用数据库级联操作与数据融合机制 (完美适配真实字段名)
        """
        merged_count = len(course_map) + len(concept_map)
        if merged_count == 0: return 0
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # ==========================================
            # 1. 替换二级（Course）
            # ==========================================
            for old_c, new_c in course_map.items():
                cursor.execute("UPDATE chunks SET course = ? WHERE course = ?", (new_c, old_c))
                
                # 学科合并：融合用户的认知数据 (🌟 修正字段名：mastery_level, struggle_index, query_count)
                cursor.execute("""
                    INSERT INTO user_cognition (uid, course, concept, query_count, mastery_level, struggle_index, last_interact)
                    SELECT uid, ?, concept, query_count, mastery_level, struggle_index, last_interact FROM user_cognition WHERE course = ?
                    ON CONFLICT(uid, course, concept) DO UPDATE SET 
                        query_count = user_cognition.query_count + excluded.query_count,
                        mastery_level = (user_cognition.mastery_level + excluded.mastery_level) / 2.0,
                        struggle_index = MAX(user_cognition.struggle_index, excluded.struggle_index),
                        last_interact = MAX(user_cognition.last_interact, excluded.last_interact);
                """, (new_c, old_c))
                cursor.execute("DELETE FROM user_cognition WHERE course = ?", (old_c,))
                
                # 学科合并：融合概念池词频
                cursor.execute("""
                    INSERT INTO concept_pool (course, concept, frequency)
                    SELECT ?, concept, frequency FROM concept_pool WHERE course = ?
                    ON CONFLICT(course, concept) DO UPDATE SET frequency = concept_pool.frequency + excluded.frequency;
                """, (new_c, old_c))
                cursor.execute("DELETE FROM concept_pool WHERE course = ?", (old_c,))
            
            # ==========================================
            # 2. 替换三级（Concepts - 在倒排桥梁表中更新）
            # ==========================================
            for old_c, new_c in concept_map.items():
                # 桥梁表：纯关联关系，直接 IGNORE 然后删旧关系
                cursor.execute('''
                    INSERT OR IGNORE INTO chunk_tag_mapping (chunk_id, tag_name)
                    SELECT chunk_id, ? FROM chunk_tag_mapping WHERE tag_name = ?
                ''', (new_c, old_c))
                cursor.execute("DELETE FROM chunk_tag_mapping WHERE tag_name = ?", (old_c,))
                
                # 🌟 修复点：同步认知表 (修正字段名，并安全合并查询次数)
                cursor.execute("""
                    INSERT INTO user_cognition (uid, course, concept, query_count, mastery_level, struggle_index, last_interact)
                    SELECT uid, course, ?, query_count, mastery_level, struggle_index, last_interact FROM user_cognition WHERE concept = ?
                    ON CONFLICT(uid, course, concept) DO UPDATE SET 
                        query_count = user_cognition.query_count + excluded.query_count,
                        mastery_level = (user_cognition.mastery_level + excluded.mastery_level) / 2.0,
                        struggle_index = MAX(user_cognition.struggle_index, excluded.struggle_index),
                        last_interact = MAX(user_cognition.last_interact, excluded.last_interact);
                """, (new_c, old_c))
                cursor.execute("DELETE FROM user_cognition WHERE concept = ?", (old_c,))
                
                # 概念池词频合并
                cursor.execute("""
                    INSERT INTO concept_pool (course, concept, frequency)
                    SELECT course, ?, frequency FROM concept_pool WHERE concept = ?
                    ON CONFLICT(course, concept) DO UPDATE SET frequency = concept_pool.frequency + excluded.frequency;
                """, (new_c, old_c))
                cursor.execute("DELETE FROM concept_pool WHERE concept = ?", (old_c,))
            
            conn.commit()
            
        return merged_count
    
    def clean_and_recalc_concepts(self) -> List[str]:
        """
        🧹 扫地僧机制：清理孤儿概念，并精准重置存活概念的词频
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 1. 揪出所有失去了切片关联的“孤儿节点”
            cursor.execute('''
                SELECT concept FROM concept_pool 
                WHERE concept NOT IN (SELECT DISTINCT tag_name FROM chunk_tag_mapping)
            ''')
            orphans = [r[0] for r in cursor.fetchall()]
            
            # 2. 物理抹除孤儿节点及认知追踪记录
            if orphans:
                placeholders = ','.join(['?'] * len(orphans))
                cursor.execute(f'DELETE FROM concept_pool WHERE concept IN ({placeholders})', orphans)
                cursor.execute(f'DELETE FROM user_cognition WHERE concept IN ({placeholders})', orphans)
                
            # 3. 完美修正图谱词频（根据当前桥梁表中真实的连线数量重新算分）
            cursor.execute('''
                UPDATE concept_pool
                SET frequency = (
                    SELECT COUNT(*) FROM chunk_tag_mapping 
                    WHERE chunk_tag_mapping.tag_name = concept_pool.concept
                )
            ''')
            # 把词频被清零的冗余数据兜底删掉（双重保险）
            cursor.execute('DELETE FROM concept_pool WHERE frequency = 0')
            
            conn.commit()
            return orphans
        
    def get_latest_source_name(self) -> str:
        """
        获取数据库中最新录入的一条 source_name (用于撤销功能)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 按照创建时间倒序排列，取第一条
            cursor.execute('SELECT source_name FROM documents ORDER BY created_at DESC LIMIT 1')
            row = cursor.fetchone()
            return row[0] if row else ""