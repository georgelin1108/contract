import streamlit as st
import os
import re
import json
import uuid
import hashlib
import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

from pymongo import MongoClient
import chromadb
from chromadb.utils import embedding_functions
import ollama
from pypdf import PdfReader
from docx import Document as DocxReader
from docxtpl import DocxTemplate


# 系統設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

UPLOAD_DIR = os.path.abspath("./uploaded_files")
CHROMA_DIR = os.path.abspath("./chroma_db")
MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "GlobalLifeContractDB"

TOPIC_KEYWORDS = {
    "維護時間": ["維護時間", "服務時間", "基本時間", "星期一", "週一", "週五", "9:00", "18:00"],
    "維護人力": ["人力", "工程師", "窗口", "資深工程師", "實習生", "專責"],
    "異常回覆時限": ["異常", "當機", "回覆", "需求反應單", "工作天", "4小時"],
    "修復時限": ["修復", "排除故障", "恢復運作", "24小時"],
    "違約金": ["違約金", "懲罰性違約金", "千分之", "萬分之", "上限", "罰則"],
    "保密與開源": ["保密", "機密", "揭露", "公開", "GitHub", "開源", "第三人"],
    "管轄法院": ["管轄法院", "第一審", "地方法院", "中華民國法律", "準據法", "加州"],
    "智慧財產權": ["智慧財產權", "著作財產權", "原始碼", "系統設計", "衍生著作", "成果歸屬"],
    "保密期間": ["保密期間", "終止後", "解除後", "五年", "5年"],
    "損害賠償": ["損害賠償", "直接損害", "間接損害", "完全賠償"],
    "付款價金": ["費用", "價金", "付款", "匯款", "金額", "總額"],
    "交付驗收": ["交付", "驗收", "測試版", "修改", "Bug", "bug", "跑版"],
}

TOPIC_ALIAS = {
    "維護時間": ["服務時間", "基本維護時間", "服務水準"],
    "維護人力": ["人力配置", "專責窗口", "資深工程師"],
    "異常回覆時限": ["回覆時限", "SLA回覆"],
    "修復時限": ["排除故障", "恢復運作"],
    "違約金": ["懲罰性違約金", "罰則"],
    "保密與開源": ["保密義務", "開源宣告", "公開揭露"],
    "管轄法院": ["法院", "準據法", "爭議解決"],
    "智慧財產權": ["著作財產權", "成果歸屬"],
    "保密期間": [],
    "損害賠償": [],
    "付款價金": ["維護費用", "付款條件"],
    "交付驗收": ["交付條件", "驗收條件"],
}


# 基本工具
def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def ensure_dirs():
    ensure_upload_dir()
    os.makedirs(CHROMA_DIR, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def make_output_path(filename: str) -> str:
    ensure_upload_dir()
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def short_text(text: str, limit: int = 320) -> str:
    text = normalize_text(text)
    return text if len(text) <= limit else text[:limit] + "..."


def safe_json_load(s: str) -> Dict[str, Any]:
    if not s:
        return {}

    s = re.sub(r"```json\s*", "", s)
    s = re.sub(r"```", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except Exception:
            pass
    return {}


def chunk_text(text: str, chunk_size: int = 650, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def normalize_topic_name(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        return ""
    for canonical, aliases in TOPIC_ALIAS.items():
        if topic == canonical:
            return canonical
        for alias in aliases:
            if topic == alias or alias in topic or topic in alias:
                return canonical
    return topic


def detect_contract_mode_from_text(text: str) -> str:
    text = text or ""
    flags = {
        "維護": any(k in text for k in ["維護", "維運", "故障", "SLA", "修復"]),
        "開發": any(k in text for k in ["開發", "系統設計", "原始碼", "程式碼", "平台", "智慧財產權"]),
        "保密": any(k in text for k in ["保密", "機密", "揭露", "GitHub", "開源"]),
    }
    active = [k for k, v in flags.items() if v]
    if len(active) >= 2:
        return "混合型"
    return active[0] if active else "其他"


def detect_topics(text: str) -> List[str]:
    text = normalize_text(text)
    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in text for k in keywords):
            found.append(topic)
    return found


def score_topic_overlap(a: List[str], b: List[str]) -> int:
    return len(set(a or []) & set(b or []))


def lexical_score(text: str, query: str) -> int:
    text = normalize_text(text)
    query = normalize_text(query)
    score = 0
    for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", query):
        if token in text:
            score += 1
    return score


def parse_core_topics_field(val: Any) -> List[str]:
    if isinstance(val, list):
        return [normalize_topic_name(str(x)) for x in val if str(x).strip()]
    if isinstance(val, str):
        parts = re.split(r"[、,，;；\s]+", val)
        return [normalize_topic_name(x) for x in parts if x.strip()]
    return []


# 草稿切分
def split_draft_into_articles(text: str) -> List[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return []

    pattern = r"(第[一二三四五六七八九十百0-9]+條[：:]\s*.*?)(?=(?:\n?第[一二三四五六七八九十百0-9]+條[：:])|$)"
    matches = re.findall(pattern, text, flags=re.S)

    articles = []
    if matches:
        for raw in matches:
            raw = normalize_text(raw)
            m = re.match(r"(第[一二三四五六七八九十百0-9]+條)[：:]\s*([^\n ]+)?\s*(.*)", raw, flags=re.S)
            if m:
                content = normalize_text(raw)
                articles.append({
                    "article_no": m.group(1).strip(),
                    "title": (m.group(2) or "").strip(),
                    "content": content,
                    "topics": detect_topics(content),
                })
            else:
                articles.append({
                    "article_no": "",
                    "title": "",
                    "content": raw,
                    "topics": detect_topics(raw),
                })
        return articles

    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [{
        "article_no": "",
        "title": "",
        "content": p,
        "topics": detect_topics(p),
    } for p in paras]


def build_article_map(articles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    article_map = {}
    for idx, a in enumerate(articles, start=1):
        no = (a.get("article_no", "") or "").strip()
        title = (a.get("title", "") or "").strip()
        if no:
            article_map[no] = a
        article_map[f"ARTICLE_{idx}"] = a
        if title:
            article_map[title] = a
    return article_map


def article_to_key(article: Dict[str, Any], idx: int) -> str:
    no = (article.get("article_no", "") or "").strip()
    return no if no else f"ARTICLE_{idx}"


# 檔案處理
def save_upload_file(up_file) -> Dict[str, Any]:
    ensure_upload_dir()
    data = up_file.getvalue()
    if not data:
        raise ValueError("上傳檔案大小為 0 bytes，請重新上傳。")

    ext = os.path.splitext(up_file.name)[-1].lower().lstrip(".")
    file_id = str(uuid.uuid4())
    storage_path = os.path.join(UPLOAD_DIR, f"{file_id}_{up_file.name}")

    with open(storage_path, "wb") as f:
        f.write(data)

    return {
        "doc_id": file_id,
        "file_name": up_file.name,
        "file_type": ext,
        "storage_path": storage_path,
        "sha256": sha256_bytes(data),
        "byte_size": len(data),
        "created_at": datetime.datetime.now(),
    }


def extract_text_from_pdf(file_or_path, max_pages: int = 30) -> str:
    try:
        reader = PdfReader(file_or_path)
        texts = [page.extract_text() or "" for page in reader.pages[:max_pages]]
        return normalize_text("\n".join(texts))
    except Exception as e:
        logging.error(f"PDF 讀取失敗: {e}")
        return ""


def extract_text_from_docx(file_or_path) -> str:
    try:
        doc = DocxReader(file_or_path)
        paras = []
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                paras.append(t)
        return normalize_text("\n".join(paras))
    except Exception as e:
        logging.error(f"DOCX 讀取失敗: {e}")
        return ""


# 資料庫連線
@st.cache_resource(show_spinner=False)
def get_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return client, db


@st.cache_resource(show_spinner=False)
def get_chroma():
    ensure_dirs()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )

    template_collection = chroma_client.get_or_create_collection(
        name="contract_templates_fulltext",
        embedding_function=ollama_ef
    )

    chunk_collection = chroma_client.get_or_create_collection(
        name="contract_template_chunks",
        embedding_function=ollama_ef
    )

    return chroma_client, template_collection, chunk_collection


mongo_client, db = get_mongo()
templates = db["templates"]
history_db = db["ContractHistory"]

chroma_client, template_collection, chunk_collection = get_chroma()


# Ollama
def ollama_json(prompt: str, model: str = MODEL, temperature: float = 0.0, top_p: float = 0.1) -> Dict[str, Any]:
    try:
        res = ollama.generate(
            model=model,
            prompt=prompt.strip(),
            format="json",
            options={"temperature": temperature, "top_p": top_p},
        )
        return safe_json_load(res.get("response", "{}"))
    except Exception as e:
        logging.error(f"Ollama JSON 失敗: {e}")
        return {}


# 入庫分析
def llm_ingest_contract(text: str) -> Dict[str, Any]:
    prompt = f"""
你是法務文件入庫助理。請只輸出合法 JSON，不要輸出其他文字。

請根據以下合約內容抽取：
- contract_type: 合約類型
- summary: 80~160 字摘要（繁中）
- keywords: 8~15 個關鍵字（繁中 list）
- template_role: 模板用途，請填「標準模板」或「其他」
- core_topics: 這份模板最核心的審查主題，從下列主題中選 3~8 個
  維護時間、維護人力、異常回覆時限、修復時限、違約金、保密與開源、管轄法院、智慧財產權、保密期間、損害賠償、付款價金、交付驗收

合約內容：
{text[:7000]}

JSON 格式：
{{
  "contract_type": "維護合約/保密協定/開發合約/其他",
  "summary": "摘要",
  "keywords": ["關鍵字1", "關鍵字2"],
  "template_role": "標準模板",
  "core_topics": ["違約金", "管轄法院"]
}}
"""
    obj = ollama_json(prompt)
    if "keywords" not in obj or not isinstance(obj["keywords"], list):
        obj["keywords"] = []
    if "core_topics" not in obj or not isinstance(obj["core_topics"], list):
        obj["core_topics"] = detect_topics(text)

    obj["contract_type"] = str(obj.get("contract_type", "其他") or "其他").strip()
    obj["summary"] = str(obj.get("summary", "") or "").strip()
    obj["template_role"] = str(obj.get("template_role", "標準模板") or "標準模板").strip()
    obj["core_topics"] = [normalize_topic_name(x) for x in obj.get("core_topics", []) if str(x).strip()]
    return obj


def llm_parse_user_request(message: str) -> Dict[str, Any]:
    prompt = f"""
你是合約助理。請把需求整理成 JSON，不要輸出其他文字。

JSON 結構：
{{
  "intent": "generate | review",
  "contract_type": "維護合約/其他",
  "fields": {{
    "party_a": "",
    "party_b": "",
    "amount": "",
    "term": "",
    "system_name": ""
  }},
  "notes": ""
}}

使用者訊息：
{message}
"""
    obj = ollama_json(prompt)
    if "fields" not in obj or not isinstance(obj["fields"], dict):
        obj["fields"] = {}
    obj["intent"] = str(obj.get("intent", "generate") or "generate").strip()
    obj["contract_type"] = str(obj.get("contract_type", "其他") or "其他").strip()
    obj["notes"] = str(obj.get("notes", "") or "").strip()
    return obj


# 模板搜尋與選擇
def query_templates_fulltext(draft_text: str, n_results: int = 12) -> List[Dict[str, Any]]:
    try:
        results = template_collection.query(
            query_texts=[draft_text[:3500]],
            n_results=n_results
        )
    except Exception as e:
        logging.error(f"模板全文檢索失敗: {e}")
        return []

    refs = []
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])

    if not docs or not docs[0]:
        return refs

    seen_doc_ids = set()
    for i, doc_text in enumerate(docs[0]):
        meta = metas[0][i] if metas and metas[0] else {}
        doc_id = meta.get("doc_id")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        mongo_doc = templates.find_one({"doc_id": doc_id}) or {}

        refs.append({
            "doc_id": doc_id,
            "file_name": meta.get("file_name", "未知模板"),
            "contract_type": meta.get("contract_type", "其他"),
            "summary": meta.get("summary", ""),
            "keywords": mongo_doc.get("keywords", []) if isinstance(mongo_doc.get("keywords", []), list) else [],
            "core_topics": mongo_doc.get("core_topics", []) if isinstance(mongo_doc.get("core_topics", []), list) else [],
            "source_text": mongo_doc.get("source_text", "") or doc_text,
        })

    return refs


def select_review_templates(draft_text: str, articles: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    candidates = query_templates_fulltext(draft_text, n_results=max(10, top_k * 3))
    draft_topics = detect_topics(draft_text)
    article_topic_union = set()
    for a in articles:
        article_topic_union.update(a.get("topics", []))

    ranked = []
    for c in candidates:
        real_template_topics = detect_topics(c.get("source_text", ""))

        text_block = " ".join([
            c.get("file_name", ""),
            c.get("contract_type", ""),
            c.get("summary", ""),
            " ".join(c.get("keywords", [])),
            " ".join(real_template_topics),
        ])

        score = 0
        score += lexical_score(text_block, draft_text[:1200])
        score += score_topic_overlap(draft_topics, real_template_topics) * 4
        score += score_topic_overlap(list(article_topic_union), real_template_topics) * 3

        if detect_contract_mode_from_text(draft_text) == "混合型" and c.get("contract_type") in ["維護合約", "保密協定", "開發合約"]:
            score += 2

        c["core_topics"] = real_template_topics
        ranked.append((score, c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked[:top_k]]


def query_template_chunks_by_query(query_text: str, candidate_doc_ids: List[str], n_results: int = 12) -> List[Dict[str, Any]]:
    if not candidate_doc_ids:
        return []

    try:
        results = chunk_collection.query(
            query_texts=[query_text[:2000]],
            n_results=max(n_results, 12)
        )
    except Exception as e:
        logging.error(f"模板片段檢索失敗: {e}")
        return []

    refs = []
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])

    if not docs or not docs[0]:
        return refs

    seen = set()
    for i, doc_text in enumerate(docs[0]):
        meta = metas[0][i] if metas and metas[0] else {}
        doc_id = meta.get("doc_id")
        if doc_id not in candidate_doc_ids:
            continue

        key = (doc_id, doc_text[:120])
        if key in seen:
            continue
        seen.add(key)

        refs.append({
            "doc_id": doc_id,
            "file_name": meta.get("file_name", "未知模板"),
            "content": normalize_text(doc_text),
            "contract_type": meta.get("contract_type", "其他"),
            "topics": detect_topics(doc_text),
        })

        if len(refs) >= n_results:
            break

    return refs


def build_target_queries(draft_text: str, articles: List[Dict[str, Any]], selected_templates: List[Dict[str, Any]]) -> List[str]:
    queries = [draft_text[:1500]]

    for article in articles:
        content = article.get("content", "")
        queries.append(content[:700])

        for topic in article.get("topics", []):
            kw = " ".join(TOPIC_KEYWORDS.get(topic, [])[:5])
            queries.append(f"{topic} {kw} {content[:240]}")

    template_topics = set()
    for t in selected_templates:
        template_topics.update(t.get("core_topics", []))

    for topic in template_topics:
        kw = " ".join(TOPIC_KEYWORDS.get(topic, [])[:5])
        queries.append(f"{topic} {kw}")

    dedup = []
    seen = set()
    for q in queries:
        q = normalize_text(q)
        if q and q not in seen:
            seen.add(q)
            dedup.append(q)
    return dedup


def search_relevant_templates(draft_text: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    articles = split_draft_into_articles(draft_text)
    selected_templates = select_review_templates(draft_text, articles, top_k=top_k)
    doc_ids = [x["doc_id"] for x in selected_templates if x.get("doc_id")]

    all_chunks = []
    seen = set()
    for q in build_target_queries(draft_text, articles, selected_templates):
        refs = query_template_chunks_by_query(q, doc_ids, n_results=8)
        for r in refs:
            key = (r["doc_id"], r["content"][:150])
            if key not in seen:
                seen.add(key)
                all_chunks.append(r)

    return selected_templates, all_chunks[:36], articles


# 單條審查
def search_template_chunks_for_article(article: Dict[str, Any], selected_templates: List[Dict[str, Any]], n_results: int = 8) -> List[Dict[str, Any]]:
    doc_ids = [x["doc_id"] for x in selected_templates if x.get("doc_id")]
    article_text = article.get("content", "")
    article_topics = article.get("topics", [])

    queries = [article_text[:900]]
    for topic in article_topics:
        kw = " ".join(TOPIC_KEYWORDS.get(topic, [])[:5])
        queries.append(f"{topic} {kw} {article_text[:240]}")

    seen = set()
    out = []
    for q in queries:
        refs = query_template_chunks_by_query(q, doc_ids, n_results=n_results)
        for r in refs:
            key = (r["doc_id"], r["content"][:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)

    scored = []
    for r in out:
        score = score_topic_overlap(article_topics, r.get("topics", []))
        score += lexical_score(r.get("content", ""), article_text[:200])
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:n_results]]


def llm_review_single_article(article: Dict[str, Any], article_key: str, candidate_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_block = []
    source_names = []

    for i, c in enumerate(candidate_chunks, start=1):
        fname = c["file_name"]
        source_names.append(fname)
        chunk_block.append(
            f"【依據 {i}】檔名：{fname}\n內容：{short_text(c['content'], 260)}"
        )

    allowed_sources = "、".join(sorted(set(source_names))) if source_names else "無"

    prompt = f"""
你是企業法務審查 AI，立場固定站在甲方。
請只審查以下單一草稿條文，並根據提供的模板依據片段判斷是否有問題。

重要限制：
1. 只能輸出 JSON。
2. source 欄位必須直接填入下列其中一個真實檔名，不可填「模板檔名」或其他泛稱：
{allowed_sources}
3. 如果同一條草稿含有多個風險點，必須拆成多筆 issue。
4. suggestion 必須盡量貼近模板標準，不可自行發明模板沒有的條件。
5. 可使用的 issue_topic 優先從下列中挑選：
   維護時間、維護人力、異常回覆時限、修復時限、違約金、保密與開源、管轄法院、智慧財產權、保密期間、損害賠償、付款價金、交付驗收

輸出格式：
{{
  "major_issues": [
    {{
      "article_key": "{article_key}",
      "clause": "條款名稱",
      "issue_topic": "維護時間",
      "type": "deviation 或 conflict",
      "risk": "Critical/High/Medium/Low",
      "analysis": "分析",
      "suggestion": "建議修正方式",
      "source": "真實模板檔名"
    }}
  ],
  "general_issues": []
}}

【草稿條文】
{article.get("content", "")}

【條文偵測主題】
{"、".join(article.get("topics", []))}

【模板依據片段】
{chr(10).join(chunk_block)}
"""
    return ollama_json(prompt)


def infer_missing_topics_from_templates(selected_templates: List[Dict[str, Any]], articles: List[Dict[str, Any]]) -> List[str]:
    template_topics = set()
    for t in selected_templates:
        template_topics.update(t.get("core_topics", []))

    draft_topics = set()
    for a in articles:
        draft_topics.update(a.get("topics", []))

    return [t for t in template_topics if t not in draft_topics]


def review_articles_individually(
    draft_text: str,
    selected_templates: List[Dict[str, Any]],
    articles: List[Dict[str, Any]]
) -> Dict[str, Any]:
    all_major = []
    all_general = []

    for idx, article in enumerate(articles, start=1):
        article_key = article_to_key(article, idx)
        candidate_chunks = search_template_chunks_for_article(article, selected_templates, n_results=8)
        article_result = llm_review_single_article(article, article_key, candidate_chunks)

        all_major.extend(article_result.get("major_issues", []))
        all_general.extend(article_result.get("general_issues", []))

    missing_topics = infer_missing_topics_from_templates(selected_templates, articles)
    missing_clauses = []
    for topic in missing_topics:
        if topic in ["保密期間", "損害賠償", "智慧財產權", "付款價金", "交付驗收", "保密與開源"]:
            source = ""
            for t in selected_templates:
                if topic in t.get("core_topics", []):
                    source = t.get("file_name", "")
                    break
            missing_clauses.append({
                "clause": topic,
                "issue_topic": topic,
                "why_missing": "此主題存在於所選模板集合中，但草稿未明確規範。",
                "suggestion": "建議依對應模板補入完整條款。",
                "source": source,
            })

    return {
        "contract_type_guess": detect_contract_mode_from_text(draft_text),
        "summary": "系統已依自動選擇之模板集合，逐條審查草稿並整理主要偏離、衝突與缺漏。",
        "major_issues": all_major,
        "general_issues": all_general,
        "missing_clauses": missing_clauses,
    }


# 審查結果標準化
def extract_relevant_snippet(text: str, topic: str, max_len: int = 260) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    keywords = TOPIC_KEYWORDS.get(normalize_topic_name(topic), [])
    parts = re.split(r"[。；\n]", text)

    scored = []
    for part in parts:
        part = normalize_text(part)
        if not part:
            continue
        score = sum(1 for kw in keywords if kw and kw in part)
        if score > 0:
            scored.append((score, part))

    if scored:
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        return scored[0][1][:max_len]

    return text[:max_len]


def build_template_basis_index(top_chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    index: Dict[str, Dict[str, List[str]]] = {}
    for chunk in top_chunks:
        source = chunk.get("file_name", "未知模板")
        content = normalize_text(chunk.get("content", ""))
        if source not in index:
            index[source] = {}

        topics = chunk.get("topics") or detect_topics(content)
        for topic in set(topics):
            canonical = normalize_topic_name(topic)
            index[source].setdefault(canonical, [])
            if content not in index[source][canonical]:
                index[source][canonical].append(content)
    return index


def select_template_basis(
    topic: str,
    source: str,
    template_basis_index: Dict[str, Dict[str, List[str]]]
) -> str:
    topic = normalize_topic_name(topic)
    source = (source or "").strip()

    def pick_best(chunks: List[str], topic_name: str) -> str:
        if not chunks:
            return ""
        scored = []
        for ch in chunks:
            snippet = extract_relevant_snippet(ch, topic_name, max_len=260)
            score = lexical_score(snippet, " ".join(TOPIC_KEYWORDS.get(topic_name, [])))
            scored.append((score, snippet))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else ""

    if source in template_basis_index:
        topic_map = template_basis_index[source]
        if topic in topic_map and topic_map[topic]:
            return pick_best(topic_map[topic], topic)

        for canonical, aliases in TOPIC_ALIAS.items():
            if topic == canonical:
                for alias in [canonical] + aliases:
                    alias_norm = normalize_topic_name(alias)
                    if alias_norm in topic_map and topic_map[alias_norm]:
                        return pick_best(topic_map[alias_norm], topic)

    for _, topic_map in template_basis_index.items():
        if topic in topic_map and topic_map[topic]:
            return pick_best(topic_map[topic], topic)

    return ""


def compute_review_score(review_json: Dict[str, Any]) -> int:
    score = 100
    for item in review_json.get("major_issues", []):
        risk = str(item.get("risk", "")).lower()
        score -= {"critical": 22, "high": 16, "medium": 10}.get(risk, 5)

    for item in review_json.get("general_issues", []):
        risk = str(item.get("risk", "")).lower()
        score -= {"critical": 14, "high": 10, "medium": 6}.get(risk, 3)

    for _ in review_json.get("missing_clauses", []):
        score -= 8

    return max(0, min(100, score))


def validate_issue_binding(issue: Dict[str, Any], original_draft_text: str) -> bool:
    draft_text = issue.get("draft_text", "")
    if not draft_text:
        return False
    probe = normalize_text(draft_text)[:40]
    return probe in normalize_text(original_draft_text) if probe else False


def normalize_review_json(
    raw: Dict[str, Any],
    used_templates: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    top_chunks: List[Dict[str, Any]],
    original_draft_text: str
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    article_map = build_article_map(articles)
    template_basis_index = build_template_basis_index(top_chunks)

    def resolve_article(article_key: str) -> Dict[str, Any]:
        article_key = (article_key or "").strip()
        if article_key in article_map:
            return article_map[article_key]
        for k, v in article_map.items():
            if article_key and (article_key in k or k in article_key):
                return v
        return {"article_no": "", "title": "", "content": "", "topics": []}

    def norm_issue(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        article_key = str(item.get("article_key", "") or "").strip()
        clause = str(item.get("clause", "未命名項目")).strip()
        topic = normalize_topic_name(str(item.get("issue_topic", clause) or clause))
        source = normalize_text(str(item.get("source", "") or ""))

        article = resolve_article(article_key)
        draft_text = normalize_text(article.get("content", "") or "")
        template_basis = select_template_basis(topic, source, template_basis_index)

        issue = {
            "clause": clause,
            "risk": str(item.get("risk", "Medium")).strip(),
            "draft_text": draft_text,
            "template_basis": normalize_text(template_basis),
            "analysis": normalize_text(str(item.get("analysis", "") or "")),
            "suggestion": normalize_text(str(item.get("suggestion", "") or "")),
            "source": source,
            "type": str(item.get("type", "") or "").strip(),
            "issue_topic": topic,
            "article_key": article_key,
        }

        dt = draft_text
        if "GitHub" in dt or "開源社群" in dt:
            issue["risk"] = "Critical"
            issue["type"] = "conflict"

        if "加州地方法院" in dt or "美國加州" in dt:
            if issue["risk"] not in ["Critical"]:
                issue["risk"] = "High"

        if "萬分之一" in dt or "500 元" in dt or "500元" in dt:
            if issue["risk"] not in ["Critical"]:
                issue["risk"] = "High"

        if "3 個工作天" in dt or "3個工作天" in dt:
            if issue["risk"] not in ["Critical", "High"]:
                issue["risk"] = "High"

        if "實習生" in dt:
            if issue["risk"] not in ["Critical"]:
                issue["risk"] = "High"

        if not validate_issue_binding(issue, original_draft_text):
            return None
        return issue

    def norm_missing(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            topic = normalize_topic_name(item)
            return {
                "clause": topic,
                "why_missing": "",
                "suggestion": "",
                "source": "",
                "issue_topic": topic,
            }

        if not isinstance(item, dict):
            return None

        topic = normalize_topic_name(str(item.get("issue_topic", item.get("clause", "")) or ""))
        source = normalize_text(str(item.get("source", "") or ""))

        return {
            "clause": str(item.get("clause", topic or "未命名條款")).strip(),
            "why_missing": normalize_text(str(item.get("why_missing", "") or "")),
            "suggestion": normalize_text(str(item.get("suggestion", "") or "")),
            "source": source,
            "issue_topic": topic,
        }

    major_issues = [x for x in (norm_issue(i) for i in raw.get("major_issues", [])) if x]
    general_issues = [x for x in (norm_issue(i) for i in raw.get("general_issues", [])) if x]
    missing_clauses = [x for x in (norm_missing(i) for i in raw.get("missing_clauses", [])) if x]

    def dedup_issues(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for it in items:
            key = (it.get("article_key", ""), it.get("issue_topic", ""), it.get("source", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    def dedup_missing(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for it in items:
            key = (it.get("issue_topic", ""), it.get("source", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    major_issues = dedup_issues(major_issues)
    general_issues = dedup_issues(general_issues)
    missing_clauses = dedup_missing(missing_clauses)

    contract_type_guess = str(raw.get("contract_type_guess", "未判定") or "未判定").strip()
    mode_fallback = detect_contract_mode_from_text(original_draft_text)
    if contract_type_guess in ["未判定", "其他", "維護", "開發", "保密"] and mode_fallback != "其他":
        contract_type_guess = mode_fallback

    summary = normalize_text(str(raw.get("summary", "") or ""))
    if not summary:
        summary = "系統已根據自動選出的模板集合完成草稿審查，並整理出主要偏離、衝突及缺漏條款。"

    normalized = {
        "contract_type_guess": contract_type_guess,
        "summary": summary,
        "used_templates": [
            {
                "file_name": t.get("file_name", "未知模板"),
                "contract_type": t.get("contract_type", "其他"),
                "summary": t.get("summary", ""),
                "core_topics": t.get("core_topics", []),
            }
            for t in used_templates
        ],
        "major_issues": major_issues,
        "general_issues": general_issues,
        "missing_clauses": missing_clauses,
    }
    normalized["score"] = compute_review_score(normalized)
    return normalized


# 報價風險
def assess_price_risk(user_input: str) -> str:
    prompt = f"""
你是一位專業的金融合約分析官。請分析以下內容，並嚴格輸出 JSON 格式。
JSON 格式：
{{"vendor_name": "廠商名稱", "amount": 1000000}}

注意：
- amount 僅保留純數字
- 不要有逗號
- 不要輸出額外文字

輸入內容：
{user_input[:2000]}
"""
    try:
        data = ollama_json(prompt)
    except Exception:
        return "⚠️ 無法解析合約金額與廠商資訊。"

    vendor_clean = (data.get("vendor_name", "") or data.get("vendor", "")).replace("万", "萬")

    try:
        current_amount = int(re.sub(r"[^\d]", "", str(data.get("amount", 0))))
    except ValueError:
        current_amount = 0

    if not vendor_clean or current_amount == 0:
        return f"⚠️ 萃取資訊不足。識別結果：廠商 `{vendor_clean}`，金額 `{current_amount}`。"

    search_keyword = vendor_clean[-2:] if len(vendor_clean) >= 2 else vendor_clean
    past_records = list(history_db.find({"vendor_name": {"$regex": search_keyword}}))

    if past_records:
        avg_amount = sum(r["amount"] for r in past_records) / len(past_records)
        report = (
            "### 💰 歷史報價風險評估\n\n"
            f"- **識別廠商**：{vendor_clean}\n"
            f"- **本次報價**：新台幣 {current_amount:,.0f} 元\n"
            f"- **歷史均價**：新台幣 {avg_amount:,.0f} 元\n\n"
        )
        if current_amount > avg_amount * 1.5:
            report += "🚨 **【高風險警示】** 報價超過歷史均價 1.5 倍，建議啟動議價程序。"
        else:
            report += "✅ **【報價合理】** 報價落於該廠商之歷史合理區間內。"
        return report

    return f"⚠️ 查無與 `{vendor_clean}` 相關的歷史報價紀錄。"


# DocxTemplate 
def normalize_term(term: str) -> str:
    term = (term or "").strip()
    if not term:
        return term

    if re.search(r"(一年|1\s*年)", term) and ("至" not in term):
        start = datetime.date.today()
        try:
            end = datetime.date(start.year + 1, start.month, start.day) - datetime.timedelta(days=1)
        except ValueError:
            end = datetime.date(start.year + 1, start.month, start.day - 1) - datetime.timedelta(days=1)
        return f"{start.strftime('%Y年%m月%d日')}至{end.strftime('%Y年%m月%d日')}"

    return term


def search_templates(contract_type: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    ct = (contract_type or "").strip()

    candidates = list(
        templates.find(
            {"contract_type": {"$regex": ct, "$options": "i"}, "file_type": "docx"}
        ).sort("created_at", -1).limit(50)
    )

    if not candidates:
        candidates = list(templates.find({"file_type": "docx"}).sort("created_at", -1).limit(50))

    def score(doc):
        score_val = 0
        text = " ".join(doc.get("keywords", [])) + " " + (doc.get("summary", "") or "")
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", q):
            if token in text:
                score_val += 2
        return score_val

    return sorted(candidates, key=score, reverse=True)[:limit]


def parse_template_selector(text: str) -> Dict[str, str]:
    m = re.search(r'template\s*=\s*"([^"]+)"', text)
    if m:
        return {"file_name": m.group(1).strip()}
    return {}


def get_template_by_selector(selector: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if not selector:
        return None
    if selector.get("file_name"):
        return templates.find_one({
            "file_name": {"$regex": re.escape(selector["file_name"]), "$options": "i"}
        })
    return None


def generate_contract_from_template(template_path: str, output_path: str, fields: Dict[str, Any]) -> bool:
    try:
        doc = DocxTemplate(template_path)
        context = fields.copy()
        context["today"] = datetime.date.today().strftime("%Y年%m月%d日")

        if "amount" in context and context["amount"]:
            try:
                clean_amount = re.sub(r"[^\d]", "", str(context["amount"]))
                context["amount_formatted"] = f"{int(clean_amount):,}" if clean_amount else ""
            except ValueError:
                context["amount_formatted"] = str(context["amount"])

        if "term" in context and context["term"]:
            context["term"] = normalize_term(context["term"])

        doc.render(context)
        doc.save(output_path)
        return True
    except Exception as e:
        logging.error(f"合約生成失敗: {e}")
        return False

# 入庫
def upsert_template_vectors(meta: Dict[str, Any], full_text: str):
    full_text = normalize_text(full_text)
    chunks = chunk_text(full_text)

    try:
        template_collection.upsert(
            ids=[meta["doc_id"]],
            documents=[full_text[:12000]],
            metadatas=[{
                "doc_id": meta["doc_id"],
                "file_name": meta["file_name"],
                "contract_type": meta.get("contract_type", "其他"),
                "summary": meta.get("summary", ""),
                "keywords": ",".join(meta.get("keywords", [])),
                "core_topics": ",".join(meta.get("core_topics", [])),
            }]
        )
    except Exception as e:
        logging.error(f"模板全文向量入庫失敗: {e}")

    if chunks:
        ids = [f"{meta['doc_id']}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "doc_id": meta["doc_id"],
            "file_name": meta["file_name"],
            "contract_type": meta.get("contract_type", "其他"),
        } for _ in chunks]

        try:
            chunk_collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
        except Exception as e:
            logging.error(f"模板片段向量入庫失敗: {e}")


def handle_upload(files):
    inserted = 0
    skipped = 0

    for f in files:
        meta = save_upload_file(f)
        exist = templates.find_one({"sha256": meta["sha256"]})
        if exist:
            skipped += 1
            continue

        if meta["file_type"] == "pdf":
            text = extract_text_from_pdf(meta["storage_path"])
        else:
            text = extract_text_from_docx(meta["storage_path"])

        if not text.strip():
            logging.warning(f"檔案無法解析文字：{meta['file_name']}")
            skipped += 1
            continue

        ing = llm_ingest_contract(text)

        doc = {
            **meta,
            "contract_type": ing.get("contract_type", "其他"),
            "summary": ing.get("summary", ""),
            "keywords": ing.get("keywords", []),
            "template_role": ing.get("template_role", "標準模板"),
            "core_topics": ing.get("core_topics", []),
            "source_text": text[:20000],
        }

        templates.insert_one(doc)
        upsert_template_vectors(doc, text)
        inserted += 1

    return inserted, skipped


# UI helpers
def risk_label(risk: str) -> str:
    r = (risk or "").lower()
    if r == "critical":
        return "🔴 極高風險"
    if r == "high":
        return "🟠 高風險"
    if r == "medium":
        return "🟡 中度風險"
    return "🟢 低風險"


def render_issue_block(item: Dict[str, Any]):
    issue_type = item.get("type", "")
    issue_type_label = ""
    if issue_type == "conflict":
        issue_type_label = "（衝突）"
    elif issue_type == "deviation":
        issue_type_label = "（偏離）"

    st.markdown(f"#### {risk_label(item.get('risk'))}｜{item.get('clause', '未命名條款')} {issue_type_label}")
    with st.container(border=True):
        if item.get("draft_text"):
            st.markdown("**草稿內容**")
            st.write(item["draft_text"])

        if item.get("template_basis"):
            st.markdown("**模板標準依據**")
            st.write(item["template_basis"])

        if item.get("analysis"):
            st.markdown("**風險分析**")
            st.write(item["analysis"])

        if item.get("suggestion"):
            st.markdown("**建議修正方式**")
            st.write(item["suggestion"])

        if item.get("source"):
            st.caption(f"參考來源：{item['source']}")


def render_missing_block(item: Dict[str, Any]):
    with st.container(border=True):
        st.markdown(f"**{item.get('clause', '未命名條款')}**")
        if item.get("why_missing"):
            st.markdown("**缺漏原因**")
            st.write(item["why_missing"])
        if item.get("suggestion"):
            st.markdown("**建議補入方向**")
            st.write(item["suggestion"])
        if item.get("source"):
            st.caption(f"參考來源：{item['source']}")


def render_review_dashboard(data: Dict[str, Any]):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("🛡️ 合約安全指數", f"{data.get('score', 100)} 分")
    with col2:
        st.info(
            f"**合約類型判定**：{data.get('contract_type_guess', '未判定')}\n\n"
            f"**AI 總結摘要**：{data.get('summary', '無摘要')}"
        )

    used_templates = data.get("used_templates", [])
    if used_templates:
        st.markdown("### 📚 本次審查主要參考模板")
        for t in used_templates:
            with st.container(border=True):
                st.markdown(f"**{t.get('file_name', '未知模板')}**")
                st.caption(f"類型：{t.get('contract_type', '其他')}")
                if t.get("core_topics"):
                    st.caption("核心主題：" + "、".join(t.get("core_topics", [])))
                if t.get("summary"):
                    st.write(t["summary"])

    major = data.get("major_issues", [])
    general = data.get("general_issues", [])
    missing = data.get("missing_clauses", [])

    if major:
        st.markdown("### 🚨 重大偏離 / 衝突")
        for item in major:
            render_issue_block(item)

    if general:
        st.markdown("### ⚠️ 一般偏離")
        for item in general:
            render_issue_block(item)

    if missing:
        st.markdown("### 🧩 缺漏條款")
        for item in missing:
            render_missing_block(item)


# UI
st.set_page_config(
    page_title="企業智能法務中樞",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Helvetica Neue', Arial, '微軟正黑體', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.02);
    }
    div[data-testid="stFileUploader"], .stExpander {
        background-color: #FFFFFF;
        border: 1.5px dashed #CBD5E1;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid #F1F5F9;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #0F172A !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 🧭 系統導覽")
    page = st.radio("功能模組：", ["🤖 智能法務助理", "🗃️ 企業知識庫管理"])
    st.markdown("---")

    if st.button("🧪 注入模擬歷史報價資料"):
        if history_db.count_documents({}) == 0:
            history_db.insert_many([
                {"vendor_name": "萬旭浤", "amount": 1000000},
                {"vendor_name": "萬旭浤", "amount": 1100000},
            ])
            st.success("✅ 歷史資料已注入。")
        else:
            st.info("ℹ️ 資料已存在。")


if page == "🤖 智能法務助理":
    st.markdown("## ⚖️ 智能合約審查與生成系統")
    st.info(
        "💡 指令：`/review`（系統自動選模板審查草稿）｜`/risk [文字]`（報價預警）｜`/generate [文字]`（套版生成）",
        icon="ℹ️"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "您好！我是您的 AI 法務助理。請上傳合約草稿後輸入 `/review`，我會先自動選擇最相關模板，再進行整份審查。"
        }]

    with st.sidebar:
        st.markdown("### 📄 快速審查通道")
        draft_file = st.file_uploader("📂 上傳合約草稿", type=["pdf", "docx"], key="draft_file")

        if st.button("🧹 清空對話視窗"):
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.last_generated_path = None
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], dict):
                render_review_dashboard(msg["content"])
            else:
                st.markdown(msg["content"])

    user_msg = st.chat_input("輸入 /review、/risk 或 /generate...")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        msg_text = user_msg.strip()

        if msg_text.startswith("/review"):
            draft_content = msg_text.replace("/review", "", 1).strip()

            if not draft_content and draft_file is not None:
                with st.spinner("正在解析草稿內容..."):
                    if draft_file.name.lower().endswith(".pdf"):
                        draft_content = extract_text_from_pdf(draft_file)
                    else:
                        draft_content = extract_text_from_docx(draft_file)

            if not draft_content:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ 系統未偵測到草稿內容。請貼上文字或上傳 PDF / DOCX。"
                })
            else:
                with st.spinner("🧠 正在自動選擇模板並逐條審查草稿..."):
                    top_templates, top_chunks, articles = search_relevant_templates(draft_content, top_k=5)

                    if not top_templates:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ 目前知識庫中找不到可用模板。請先到「企業知識庫管理」上傳模板。"
                        })
                    else:
                        raw_review = review_articles_individually(
                            draft_content,
                            top_templates,
                            articles
                        )
                        review_json = normalize_review_json(
                            raw_review,
                            top_templates,
                            articles,
                            top_chunks,
                            draft_content
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": review_json
                        })
            st.rerun()

        elif msg_text.startswith("/risk"):
            with st.spinner("比對歷史報價資料庫中..."):
                risk_report = assess_price_risk(msg_text.replace("/risk", "", 1).strip())
            st.session_state.messages.append({"role": "assistant", "content": risk_report})
            st.rerun()

        elif msg_text.startswith("/generate"):
            with st.spinner("啟動 DocxTemplate 套版生成中..."):
                parsed = llm_parse_user_request(user_msg)
                ct = parsed.get("contract_type", "其他")

                base = (
                    get_template_by_selector(parse_template_selector(user_msg))
                    or (search_templates(ct, user_msg, 1) or [None])[0]
                )

                if not base:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ 查無相關範本，請先到知識庫上傳模板。"
                    })
                else:
                    out_name = f"{ct}_自動生成_{datetime.date.today().strftime('%Y%m%d')}.docx"
                    out_path = make_output_path(out_name)

                    ok = generate_contract_from_template(
                        base["storage_path"],
                        out_path,
                        parsed.get("fields", {})
                    )

                    if ok:
                        st.session_state.last_generated_path = out_path
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🎉 合約生成完畢！已成功套用範本 `{base['file_name']}`。"
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ 生成合約失敗，請確認 DOCX 範本內含正確的 `{{變數}}` 標籤。"
                        })
            st.rerun()

        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "未能識別指令。請輸入 `/review`、`/risk` 或 `/generate`。"
            })
            st.rerun()

    if st.session_state.get("last_generated_path"):
        try:
            with open(st.session_state.last_generated_path, "rb") as f:
                st.download_button(
                    "📥 點擊下載生成的 DOCX 合約",
                    data=f.read(),
                    file_name=os.path.basename(st.session_state.last_generated_path)
                )
        except Exception:
            pass

elif page == "🗃️ 企業知識庫管理":
    st.markdown("## 🗃️ 企業知識庫維護中樞")

    with st.container():
        new_files = st.file_uploader(
            "📂 擴充知識庫（支援 PDF / DOCX）",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        if st.button("🚀 確認上傳並執行向量化解析"):
            if new_files:
                with st.spinner("🚀 處理中..."):
                    inserted, skipped = handle_upload(new_files)
                    st.success(f"✅ 成功解析並入庫 {inserted} 份模板。")
                    if skipped:
                        st.info(f"ℹ️ 略過 {skipped} 份重複或無法解析的檔案。")
            else:
                st.warning("請先選擇要上傳的檔案。")

    st.markdown("---")
    docs = list(templates.find({}).sort("created_at", -1))
    st.metric("目前知識庫總數量", f"{len(docs)} 份")

    for doc in docs:
        with st.expander(f"📄 {doc.get('file_name')} | 類型: {doc.get('contract_type', '其他')}"):
            st.write(f"**🤖 AI 摘要**：{doc.get('summary', '')}")
            if doc.get("keywords"):
                st.caption("關鍵字：" + "、".join(doc.get("keywords", [])))
            if doc.get("core_topics"):
                st.caption("核心主題：" + "、".join(doc.get("core_topics", [])))
            st.caption(f"入庫時間：{doc.get('created_at')}")

            if st.button("🗑️ 永久刪除此範本", key=f"del_{doc['doc_id']}", type="primary"):
                try:
                    template_collection.delete(ids=[doc["doc_id"]])
                except Exception as e:
                    logging.warning(f"刪除模板全文向量失敗: {e}")

                try:
                    chunk_ids = [f"{doc['doc_id']}_chunk_{i}" for i in range(500)]
                    chunk_collection.delete(ids=chunk_ids)
                except Exception as e:
                    logging.warning(f"刪除模板片段向量時發生警告: {e}")

                templates.delete_one({"doc_id": doc["doc_id"]})

                try:
                    if os.path.exists(doc["storage_path"]):
                        os.remove(doc["storage_path"])
                except Exception:
                    pass

                st.rerun()