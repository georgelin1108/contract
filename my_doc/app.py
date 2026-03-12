import streamlit as st
import os
import re
import json
import uuid
import hashlib
import datetime
from typing import Dict, Any, List, Optional

from pymongo import MongoClient
import chromadb
from chromadb.utils import embedding_functions
import ollama
from pypdf import PdfReader
from docx import Document as DocxReader

# =========================
# 基本設定
# =========================
UPLOAD_DIR = os.path.abspath("./uploaded_files")
MODEL = "llama3"

def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def make_output_path(filename: str) -> str:
    ensure_upload_dir()
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")

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

def extract_text_from_pdf(file_or_path, max_pages: int = 12) -> str:
    reader = PdfReader(file_or_path)
    texts = []
    for p in reader.pages[:max_pages]:
        texts.append(p.extract_text() or "")
    return "\n".join(texts).strip()

def extract_text_from_docx(file_or_path) -> str:
    doc = DocxReader(file_or_path)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a:b+1])
        return {}

# =========================
# MongoDB 連線
# =========================
client = MongoClient("mongodb://localhost:27017/")
db = client["GlobalLifeContractDB"]
templates = db["templates"]

# =========================
# Vector DB (ChromaDB) 設置
# =========================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text" 
)
collection = chroma_client.get_or_create_collection(
    name="contract_chunks", 
    embedding_function=ollama_ef
)

# =========================
# 文本切塊函式 (Chunking)
# =========================
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    text = re.sub(r'\n+', '\n', text)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# =========================
# 解析 template selector
# =========================
def parse_template_selector(text: str) -> Dict[str, str]:
    m = re.search(r'template\s*=\s*"([^"]+)"', text)
    if m: return {"file_name": m.group(1).strip()}
    return {}

def get_template_by_selector(selector: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if not selector: return None
    if selector.get("file_name"):
        return templates.find_one({"file_name": {"$regex": re.escape(selector["file_name"]), "$options": "i"}})
    return None

# =========================
# LLM：Ingest（入庫分類與抽取）
# =========================
def llm_ingest_contract(text: str) -> Dict[str, Any]:
    prompt = f"""
你是法務文件入庫助理。請只輸出合法 JSON（不要額外文字）。
目標：從合約內容抽取：
- contract_type: 合約類型
- summary: 80~160 字摘要（繁中）
- keywords: 8~15 個關鍵字（繁中）
- fields: party_a, party_b, rep_a, rep_b, amount, term, system_name

合約內容（節錄）：
{text[:6000]}
""".strip()
    res = ollama.generate(model=MODEL, prompt=prompt, format="json")
    obj = safe_json_load(res.get("response", "{}"))
    if "keywords" not in obj or not isinstance(obj["keywords"], list): obj["keywords"] = []
    if "fields" not in obj or not isinstance(obj["fields"], dict): obj["fields"] = {}
    for k in ["party_a","party_b","rep_a","rep_b","amount","term","system_name"]:
        obj["fields"][k] = str(obj["fields"].get(k,"") or "")
    obj["contract_type"] = str(obj.get("contract_type","其他") or "其他").strip()
    return obj

# =========================
# RAG 檢索
# =========================
def search_similar_clauses(query: str, contract_type: str = "", limit: int = 12) -> List[Dict[str, Any]]:
    # 🚨 【關鍵修改】：直接拿掉 where_clause 篩選，強制進行全庫檢索！
    # 讓系統純粹靠「語意相似度」去抓地雷條款，不被合約分類標籤綁死。
    results = collection.query(
        query_texts=[query],
        n_results=limit
    )
    
    refs = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            refs.append({
                "content": results['documents'][0][i],
                "file_name": results['metadatas'][0][i]['file_name'],
                "doc_id": results['metadatas'][0][i]['doc_id']
            })
    return refs
# =========================
# LLM Reviewer：基於 RAG 檢索結果進行嚴格審查 (英文骨架防禦版)
# =========================
def llm_review_contract(user_contract_text: str, refs: List[Dict[str, Any]]) -> str:
    ref_block = []
    for i, r in enumerate(refs):
        ref_block.append(f"【Historical Contract {i+1}】 (File: {r['file_name']})\n{r['content']}")
    refs_text = "\n\n".join(ref_block) if ref_block else "(No historical data found)"

    prompt = f"""
You are a rigorous Taiwanese legal contract reviewer. Your task is to perform a strict text-based comparison between the [Draft Contract] and the [Historical Contracts].

🚨 CRITICAL INSTRUCTIONS:
1. Output language MUST be strictly Traditional Chinese (zh-TW).
2. DO NOT hallucinate. Do not invent any countries, cities, or laws (e.g., Syria, Libya, Manhattan) that are not explicitly written in the provided text.
3. Compare the following core clauses: Payment Terms, Maintenance Hours, Penalties (違約金), Confidentiality (保密義務), and Jurisdiction (管轄法院).

[Historical Contracts]
{refs_text}

[Draft Contract]
{user_contract_text[:6000]}

Please format your response EXACTLY as follows in Traditional Chinese:

**🔴 合約問題與風險檢查**：
(List the precise differences. Example: 草稿規定「...」，但歷史合約規定「...」，對甲方不利。)

**🟢 建議新增條款**：
(List important missing clauses, such as confidentiality.)

**🟡 建議修改條款**：
(Provide specific text modifications based on the Historical Contracts.)

**📚 參考文件與來源**：
(Quote the EXACT sentences from the Historical Contracts and cite the filename.)
""".strip()

    res = ollama.generate(model=MODEL, prompt=prompt, options={"temperature": 0.0, "top_p": 0.1})
    return res.get("response","")

# =========================
# DOCX 生成替換邏輯 
# =========================
def normalize_term(term: str) -> str:
    term = (term or "").strip()
    if not term: return term
    if re.search(r"(一年|1\s*年)", term) and ("至" not in term):
        start = datetime.date.today()
        try: end = datetime.date(start.year + 1, start.month, start.day) - datetime.timedelta(days=1)
        except ValueError: end = datetime.date(start.year + 1, start.month, start.day - 1) - datetime.timedelta(days=1)
        return f"{start.strftime('%Y年%m月%d日')}至{end.strftime('%Y年%m月%d日')}"
    return term

def rewrite_paragraph_plain(paragraph, new_text: str):
    for r in paragraph.runs[::-1]:
        paragraph._element.remove(r._element)
    run = paragraph.add_run(new_text)
    run.font.name = None

def apply_field_updates_docx(doc, party_a=None, party_b=None, rep_a=None, rep_b=None, amount=None, term=None, system_name=None, remove_title=True) -> int:
    hit = 0
    def iter_all_paragraphs(d):
        for p in d.paragraphs: yield p
        for t in d.tables:
            for row in t.rows:
                for cell in row.cells:
                    for p in cell.paragraphs: yield p

    amt = str(amount).replace(",", "").replace("，", "").strip() if amount else None
    t_norm = normalize_term(term) if term else None
    sysname = (system_name or "").strip() if system_name else None

    for p in iter_all_paragraphs(doc):
        txt = p.text or ""
        old_txt = txt

        def sub(pattern, repl, flags=0):
            nonlocal txt
            txt = re.sub(pattern, repl, txt, flags=flags)

        if party_a:
            sub(r"(甲\s*方\s*[:：]\s*).+", rf"\1{party_a}")
            sub(r"^(.*甲\s*方\s+)([^\n\r]+)$", rf"\1{party_a}")
        if party_b:
            sub(r"(乙\s*方\s*[:：]\s*).+", rf"\1{party_b}")
            sub(r"^(.*乙\s*方\s+)([^\n\r]+)$", rf"\1{party_b}")
        if sysname:
            sub(r"(「)\s*[○ＯoＯ]+\s*(」)", rf"\1{sysname}\2")
        if amt:
            sub(r"(新[臺台]幣)\s*[0-9,，○ＯoＯ]+(\s*元整)", rf"\1{amt}\2")
        if t_norm:
            sub(r"(維護期間\s*[:：]\s*).+", rf"\1{t_norm}")

        if txt != old_txt:
            rewrite_paragraph_plain(p, txt)
            hit += 1

    return hit

def search_templates(contract_type: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    ct = (contract_type or "").strip()
    candidates = list(templates.find({"contract_type": {"$regex": ct, "$options": "i"}, "file_type": "docx"}).sort("created_at", -1).limit(50))
    if not candidates:
        candidates = list(templates.find({"file_type": "docx"}).sort("created_at", -1).limit(50))
    def score(doc):
        s = 0
        text = " ".join(doc.get("keywords", [])) + " " + (doc.get("summary","") or "")
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", q):
            if token in text: s += 2
        return s
    return sorted(candidates, key=score, reverse=True)[:limit]

def llm_parse_user_request(message: str) -> Dict[str, Any]:
    prompt = f"""
你是合約助理。請把使用者需求整理成 JSON:
{{ "intent": "generate | review", "contract_type": "維護合約/其他", "fields": {{}}, "notes": "" }}
使用者訊息：{message}
""".strip()
    res = ollama.generate(model=MODEL, prompt=prompt, format="json")
    return safe_json_load(res.get("response","{}"))

def handle_upload(files):
    inserted = 0
    for f in files:
        meta = save_upload_file(f)
        exist = templates.find_one({"sha256": meta["sha256"]})
        if exist: continue

        text = extract_text_from_pdf(meta["storage_path"]) if meta["file_type"] == "pdf" else extract_text_from_docx(meta["storage_path"])
        ing = llm_ingest_contract(text)
        
        doc = {**meta, "contract_type": ing.get("contract_type", "其他"), "summary": ing.get("summary", ""), "keywords": ing.get("keywords", []), "fields": ing.get("fields", {}), "source_text": text[:12000]}
        templates.insert_one(doc)

        chunks = chunk_text(text)
        if chunks:
            collection.add(
                documents=chunks,
                metadatas=[{"doc_id": meta["doc_id"], "file_name": meta["file_name"], "contract_type": ing.get("contract_type", "其他")}] * len(chunks),
                ids=[f"{meta['doc_id']}_chunk_{i}" for i in range(len(chunks))]
            )
        inserted += 1
    return inserted

# =========================
# UI：頁面與樣式設定 (極致美化版)
# =========================
st.set_page_config(page_title="智能合約系統", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

# 💎 核心 CSS 美化 💎
st.markdown("""
<style>
    /* 全域背景與字體設定 */
    .stApp { 
        background-color: #F8FAFC; 
        font-family: 'Helvetica Neue', Arial, '微軟正黑體', sans-serif;
    }
    
    /* 側邊欄美化 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.02);
    }
    
    /* 統一卡片式外觀 (上傳區塊、下拉選單容器) */
    div[data-testid="stFileUploader"], .stExpander {
        background-color: #FFFFFF;
        border: 1.5px dashed #CBD5E1;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #2563EB;
        background-color: #F0F9FF;
    }
    
    /* 聊天訊息氣泡美化 */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border: 1px solid #F1F5F9;
        margin-bottom: 1rem;
    }
    /* 使用者說話的氣泡加一點淡藍色背景區隔 */
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(odd) {
        background-color: #F8FAFC;
        border-left: 4px solid #94A3B8;
    }
    /* 系統助理說話的氣泡加上強調色 */
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(even) {
        border-left: 4px solid #2563EB;
    }

    /* 按鈕全局美化 */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transform: translateY(-1px);
        color: white;
    }
    
    /* 針對危險操作按鈕 (刪除) 改變顏色 */
    button[kind="primary"] {
        background-color: #EF4444 !important;
    }
    button[kind="primary"]:hover {
        background-color: #DC2626 !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
    }

    /* 文字輸入框與輸入區域 */
    .stChatInputContainer {
        border-radius: 24px !important;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1) !important;
        border: 1px solid #E2E8F0 !important;
    }
    
    /* 標題顏色加深 */
    h1, h2, h3 { color: #0F172A; font-weight: 700; }
    p, li { color: #334155; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# =========================
# 側邊欄：頁面導航
# =========================
with st.sidebar:
    st.markdown("## 🧭 系統導覽")
    page = st.radio("請選擇功能模組：", ["🤖 智能法務助理", "🗃️ 企業知識庫管理"])
    st.markdown("---")

# =========================
# 頁面 1：智能合約助理 (聊天與審查)
# =========================
if page == "🤖 智能法務助理":
    st.markdown("## ⚖️ 智能合約審查與生成系統")
    st.info('💡 **快捷指令**：\n- 在側邊欄上傳草稿後，輸入 `/review` 執行高強度審查。\n- 輸入 `/generate 幫我跟萬旭浤簽一份維護合約...` 體驗自動生成。', icon="ℹ️")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的專屬 AI 法務助理。請上傳合約草稿並輸入 `/review` 讓我為您把關風險，或是告訴我您的需求，讓我幫您生成一份合約。"}]

    with st.sidebar:
        st.markdown("### 📄 快速審查通道")
        draft_file = st.file_uploader("📂 拖曳或點擊上傳合約草稿", type=["pdf","docx"], key="draft_file")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧹 清空對話視窗"):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input('輸入 /generate 或 /review...')

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        msg_text = user_msg.strip()

        # 🎯 審查合約
        if msg_text.startswith("/review"):
            draft_content = msg_text.replace("/review", "", 1).strip()
            
            if not draft_content and draft_file is not None:
                with st.spinner("正在解析草稿內容..."):
                    if draft_file.name.lower().endswith(".pdf"):
                        draft_content = extract_text_from_pdf(draft_file)
                    else:
                        draft_content = extract_text_from_docx(draft_file)

            if not draft_content:
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ 系統未偵測到草稿內容。請直接在對話框貼上文字，或在左側邊欄上傳檔案。"})
            else:
                with st.spinner("啟動企業知識庫 RAG 檢索與法務審查分析中..."):
                    parsed = llm_parse_user_request(draft_content[:1000])
                    ct = parsed.get("contract_type", "其他")
                    search_query = f"違約金 管轄法院 保密義務 付款方式 維護時間 服務水準 賠償責任 {draft_content[:600]}"
                    refs = search_similar_clauses(query=search_query, contract_type=ct, limit=12)
                    review_result = llm_review_contract(draft_content, refs)
                    
                st.session_state.messages.append({"role": "assistant", "content": f"### 📊 智能法務風險評估報告\n\n{review_result}"})
            st.rerun()

        # 🎯 生成合約
        elif msg_text.startswith("/generate"):
            with st.spinner("分析商業條件並生成標準文件中..."):
                parsed = llm_parse_user_request(user_msg)
                ct = parsed.get("contract_type", "其他")
                fields = parsed.get("fields", {})
                notes = parsed.get("notes", "")
                fields["term"] = normalize_term(fields.get("term",""))

                selector = parse_template_selector(user_msg)
                forced = get_template_by_selector(selector)
                base = forced if forced else (search_templates(ct, user_msg + " " + notes, limit=1) or [None])[0]

                if not base or base.get("file_type") != "docx":
                    st.session_state.messages.append({"role": "assistant", "content": "⚠️ 知識庫中尚未建立對應的 DOCX 標準範本。請先至「企業知識庫管理」頁面上傳。"})
                else:
                    out_name = f"{ct}_自動生成_{datetime.date.today().strftime('%Y%m%d')}.docx"
                    out_path = make_output_path(out_name)
                    doc = DocxReader(base["storage_path"])
                    hits = apply_field_updates_docx(doc, **fields, remove_title=True)
                    doc.save(out_path)
                    st.session_state.last_generated_path = out_path
                    cite = f"🎉 **標準合約生成完畢！** (成功匹配並更新 {hits} 處欄位)\n\n**套用之標準範本**：`{base.get('file_name','')}`"
                    st.session_state.messages.append({"role": "assistant", "content": cite})
            st.rerun()

        else:
            st.session_state.messages.append({"role": "assistant", "content": "未能識別指令。請輸入 `/review` 執行草稿審查，或 `/generate [商業條件]` 生成標準合約。"})
            st.rerun()

    if "last_generated_path" in st.session_state and st.session_state.last_generated_path and os.path.exists(st.session_state.last_generated_path):
        st.markdown("<br>", unsafe_allow_html=True)
        with open(st.session_state.last_generated_path, "rb") as f:
            st.download_button("📥 點擊下載生成的 DOCX 檔案", data=f.read(), file_name=os.path.basename(st.session_state.last_generated_path), use_container_width=True)

# =========================
# 頁面 2：模板後台管理 (新增、刪除、查詢)
# =========================
elif page == "🗃️ 企業知識庫管理":
    st.markdown("## 🗃️ 企業知識庫維護中樞")
    st.caption("在此集中管理系統 RAG 檢索所依賴的標準範本。維持資料庫的潔淨將有助於提升審查的精準度。")

    st.markdown("### 📥 擴充標準知識庫")
    with st.container():
        new_files = st.file_uploader("📂 支援 PDF 或 DOCX 格式，可多選上傳", type=["pdf", "docx"], accept_multiple_files=True)
        if st.button("🚀 確認上傳並執行向量化解析"):
            if new_files:
                with st.spinner("啟動文件解析與 ChromaDB 語意向量建置中..."):
                    n = handle_upload(new_files)
                    st.success(f"✅ 系統通知：已成功解析並入庫 {n} 份標準範本！")
            else:
                st.warning("請先將檔案拖曳至上方虛線框內。")

    st.markdown("---")
    st.markdown("### 🔍 知識庫資產列表")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_kw = st.text_input("輸入檔名或合約類型進行快速檢索", placeholder="例如：保密協定、維護合約...")
    
    query_filter = {}
    if search_kw:
        query_filter = {"$or": [{"file_name": {"$regex": search_kw, "$options": "i"}}, {"contract_type": {"$regex": search_kw, "$options": "i"}}]}
    
    all_docs = list(templates.find(query_filter).sort("created_at", -1))
    
    with col2:
        st.metric(label="目前資料庫總數量", value=f"{len(all_docs)} 份")

    if not all_docs:
        st.info("知識庫目前尚無符合條件的範本資料。")

    for doc in all_docs:
        with st.expander(f"📄 {doc.get('file_name', '未命名')}  |  標籤: {doc.get('contract_type', '未知')}"):
            st.write(f"**🤖 系統自動摘要**：{doc.get('summary', '無')}")
            st.write(f"**🔑 關聯關鍵字**：{', '.join(doc.get('keywords', []))}")
            st.caption(f"入庫時間: {doc.get('created_at', '')} | 數位指紋 (SHA256): `{doc.get('sha256', '')[:12]}...`")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🗑️ 永久刪除此範本 (將同步清理向量庫)", key=f"del_{doc['doc_id']}", type="primary"):
                with st.spinner("正在安全抹除資料庫與實體檔案..."):
                    collection.delete(where={"doc_id": doc["doc_id"]})
                    templates.delete_one({"doc_id": doc["doc_id"]})
                    try:
                        if os.path.exists(doc["storage_path"]):
                            os.remove(doc["storage_path"])
                    except Exception as e:
                        pass
                st.success("檔案抹除完畢！畫面即將重新載入。")
                st.rerun()