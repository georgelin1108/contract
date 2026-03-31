import os
import re
import datetime
import logging
from typing import Dict, Any, List, Tuple, Optional

import ollama
from docxtpl import DocxTemplate

from config import (
    MODEL, ALL_TOPICS_FOR_PROMPT, TOPIC_KEYWORDS, TOPIC_ALIAS,
    CRITICAL_RISK_TRIGGERS, HIGH_RISK_TRIGGERS
)

from utils import (
    safe_json_load, normalize_text, detect_topics, lexical_score, score_topic_overlap,
    detect_contract_mode_from_text, build_article_map, normalize_topic_name,
    article_to_key, short_text, split_draft_into_articles, extract_text_from_pdf,
    extract_text_from_docx, make_output_path, normalize_term, save_upload_file, parse_template_selector
)

from database import (
    query_templates_fulltext, query_template_chunks_by_query, find_history_by_vendor_keyword,
    template_exists_by_sha256, insert_template_doc, upsert_template_vectors,
    get_template_by_selector, search_templates_sql
)

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

# 文件入庫與意圖分析
def llm_ingest_contract(text: str) -> Dict[str, Any]:
    prompt = f"""
你是法務文件入庫助理。請只輸出合法 JSON。
請根據以下合約內容抽取 metadata。
core_topics 請優先從下列主題中選 3~10 個：
{", ".join(ALL_TOPICS_FOR_PROMPT)}

合約內容：
{text[:7000]}

JSON 格式：
{{
  "contract_type": "請根據合約真實屬性，精煉出標準的合約類型名稱（例如：維護合約、租賃合約、採購合約、授權合約等，限10字以內）",
  "summary": "80~160字摘要",
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

def handle_upload(files):
    from utils import chunk_text # 避免循環依賴延遲載入
    inserted = 0
    skipped = 0

    for f in files:
        meta = save_upload_file(f)
        exist = template_exists_by_sha256(meta["sha256"])
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

        chunks = chunk_text(text)
        
        insert_template_doc(doc)
        upsert_template_vectors(doc, text, chunks)
        inserted += 1

    return inserted, skipped

def llm_parse_user_request(message: str) -> Dict[str, Any]:
    prompt = f"""
你是合約助理。請把需求整理成 JSON，不要輸出其他文字。
JSON 結構：
{{
  "intent": "generate | review",
  "contract_type": "維護合約/其他",
  "fields": {{"party_a": "", "party_b": "", "amount": "", "term": "", "system_name": ""}},
  "notes": ""
}}
使用者訊息：{message}
"""
    obj = ollama_json(prompt)
    if "fields" not in obj or not isinstance(obj["fields"], dict):
        obj["fields"] = {}
    obj["intent"] = str(obj.get("intent", "generate") or "generate").strip()
    obj["contract_type"] = str(obj.get("contract_type", "其他") or "其他").strip()
    return obj

# RAG 檢索與關聯模板選擇
def select_review_templates(draft_text: str, articles: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    candidates = query_templates_fulltext(draft_text, n_results=max(12, top_k * 4))
    draft_topics = detect_topics(draft_text)
    article_topic_union = set()
    for a in articles:
        article_topic_union.update(a.get("topics", []))

    ranked = []
    seen_file_names = set()

    for c in candidates:
        file_name = c.get("file_name", "")
        if file_name in seen_file_names:
            continue
        seen_file_names.add(file_name)

        real_template_topics = detect_topics(c.get("source_text", ""))
        text_block = " ".join([
            c.get("file_name", ""), c.get("contract_type", ""), c.get("summary", ""),
            " ".join(c.get("keywords", [])), " ".join(real_template_topics),
        ])

        score = lexical_score(text_block, draft_text[:1200])
        score += score_topic_overlap(draft_topics, real_template_topics) * 4
        score += score_topic_overlap(list(article_topic_union), real_template_topics) * 3

        if detect_contract_mode_from_text(draft_text) == "混合型" and c.get("contract_type") in ["維護合約", "保密協定", "開發合約"]:
            score += 2

        c["core_topics"] = real_template_topics
        ranked.append((score, c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked[:top_k]]

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

    return selected_templates, all_chunks[:40], articles

def search_template_chunks_for_article(article: Dict[str, Any], selected_templates: List[Dict[str, Any]], n_results: int = 10) -> List[Dict[str, Any]]:
    doc_ids = [x["doc_id"] for x in selected_templates if x.get("doc_id")]
    article_text = article.get("content", "")
    article_topics = article.get("topics", [])

    queries = [article_text[:900]]
    for topic in article_topics:
        kw_list = TOPIC_KEYWORDS.get(topic, [])
        if kw_list:
            queries.append(f"{topic} " + " ".join(kw_list[:6]))
        queries.append(f"{topic} {article_text[:120]}")

    seen = set()
    out = []
    
    for q in queries:
        refs = query_template_chunks_by_query(q, doc_ids, n_results=15)
        for r in refs:
            key = (r["doc_id"], r["content"][:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)

    scored = []
    for r in out:
        score = score_topic_overlap(article_topics, r.get("topics", []))
        score += lexical_score(r.get("content", ""), article_text[:300]) * 2
        
        r_content = normalize_text(r.get("content", ""))
        for t in article_topics:
            if any(k in r_content for k in TOPIC_KEYWORDS.get(t, [])):
                score += 15 # 權重加大，確保重要條款排在最前
                
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:n_results]]

# 審查邏輯 
def llm_review_single_article(article: Dict[str, Any], article_key: str, candidate_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_block = []
    source_names = []

    for i, c in enumerate(candidate_chunks, start=1):
        fname = c["file_name"]
        source_names.append(fname)
        chunk_block.append(f"【依據 {i}】檔名：{fname}\n內容：{c['content']}")

    allowed_sources = "、".join(sorted(set(source_names))) if source_names else "無"
    prompt_topics = "、".join(ALL_TOPICS_FOR_PROMPT)

    prompt = f"""
你是企業法務審查 AI，立場固定站在甲方。請根據提供的【模板依據片段】，審查【草稿條文】是否有偏離、衝突或遺漏。

嚴格限制：
1. 只能輸出 JSON，不可輸出多餘的文字。
2. 判斷標準【必須完全依據模板片段】。絕不可憑空捏造、猜測或引用「行業慣例」。若模板片段中沒有相關規定，請直接略過不要將其列為問題。
3. issue_topic 必須嚴格且完全照抄下列清單之一，絕對不可自行合併或創造新詞（例如：只能寫「維護時間」，不可寫「維護時間與人力」）：{prompt_topics}
4. 若單一草稿條款包含多個風險主題（如同時涉及時間與違約金），請務必拆分成多筆獨立的 JSON 物件。
5. analysis (風險分析)：具體說明草稿與模板的差異，以及對甲方的潛在風險。
6. suggestion (建議修正)：直接依據模板的具體數字或標準給出具體修改方向。不可自行發明模板沒有的新制度或新期限。
7. template_snippet (模板原文)：請從【模板依據片段】中，精準擷取與此風險最相關的「原文字句」（請挑選最關鍵的1~2句即可，切勿整段照抄）。
8. source 欄位必須填入對應的真實檔名：{allowed_sources}。絕不可留空。

輸出格式：
{{
  "major_issues": [
    {{
      "article_key": "{article_key}",
      "clause": "條款名稱",
      "issue_topic": "對應的主題",
      "type": "deviation 或 conflict",
      "risk": "Critical/High/Medium/Low",
      "template_snippet": "(動態生成) 乙方應於接獲甲方需求反應單後之 4 小時內回覆，並於 24 小時內排除故障...",
      "analysis": "(動態生成) 草稿約定為3個工作天，但模板標準為24小時內，這會延誤甲方系統恢復時間。",
      "suggestion": "(動態生成) 建議依模板規定，將修復時限修正為「24小時內」。",
      "source": "真實模板檔名.docx"
    }}
  ],
  "general_issues": []
}}

【草稿條文】
{article.get("content", "")}

【模板依據片段】
{chr(10).join(chunk_block)}
"""
    return ollama_json(prompt)

def infer_missing_topics_from_templates(selected_templates: List[Dict[str, Any]], articles: List[Dict[str, Any]], draft_text: str) -> List[str]:
    template_topics = set()
    for t in selected_templates:
        template_topics.update(t.get("core_topics", []))
    draft_topics = set()
    for a in articles:
        draft_topics.update(a.get("topics", []))
        
    missing = set()
    for topic in template_topics:
        if topic not in draft_topics:
            missing.add(topic)
    return sorted(missing)

def review_articles_individually(draft_text: str, selected_templates: List[Dict[str, Any]], articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_major = []
    all_general = []

    for idx, article in enumerate(articles, start=1):
        article_key = article_to_key(article, idx)
        candidate_chunks = search_template_chunks_for_article(article, selected_templates, n_results=10)
        article_result = llm_review_single_article(article, article_key, candidate_chunks)

        all_major.extend(article_result.get("major_issues", []))
        all_general.extend(article_result.get("general_issues", []))

    missing_topics = infer_missing_topics_from_templates(selected_templates, articles, draft_text)
    missing_clauses = []
    
    for topic in missing_topics:
        source = ""
        for t in selected_templates:
            if topic in t.get("core_topics", []) or topic in detect_topics(t.get("source_text", "")):
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

def normalize_review_json(raw: Dict[str, Any], used_templates: List[Dict[str, Any]], articles: List[Dict[str, Any]], top_chunks: List[Dict[str, Any]], original_draft_text: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    article_map = build_article_map(articles)
    
    def norm_issue(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict): return None

        article_key = str(item.get("article_key", "") or "").strip()
        clause = str(item.get("clause", "未命名項目")).strip()
        topic = normalize_topic_name(str(item.get("issue_topic", clause) or clause))
        source = normalize_text(str(item.get("source", "") or ""))

        article = article_map.get(article_key, {"content": ""})
        draft_text = normalize_text(article.get("content", "") or "")

        # 直接取用 LLM 擷取的精華原文字句
        extracted_snippet = str(item.get("template_snippet", "詳見模板內容")).strip()

        issue = {
            "clause": clause,
            "risk": str(item.get("risk", "Medium")).strip(),
            "draft_text": draft_text,
            "template_basis": extracted_snippet,
            "analysis": normalize_text(str(item.get("analysis", ""))),
            "suggestion": normalize_text(str(item.get("suggestion", ""))),
            "source": source,
            "type": str(item.get("type", "deviation")).strip(),
            "issue_topic": topic,
            "article_key": article_key,
        }

        dt = draft_text
        topic_norm = normalize_topic_name(topic)

        for trigger in CRITICAL_RISK_TRIGGERS.get(topic_norm, []):
            if trigger in dt:
                issue["risk"] = "Critical"
                issue["type"] = "conflict"
                break
                
        if issue["risk"] != "Critical":
            for trigger in HIGH_RISK_TRIGGERS.get(topic_norm, []):
                if trigger in dt:
                    issue["risk"] = "High"
                    break
        
        return issue

    raw_major = [x for x in (norm_issue(i) for i in raw.get("major_issues", [])) if x]
    raw_general = [x for x in (norm_issue(i) for i in raw.get("general_issues", [])) if x]
    
    def dedup_issues(items):
        seen = set()
        out = []
        for it in items:
            key = (it.get("article_key", ""), it.get("issue_topic", ""), it.get("source", ""))
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    major_issues = dedup_issues(raw_major)
    general_issues = dedup_issues(raw_general)

    uniq_templates = []
    seen_template_names = set()
    for t in used_templates:
        fname = t.get("file_name", "未知模板")
        if fname not in seen_template_names:
            seen_template_names.add(fname)
            uniq_templates.append({
                "file_name": fname,
                "contract_type": t.get("contract_type", "其他"),
                "summary": t.get("summary", ""),
                "core_topics": t.get("core_topics", []),
            })

    normalized = {
        "contract_type_guess": raw.get("contract_type_guess", "未判定"),
        "summary": "系統已根據動態檢索出的模板條文完成草稿審查，並結合預設紅線整理出潛在風險。",
        "used_templates": uniq_templates,
        "major_issues": major_issues,
        "general_issues": general_issues,
        "missing_clauses": raw.get("missing_clauses", []),
        "score": 85 
    }
    
    return normalized

# 報價風險預警
def assess_price_risk(user_input: str) -> str:
    prompt = f"""
你是一位專業的金融合約分析官。請分析內容並嚴格輸出 JSON 格式。
{{"vendor_name": "廠商名稱", "amount": 1000000}}
輸入內容：{user_input[:2000]}
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
    past_records = find_history_by_vendor_keyword(search_keyword)

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

# DOCX 套版生成
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