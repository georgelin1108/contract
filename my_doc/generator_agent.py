import json
import ollama
from docxtpl import DocxTemplate
import datetime

class MultiSourceGenerator:
    def __init__(self, template_path):
        self.template_path = template_path

    def generate_contract_context(self, email_text, meeting_notes, user_wishes):
        """
        核心整合模組：將郵件、會議紀錄、使用者需求餵給 LLM
        """
        prompt = f"""
        你是一位全球人壽的法務助理。請根據以下多個來源，提取出最精確、最新的合約資訊。
        如果不同來源之間有衝突，請以「使用者需求」為最高優先，其次是「會議紀錄」，最後是「郵件」。

        【郵件往來】：
        {email_text}

        【會議紀錄】：
        {meeting_notes}

        【使用者個人需求/想法】：
        {user_wishes}

        請嚴格輸出 JSON 格式，包含以下欄位：
        - vendor_name (廠商全稱)
        - amount (合約總額，純數字)
        - period (合約有效期限)
        - special_terms (特殊條款，根據整合後的共識撰寫)
        - sla (服務水準協議)
        """
        
        try:
            response = ollama.generate(model='llama3', prompt=prompt, format='json')
            extracted_data = json.loads(response['response'])
            return extracted_data
        except Exception as e:
            return f"整合失敗: {str(e)}"

    def create_word_contract(self, data):
        """
        將整合後的數據填入 Word 模板
        """
        doc = DocxTemplate(self.template_path)
        context = {
            **data,
            "today": datetime.date.today().strftime("%Y-%m-%d"),
            "amount_formatted": f"{int(data['amount']):,}"
        }
        doc.render(context)
        file_name = f"Generated_Contract_{data['vendor_name']}.docx"
        doc.save(file_name)
        return file_name

# --- 測試執行範例 ---
if __name__ == "__main__":
    generator = MultiSourceGenerator("maintenance_template.docx")
    
    # 模擬多源數據
    email = "對方在郵件說報價 120 萬，但我們還在議價。"
    meeting = "會議中雙方達成共識，金額調降至 110 萬，含稅。"
    user_idea = "我希望在合約中加入一條：每季必須提供一次健康檢查報告，且 SLA 提高到 99.99%。"
    
    # 1. 執行整合
    print("🤖 AI 正在整合多來源資訊...")
    final_data = generator.generate_contract_context(email, meeting, user_idea)
    print(f"✅ 整合後的 JSON 資料：\n{json.dumps(final_data, indent=2, ensure_ascii=False)}")
    
    # 2. 生成 Word (如果模板已準備好)
    # generator.create_word_contract(final_data)