import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import io

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Fakturis Stabil", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Fakturis Pro (Auto-Detect + Brand Aware)")
st.markdown("""
**Status:** Stabil.
**Teknologi:** Auto-Detect Model (Anti-404) + Brand Detection Context.
""")

# --- API KEY ---
API_KEY_RAHASIA = "AIzaSyCHDgY3z-OMdRdXuvb1aNj7vKpJWqZU2O0"

# ==========================================
# 2. LOAD DATABASE
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    try:
        df_raw = pd.read_csv(sheet_url, header=None)
        header_idx = -1
        for i, row in df_raw.iterrows():
            if any("kode barang" in str(x).lower() for x in row.tolist()):
                header_idx = i; break
        if header_idx == -1: return None

        df = pd.read_csv(sheet_url, header=header_idx)
        df.columns = df.columns.str.strip()
        
        col_map = {}
        for col in df.columns:
            c_low = col.lower()
            if "kode" in c_low and "barang" in c_low: col_map['kode'] = col
            if "nama" in c_low and "barang" in c_low: col_map['nama'] = col
            if "merek" in c_low or "merk" in c_low: col_map['merk'] = col

        if len(col_map) < 3: return None
        
        df = df.rename(columns={col_map['kode']: 'Kode', col_map['nama']: 'Nama', col_map['merk']: 'Merk'})
        df = df[['Kode', 'Nama', 'Merk']].dropna(subset=['Nama'])
        
        # Kolom bantu
        df['Search_Key'] = df['Nama'] + " " + df['Merk']
        df['Search_Key'] = df['Search_Key'].astype(str).str.lower()
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df_db = load_data()

# ==========================================
# 3. SMART CONTEXT GENERATOR (Brand Aware)
# ==========================================
def get_smart_context(raw_text, df):
    text_lower = raw_text.lower()
    all_brands = df['Merk'].dropna().unique()
    found_brands = []
    
    brand_aliases = {
        "kim": "KIM", "whitelab": "WHITELAB", "bonavie": "BONAVIE", 
        "goute": "GOUTE", "syb": "SYB", "yu chun mei": "YU CHUN MEI", 
        "ycm": "YU CHUN MEI", "thai": "THAI", "javinci": "JAVINCI",
        "diosys": "DIOSYS", "implora": "IMPLORA", "hanasui": "HANASUI"
    }

    # Cek Brand
    for brand in all_brands:
        if str(brand).lower() in text_lower:
            found_brands.append(brand)
    
    for alias, real in brand_aliases.items():
        if alias in text_lower and real not in found_brands:
            found_brands.append(real)

    # Filter Logic
    if found_brands:
        brand_df = df[df['Merk'].isin(found_brands)]
        # Jika item sedikit, tambah backup TF-IDF agar aman
        if len(brand_df) < 50:
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(df['Search_Key'])
            clean_chat = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_lower)
            query_vec = vectorizer.transform([clean_chat])
            scores = cosine_similarity(query_vec, matrix).flatten()
            top_indices = scores.argsort()[-50:][::-1]
            text_df = df.iloc[top_indices]
            return pd.concat([brand_df, text_df]).drop_duplicates(subset=['Kode'])
        else:
            return brand_df
    else:
        # Fallback biasa
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(df['Search_Key'])
        clean_chat = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_lower)
        query_vec = vectorizer.transform([clean_chat])
        scores = cosine_similarity(query_vec, matrix).flatten()
        top_indices = scores.argsort()[-100:][::-1]
        return df.iloc[top_indices]

# ==========================================
# 4. MODEL AUTO-DETECT (ANTI-404)
# ==========================================
def find_working_model():
    """Mencari model yang valid di akun user"""
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Prioritas Model (Cari yang ada di list user)
        priority_list = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro",
            "models/gemini-pro"
        ]
        
        for p in priority_list:
            if p in available_models:
                return p
        
        # Jika tidak ada yang cocok, ambil yang pertama tersedia
        if available_models:
            return available_models[0]
            
        return "models/gemini-pro" # Harapan terakhir
    except:
        return "models/gemini-pro"

# ==========================================
# 5. AI PROCESSOR
# ==========================================
def clean_and_parse_json(text_response):
    try:
        text = text_response.replace("```json", "").replace("```", "")
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx : end_idx + 1]
        elif text.strip().startswith('{'):
            text = f"[{text}]"
        return json.loads(text)
    except:
        return []

def process_with_ai(api_key, raw_text, context_df):
    genai.configure(api_key=api_key)
    
    # --- AUTO DETECT MODEL ---
    model_name = find_working_model()
    # -------------------------
    
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 8192,
    }
    
    try:
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        csv_context = context_df[['Kode', 'Nama', 'Merk']].to_csv(index=False)

        prompt = f"""
        Role: Expert Data Entry.
        Task: Extract items from INPUT CHAT based on CANDIDATE LIST.
        
        CANDIDATE LIST (Database):
        {csv_context}
        
        INPUT CHAT:
        {raw_text}
        
        RULES:
        1. Context Hierarchy: 
           - Header "Diosys 100ml" applies to lines below it (N.black, Red wine, etc) -> "Diosys 100ml Natural Black".
        2. Quantity: 
           - "12pcs (12+1)" -> Qty: 12, Bonus: 1.
        3. Typo Fix: "Cerry" -> "Cherry".
        
        OUTPUT JSON:
        [
            {{"kode": "DB_CODE", "nama_barang": "DB_NAME", "qty_input": "12", "keterangan": "Bonus 1"}}
        ]
        """
        
        response = model.generate_content(prompt)
        result = clean_and_parse_json(response.text)
        return result, model_name

    except Exception as e:
        return [], str(e)

# ==========================================
# 6. USER INTERFACE
# ==========================================
with st.sidebar:
    st.success("✅ Sistem Stabil")
    if st.button("Hapus Cache"):
        st.cache_data.clear()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Input PO")
    raw_text = st.text_area("Paste Chat Sales:", height=450)
    process_btn = st.button("🚀 PROSES (AUTO)", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Analisa")
    
    if process_btn and raw_text:
        with st.spinner("🤖 Mencari Model & Menganalisa..."):
            smart_df = get_smart_context(raw_text, df_db)
            ai_results, info = process_with_ai(API_KEY_RAHASIA, raw_text, smart_df)
            
            if isinstance(ai_results, list) and len(ai_results) > 0:
                st.success(f"Sukses! (Model: {info})")
                
                df_res = pd.DataFrame(ai_results)
                
                # Normalisasi Kolom
                df_res.columns = [x.lower() for x in df_res.columns]
                rename_map = {"kode": "Kode", "nama_barang": "Nama Barang", "qty_input": "Qty", "keterangan": "Ket"}
                df_res = df_res.rename(columns=rename_map)
                
                # Display
                cols = ["Kode", "Nama Barang", "Qty", "Ket"]
                valid_cols = [c for c in cols if c in df_res.columns]
                st.dataframe(df_res[valid_cols], hide_index=True, use_container_width=True)
                
                # Copy Text
                txt_out = f"Customer: {raw_text.splitlines()[0]}\n"
                for _, row in df_res.iterrows():
                    q = row.get('Qty', '1')
                    k = row.get('Kode', '-')
                    n = row.get('Nama Barang', 'Unknown')
                    bonus = f"({row.get('Ket')})" if row.get('Ket') else ""
                    txt_out += f"{k} | {n} | {q} {bonus}\n"
                st.text_area("Copy Hasil:", value=txt_out, height=200)
                
                # Download
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='PO')
                st.download_button("📥 Download Excel", data=buffer.getvalue(), file_name="PO_AI.xlsx", mime="application/vnd.ms-excel")
                
            else:
                st.error("Gagal.")
                st.write("Detail Error:", info)
