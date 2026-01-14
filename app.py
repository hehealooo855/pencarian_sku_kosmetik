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
st.set_page_config(page_title="AI Fakturis Ultimate", page_icon="💎", layout="wide")
st.title("💎 AI Fakturis Pro (Auto-Detect Model)")
st.markdown("""
**Status:** Auto-Detect Model Active.
**Fix:** Mencari model AI yang tersedia secara otomatis (Anti Error 404).
""")

# --- PENTING: TEMPEL KUNCI BARU ANDA DI SINI ---
API_KEY_RAHASIA = "AIzaSyD5Oz4FQo4o0lpj6PvcUYHYZp7XDVJa-qc" 

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
# 3. TEXT PRE-PROCESSOR (KAMUS SINGKATAN)
# ==========================================
def expand_abbreviations(text):
    """
    Mengubah singkatan sales menjadi nama lengkap agar AI paham.
    Contoh: 'N.black' -> 'Natural Black'
    """
    replacements = {
        r"\bn\.black": "Natural Black",
        r"\bd\.brown": "Dark Brown",
        r"\bl\.blonde": "Light Blonde",
        r"\bg\.blonde": "Golden Blonde",
        r"\bbl\b": "Bleaching",
        r"\bcerry": "Cherry",
        r"\bblue bl": "Blue Bleaching",
        r"\bash": "Ash Grey",
        r"tiga kenza": "Tiga Kenza", 
    }
    
    text_lower = text.lower()
    for pattern, replacement in replacements.items():
        text_lower = re.sub(pattern, replacement.lower(), text_lower)
    
    return text_lower

# ==========================================
# 4. SMART CONTEXT GENERATOR
# ==========================================
def get_smart_context(raw_text, df):
    # Bersihkan teks dulu dengan Pre-Processor
    clean_text = expand_abbreviations(raw_text)
    
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
        if str(brand).lower() in clean_text:
            found_brands.append(brand)
    
    for alias, real in brand_aliases.items():
        if alias in clean_text and real not in found_brands:
            found_brands.append(real)

    if found_brands:
        # Ambil SEMUA item dari brand yang terdeteksi
        brand_df = df[df['Merk'].isin(found_brands)]
        
        # Tambahkan backup TF-IDF untuk item non-brand
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(df['Search_Key'])
        clean_search = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_text)
        query_vec = vectorizer.transform([clean_search])
        scores = cosine_similarity(query_vec, matrix).flatten()
        top_indices = scores.argsort()[-100:][::-1]
        text_df = df.iloc[top_indices]
        
        return pd.concat([brand_df, text_df]).drop_duplicates(subset=['Kode'])
    else:
        # Fallback biasa
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(df['Search_Key'])
        clean_search = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_text)
        query_vec = vectorizer.transform([clean_search])
        scores = cosine_similarity(query_vec, matrix).flatten()
        top_indices = scores.argsort()[-100:][::-1]
        return df.iloc[top_indices]

# ==========================================
# 5. JSON CLEANER
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

# ==========================================
# 6. AI PROCESSOR (AUTO DETECT MODEL)
# ==========================================
def get_available_model(api_key):
    """Mencari model yang tersedia di akun user secara otomatis"""
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Prioritas: Flash -> 1.5 Pro -> 1.0 Pro -> Gemini Pro
        for m in models:
            if 'flash' in m.lower(): return m
        for m in models:
            if '1.5' in m and 'pro' in m.lower(): return m
        
        return models[0] if models else "models/gemini-pro"
    except:
        return "models/gemini-pro"

def process_with_ai(api_key, raw_text, context_df):
    # Cari model yang valid dulu
    model_name = get_available_model(api_key)
    
    # Pre-process text sebelum dikirim ke AI
    optimized_text = expand_abbreviations(raw_text)
    
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
        {optimized_text}
        
        INSTRUCTIONS:
        1. Process EVERY line. Do NOT skip any line containing quantity.
        2. Context Hierarchy: 
           - Header "Diosys 100ml" applies to ALL lines below it (N.black, Brown, Coffee, etc) until a new brand appears.
           - "Brown" under "Diosys" means "Diosys Brown".
        3. Quantity Logic: 
           - "(24+3)" in Header means items are bundled/bonus.
           - Line "N.black 12pcs" means Qty: 12.
           - "12pcs (12+1)" means Qty: 12, Note: Bonus 1.
        
        OUTPUT JSON:
        [
            {{"kode": "DB_CODE", "nama_barang": "DB_NAME", "qty_input": "12", "keterangan": "Bonus info"}}
        ]
        """
        
        response = model.generate_content(prompt)
        result = clean_and_parse_json(response.text)
        return result, model_name

    except Exception as e:
        return [], f"{str(e)} (Model tried: {model_name})"

# ==========================================
# 7. USER INTERFACE
# ==========================================
with st.sidebar:
    st.success("✅ AI Auto-Detect Aktif")
    if st.button("Hapus Cache"):
        st.cache_data.clear()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Input PO")
    raw_text = st.text_area("Paste Chat Sales:", height=450)
    process_btn = st.button("🚀 PROSES", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Analisa")
    
    if process_btn and raw_text:
        with st.spinner("🤖 Mencari Model AI & Menganalisa..."):
            
            # 1. Get Context
            smart_df = get_smart_context(raw_text, df_db)
            
            # 2. AI Process
            ai_results, info = process_with_ai(API_KEY_RAHASIA, raw_text, smart_df)
            
            if isinstance(ai_results, list) and len(ai_results) > 0:
                st.success(f"Sukses! (Model Used: {info})")
                
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
                first_line = raw_text.splitlines()[0]
                txt_out = f"Customer: {first_line}\n"
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
                st.download_button("📥 Download Excel", data=buffer.getvalue(), file_name="PO_Full.xlsx", mime="application/vnd.ms-excel")
                
            else:
                st.error("Gagal membaca respon AI.")
                st.write("Detail Error:", info)
