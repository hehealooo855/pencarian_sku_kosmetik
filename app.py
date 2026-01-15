import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import io
import time

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Fakturis Final", page_icon="👑", layout="wide")
st.title("👑 AI Fakturis Pro (Precision Mode)")
st.markdown("""
**Status:** Precision Filter Active.
**Fitur:** Strict Size Matching (Anti Salah Ukuran) + Auto-Detect Model.
""")

# --- TEMPEL API KEY BARU ANDA DI SINI ---
API_KEY_RAHASIA = "AIzaSyBRCb7tZE_9etCicL4Td3nlb5cg9wzVgVs" 

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
        
        df['Search_Key'] = df['Nama'] + " " + df['Merk']
        df['Search_Key'] = df['Search_Key'].astype(str).str.lower()
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df_db = load_data()

# ==========================================
# 3. TEXT ENGINE (CLEANER & INJECTOR)
# ==========================================
def clean_typos(text):
    """Membersihkan typo sales"""
    replacements = {
        r"\bn\.black": "Natural Black",
        r"\bd\.brwon": "Dark Brown", 
        r"\bd\.brown": "Dark Brown",
        r"\bbrwon": "Brown",         
        r"\bcoffe\b": "Coffee",      
        r"\bcerry": "Cherry",        
        r"\bl\.blonde": "Light Blonde",
        r"\bg\.blonde": "Golden Blonde",
        r"\bbl\b": "Bleaching",
        r"\bblue bl": "Blue Bleaching",
        r"\bash": "Ash Grey",
        r"red wine": "Red Wine",
        r"tiga kenza": "Tiga Kenza",
        # Tambahan untuk kasus Aku Ayu
        r"goatmilk": "Goat Milk",
        r"greentea": "Green Tea",
    }
    text_lower = text.lower()
    for pattern, replacement in replacements.items():
        text_lower = re.sub(pattern, replacement.lower(), text_lower)
    return text_lower

def inject_context(raw_text, df):
    """
    Menempelkan Brand ke setiap baris item secara paksa.
    """
    lines = raw_text.split('\n')
    new_lines = []
    current_brand = ""
    
    all_brands = df['Merk'].dropna().unique()
    brand_list = [str(b).lower() for b in all_brands]
    
    # Update Alias: Tambahkan "aku ayu" agar terdeteksi sebagai brand
    aliases = {
        "kim": "kim", "whitelab": "whitelab", "bonavie": "bonavie", 
        "goute": "goute", "syb": "syb", "thai": "thai", "javinci": "javinci", 
        "diosys": "diosys", "implora": "implora", "aku ayu": "aku ayu",
        "hanasui": "hanasui"
    }

    for line in lines:
        clean_line = clean_typos(line) 
        
        # Deteksi Header Brand
        found_brand_in_line = False
        for b in brand_list:
            if b in clean_line:
                current_brand = b
                found_brand_in_line = True
                break
        
        if not found_brand_in_line:
            for k, v in aliases.items():
                if k in clean_line:
                    current_brand = v
                    found_brand_in_line = True
                    break
        
        # Inject Brand ke Item Anak (hanya jika ada angka)
        if re.search(r'\d+', clean_line):
            if current_brand and current_brand not in clean_line:
                new_line = f"{current_brand} {clean_line}"
                new_lines.append(new_line)
            else:
                new_lines.append(clean_line)
        else:
            new_lines.append(clean_line)
            
    return "\n".join(new_lines)

# ==========================================
# 4. SMART CONTEXT GENERATOR
# ==========================================
def get_smart_context(processed_text, df):
    all_brands = df['Merk'].dropna().unique()
    found_brands = []
    
    for brand in all_brands:
        if str(brand).lower() in processed_text.lower():
            found_brands.append(brand)
    
    if found_brands:
        brand_df = df[df['Merk'].isin(found_brands)]
        # Fallback TF-IDF
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(df['Search_Key'])
        clean_search = re.sub(r'[^a-zA-Z0-9\s]', ' ', processed_text)
        query_vec = vectorizer.transform([clean_search])
        scores = cosine_similarity(query_vec, matrix).flatten()
        top_indices = scores.argsort()[-100:][::-1]
        text_df = df.iloc[top_indices]
        return pd.concat([brand_df, text_df]).drop_duplicates(subset=['Kode'])
    else:
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(df['Search_Key'])
        clean_search = re.sub(r'[^a-zA-Z0-9\s]', ' ', processed_text)
        query_vec = vectorizer.transform([clean_search])
        scores = cosine_similarity(query_vec, matrix).flatten()
        top_indices = scores.argsort()[-100:][::-1]
        return df.iloc[top_indices]

# ==========================================
# 5. AI PROCESSOR (STRICT PROMPT)
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

def get_any_active_model():
    """Minta daftar model ke Google, ambil yang pertama kali ketemu"""
    try:
        active_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                active_models.append(m.name)
        active_models.sort(key=lambda x: 'flash' not in x)
        if active_models: return active_models
        return []
    except Exception as e:
        return []

def process_with_ai(api_key, processed_text, context_df):
    genai.configure(api_key=api_key)
    
    # 1. CARI MODEL
    available_models = get_any_active_model()
    if not available_models:
        return [], "CRITICAL: Tidak ada model AI yang ditemukan. Cek API Key."

    csv_context = context_df[['Kode', 'Nama', 'Merk']].to_csv(index=False)
    
    # --- PROMPT DIPERKETAT UNTUK UKURAN & DUPLIKASI ---
    prompt = f"""
    Role: Senior Data Validator.
    Task: Extract items from "INPUT CHAT" and match them EXACTLY with "CANDIDATE LIST".
    
    CANDIDATE LIST (Database):
    {csv_context}
    
    INPUT CHAT (Order Request):
    {processed_text}
    
    STRICT RULES (READ CAREFULLY):
    1. EXACT MATCH ONLY: You must match the "INPUT CHAT" item to a database item.
    2. SIZE IS CRITICAL: 
       - If Input says "250ml", YOU MUST PICK the 250ml item. 
       - DO NOT pick 500ml, 1kg, or 250gr if the input specifically asks for 250ml.
       - If Input says "Goatmilk", DO NOT pick "Strawberry".
    3. NO HALLUCINATION: 
       - Only list items that are EXPLICITLY requested in the INPUT CHAT with a quantity.
       - Do not list other variants from the Candidate List just because they are similar.
    4. ONE INPUT = ONE OUTPUT: Do not output duplicate lines for the same item.
    
    OUTPUT FORMAT (JSON ARRAY):
    [ {{"kode": "...", "nama_barang": "...", "qty_input": "...", "keterangan": "..."}} ]
    """

    generation_config = {
        "temperature": 0.0, # Nol Toleransi Kreativitas (Wajib Logis)
        "max_output_tokens": 8192,
    }

    last_error = ""
    for model_name in available_models:
        try:
            if "vision" in model_name: continue
            
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            response = model.generate_content(prompt)
            result = clean_and_parse_json(response.text)
            
            return result, model_name 
            
        except Exception as e:
            last_error = str(e)
            if "429" in str(e): continue
            continue
            
    return [], f"Semua model gagal. Err: {last_error}"

# ==========================================
# 6. USER INTERFACE
# ==========================================
with st.sidebar:
    st.success("✅ Precision Mode ON")
    if st.button("Hapus Cache"):
        st.cache_data.clear()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Input PO")
    raw_text = st.text_area("Paste Chat Sales:", height=450)
    process_btn = st.button("🚀 PROSES (FINAL)", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Analisa")
    
    if process_btn and raw_text:
        with st.spinner("🤖 Mengukur Size & Memproses Data..."):
            
            # 1. INJECT CONTEXT
            injected_text = inject_context(raw_text, df_db)
            
            # 2. Get Context
            smart_df = get_smart_context(injected_text, df_db)
            
            # 3. AI Process
            ai_results, info = process_with_ai(API_KEY_RAHASIA, injected_text, smart_df)
            
            if isinstance(ai_results, list) and len(ai_results) > 0:
                st.success(f"Sukses! (Model: `{info}`)")
                
                df_res = pd.DataFrame(ai_results)
                df_res.columns = [x.lower() for x in df_res.columns]
                rename_map = {"kode": "Kode", "nama_barang": "Nama Barang", "qty_input": "Qty", "keterangan": "Ket"}
                df_res = df_res.rename(columns=rename_map)
                
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
                st.download_button("📥 Download Excel", data=buffer.getvalue(), file_name="PO_Final.xlsx", mime="application/vnd.ms-excel")
                
            else:
                st.error("Gagal.")
                st.write("Detail Error:", info)
                st.warning("Pastikan API Key benar.")
