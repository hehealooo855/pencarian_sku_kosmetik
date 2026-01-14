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
st.set_page_config(page_title="AI Fakturis Hemat", page_icon="🍃", layout="wide")
st.title("🍃 AI Fakturis Pro (Smart Token Saving)")
st.markdown("""
**Status:** Token Saver Activated.
**Cara Kerja:** Mencari kandidat relevan dulu (TF-IDF), baru dikirim ke AI (Gemini). **Anti Jebol Kuota.**
""")

# --- API KEY DITANAM ---
API_KEY_RAHASIA = "AIzaSyCHDgY3z-OMdRdXuvb1aNj7vKpJWqZU2O0"

# ==========================================
# 2. LOAD DATABASE & TRAIN FILTER
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
        
        # Cleaning untuk Filter
        df['Search_Key'] = df['Nama'] + " " + df['Merk']
        df['Search_Key'] = df['Search_Key'].astype(str).str.lower()
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df_db = load_data()

# --- SIAPKAN PENYARING (TF-IDF) ---
@st.cache_resource
def prepare_filter(df):
    if df is None: return None, None
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df['Search_Key'])
    return vectorizer, matrix

if df_db is not None:
    tfidf_vec, tfidf_mat = prepare_filter(df_db)

# ==========================================
# 3. SMART CONTEXT (PENYARING)
# ==========================================
def get_optimized_context(raw_text, df, vectorizer, matrix):
    """
    Fungsi ini menyaring database. Dari ribuan barang, 
    kita hanya ambil 60 barang yang paling mungkin dimaksud.
    Supaya AI tidak 'kekenyangan' data.
    """
    # Bersihkan chat sales jadi keywords
    clean_chat = re.sub(r'[^a-zA-Z0-9\s]', ' ', raw_text.lower())
    
    # Hitung kemiripan
    query_vec = vectorizer.transform([clean_chat])
    scores = cosine_similarity(query_vec, matrix).flatten()
    
    # Ambil 60 kandidat teratas (Top-K)
    # Angka 60 ini cukup untuk memberi konteks, tapi sangat hemat token.
    top_indices = scores.argsort()[-60:][::-1]
    
    # Kembalikan DataFrame kecil
    return df.iloc[top_indices]

# ==========================================
# 4. AI ENGINE (GEMINI)
# ==========================================
def find_best_model():
    # Urutan prioritas model hemat & cepat
    candidates = ["models/gemini-1.5-flash", "models/gemini-pro"]
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for c in candidates:
            if c in models: return c
        return models[0] if models else "models/gemini-pro"
    except:
        return "models/gemini-pro"

def process_with_ai(api_key, raw_text, context_df):
    genai.configure(api_key=api_key)
    model_name = find_best_model()
    
    generation_config = {
        "temperature": 0.1, # Sangat logis/kaku
        "max_output_tokens": 4096,
    }
    
    try:
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        
        # Ubah Dataframe Kecil ke CSV String
        csv_context = context_df[['Kode', 'Nama', 'Merk']].to_csv(index=False)

        prompt = f"""
        Peran: Kamu adalah Admin Input Order (Fakturis).
        
        Tugas: 
        Baca "INPUT CHAT" dan cari item yang sesuai di "KANDIDAT PRODUK".
        
        KANDIDAT PRODUK (Pilih dari sini saja):
        {csv_context}
        
        INPUT CHAT:
        {raw_text}
        
        ATURAN LOGIKA:
        1. Context: Jika ada header "Goute Cushion", baris bawahnya "01:12pcs" berarti "Goute Cushion 01".
        2. Qty: ":12pcs" atau "x12" adalah jumlah.
        3. Typo: "Trii" -> "Tree", "Creme" -> "Cream".
        4. JANGAN HALUSINASI. Jika produk tidak ada di Kandidat, jangan dipaksa.
        
        OUTPUT (Hanya JSON Array):
        [
            {{"kode": "KODE_DB", "nama_barang": "NAMA_DB", "qty_input": "12", "keterangan": "..."}}
        ]
        """
        
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json), model_name

    except Exception as e:
        return [], str(e)

# ==========================================
# 5. USER INTERFACE
# ==========================================
with st.sidebar:
    st.success("✅ Sistem Hemat Token Aktif")
    if st.button("Hapus Cache"):
        st.cache_data.clear()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Input PO")
    raw_text = st.text_area("Paste Chat Sales:", height=450)
    process_btn = st.button("🚀 PROSES (HEMAT KUOTA)", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Analisa")
    
    if process_btn and raw_text:
        with st.spinner("🔍 Menyaring database & Bertanya ke AI..."):
            # 1. SARING DATABASE (Lokal & Cepat)
            optimized_df = get_optimized_context(raw_text, df_db, tfidf_vec, tfidf_mat)
            
            # 2. KIRIM HASIL SARINGAN KE AI (Hemat Kuota)
            ai_results, status_msg = process_with_ai(API_KEY_RAHASIA, raw_text, optimized_df)
            
            if isinstance(ai_results, list) and len(ai_results) > 0:
                st.success(f"Sukses! ({status_msg})")
                
                df_res = pd.DataFrame(ai_results)
                
                # Tampilkan
                st.dataframe(
                    df_res.rename(columns={"kode": "Kode", "nama_barang": "Nama Barang", "qty_input": "Qty", "keterangan": "Ket"}), 
                    hide_index=True, 
                    use_container_width=True
                )
                
                # Copy Text
                txt_out = f"Customer: {raw_text.splitlines()[0]}\n"
                for _, row in df_res.iterrows():
                    txt_out += f"{row['kode']} | {row['nama_barang']} | {row['qty_input']}\n"
                st.text_area("Copy Hasil:", value=txt_out, height=200)
                
                # Download Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='PO')
                st.download_button("📥 Download Excel", data=buffer.getvalue(), file_name="PO_Result.xlsx", mime="application/vnd.ms-excel")
                
            else:
                st.error(f"Gagal / Tidak Ada Data. Info: {status_msg}")
