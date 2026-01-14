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
st.set_page_config(page_title="AI Fakturis Anti-Error", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Fakturis Pro (Bulletproof JSON)")
st.markdown("""
**Status:** JSON Guard Activated.
**Teknologi:** TF-IDF Filter + Gemini AI + Regex Cleaner.
""")

# --- API KEY ---
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
        
        df['Search_Key'] = df['Nama'] + " " + df['Merk']
        df['Search_Key'] = df['Search_Key'].astype(str).str.lower()
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df_db = load_data()

# --- TF-IDF FILTER (PENYARING) ---
@st.cache_resource
def prepare_filter(df):
    if df is None: return None, None
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df['Search_Key'])
    return vectorizer, matrix

if df_db is not None:
    tfidf_vec, tfidf_mat = prepare_filter(df_db)

# ==========================================
# 3. FUNGSI PEMBERSIH (THE FIXER)
# ==========================================
def clean_and_parse_json(text_response):
    """
    Fungsi sakti untuk memperbaiki JSON yang rusak atau kotor.
    """
    try:
        # 1. Hapus Markdown Code Block (```json ... ```)
        text = text_response.replace("```json", "").replace("```", "")
        
        # 2. Bracket Hunter: Cari [ pertama dan ] terakhir
        # Ini membuang teks intro seperti "Berikut adalah hasilnya:"
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx : end_idx + 1]
        else:
            # Jika tidak ada kurung siku, mungkin AI cuma kasih satu objek {}
            # Kita bungkus jadi list
            if text.strip().startswith('{'):
                text = f"[{text}]"
        
        # 3. Parsing
        return json.loads(text)
        
    except json.JSONDecodeError as e:
        # Jika masih error, return pesan error spesifik tapi jangan crash
        print(f"JSON Error: {e}")
        return []

def get_optimized_context(raw_text, df, vectorizer, matrix):
    # Bersihkan chat sales
    clean_chat = re.sub(r'[^a-zA-Z0-9\s]', ' ', raw_text.lower())
    query_vec = vectorizer.transform([clean_chat])
    scores = cosine_similarity(query_vec, matrix).flatten()
    
    # Ambil 50 kandidat teratas (Hemat Token!)
    top_indices = scores.argsort()[-50:][::-1]
    return df.iloc[top_indices]

# ==========================================
# 4. AI ENGINE (GEMINI)
# ==========================================
def process_with_ai(api_key, raw_text, context_df):
    genai.configure(api_key=api_key)
    
    # Gunakan gemini-1.5-flash karena lebih cepat & murah.
    # Jika kuota habis, dia akan error di awal, bukan di parsing.
    model_name = "models/gemini-1.5-flash"
    
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 4096,
        # Kita paksa output JSON mode (Fitur baru Gemini)
        "response_mime_type": "application/json" 
    }
    
    try:
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        
        csv_context = context_df[['Kode', 'Nama', 'Merk']].to_csv(index=False)

        prompt = f"""
        Role: Expert Data Entry.
        Task: Extract items from INPUT CHAT based on CANDIDATE LIST.
        
        CANDIDATE LIST:
        {csv_context}
        
        INPUT CHAT:
        {raw_text}
        
        RULES:
        1. Context Awareness: "Goute Cushion" followed by line "01:12pcs" means "Goute Cushion 01".
        2. Quantity: ":12pcs", "x12", "12 pcs" -> qty_input: 12.
        3. Typo Fix: Map "Trii" to "Tree", "Creme" to "Cream".
        4. If item is NOT in CANDIDATE LIST, do NOT invent data. Ignore it.
        
        Return strictly a JSON Array like this:
        [
            {{"kode": "CODE_FROM_DB", "nama_barang": "NAME_FROM_DB", "qty_input": "12", "keterangan": "Bonus info"}}
        ]
        """
        
        response = model.generate_content(prompt)
        
        # Panggil Fungsi Pembersih Sakti
        result = clean_and_parse_json(response.text)
        return result, model_name

    except Exception as e:
        # Fallback ke Gemini Pro jika Flash error
        try:
            fallback_model = "models/gemini-pro"
            model = genai.GenerativeModel(fallback_model)
            response = model.generate_content(prompt) # Prompt sama
            result = clean_and_parse_json(response.text)
            return result, fallback_model
        except Exception as e2:
            return [], f"Error: {str(e)} | Fallback Error: {str(e2)}"

# ==========================================
# 5. USER INTERFACE
# ==========================================
with st.sidebar:
    st.success("✅ Anti-Error System ON")
    if st.button("Hapus Cache"):
        st.cache_data.clear()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Input PO")
    raw_text = st.text_area("Paste Chat Sales:", height=450)
    process_btn = st.button("🚀 PROSES (SAFE MODE)", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Analisa")
    
    if process_btn and raw_text:
        with st.spinner("🔍 Menyaring & Memperbaiki Struktur Data..."):
            # 1. Filter Database
            optimized_df = get_optimized_context(raw_text, df_db, tfidf_vec, tfidf_mat)
            
            # 2. Proses AI dengan Pembersih JSON
            ai_results, model_used = process_with_ai(API_KEY_RAHASIA, raw_text, optimized_df)
            
            if isinstance(ai_results, list) and len(ai_results) > 0:
                st.success(f"Berhasil! (Model: {model_used})")
                
                df_res = pd.DataFrame(ai_results)
                
                # Normalisasi Kolom (Jaga-jaga AI pakai huruf besar/kecil)
                df_res.columns = [x.lower() for x in df_res.columns]
                # Mapping ke nama yang bagus
                rename_map = {
                    "kode": "Kode", "nama_barang": "Nama Barang", 
                    "qty_input": "Qty", "keterangan": "Ket",
                    "nama": "Nama Barang" # Jaga-jaga AI halusinasi nama kolom
                }
                df_res = df_res.rename(columns=rename_map)
                
                # Pastikan kolom ada
                cols_to_show = ["Kode", "Nama Barang", "Qty"]
                if "Ket" in df_res.columns: cols_to_show.append("Ket")
                
                # Tampilkan hanya kolom yang valid
                valid_cols = [c for c in cols_to_show if c in df_res.columns]
                st.dataframe(df_res[valid_cols], hide_index=True, use_container_width=True)
                
                # Text Area Copy
                txt_out = f"Customer: {raw_text.splitlines()[0]}\n"
                for _, row in df_res.iterrows():
                    q = row.get('Qty', '1')
                    n = row.get('Nama Barang', 'Unknown')
                    k = row.get('Kode', '-')
                    ket = f"({row.get('Ket', '')})" if row.get('Ket') else ""
                    txt_out += f"{k} | {n} | {q} {ket}\n"
                st.text_area("Copy Hasil:", value=txt_out, height=200)
                
                # Download Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='PO')
                st.download_button("📥 Download Excel", data=buffer.getvalue(), file_name="PO_Fixed.xlsx", mime="application/vnd.ms-excel")
                
            else:
                st.warning("AI tidak menemukan item yang cocok, atau format respon tidak terbaca.")
                if isinstance(ai_results, str):
                    st.error(f"Detail Error: {ai_results}")
