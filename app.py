import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
import io

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Fakturis Gen-AI", page_icon="🧠", layout="wide")
st.title("🧠 AI Fakturis Pro (Auto-Key)")
st.markdown("""
**Status:** API Key Terhubung Otomatis.
**Teknologi:** Google Gemini (Stable Model).
""")

# --- API KEY DITANAM DISINI ---
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
        
        # Cleaning Ringan
        df['Clean_Text'] = df['Nama'] + " " + df['Merk']
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df_db = load_data()

# ==========================================
# 3. AI ENGINE (GEMINI PRO - STABLE)
# ==========================================
def get_relevant_products(raw_text, df):
    text_lower = raw_text.lower()
    found_brands = []
    
    # Ambil daftar brand unik dari database
    all_brands = df['Merk'].dropna().unique()
    
    # Cek brand mana yang muncul di chat
    for brand in all_brands:
        if str(brand).lower() in text_lower:
            found_brands.append(brand)
    
    # Alias Manual
    aliases = {
        "kim": "KIM", "whitelab": "WHITELAB", "bonavie": "BONAVIE", 
        "goute": "GOUTE", "syb": "SYB", "yu chun mei": "YU CHUN MEI", 
        "ycm": "YU CHUN MEI"
    }
    for alias, real in aliases.items():
        if alias in text_lower:
            found_brands.append(real)
            
    if not found_brands:
        return df 
    
    return df[df['Merk'].isin(found_brands)]

def process_with_ai(api_key, raw_text, relevant_df):
    genai.configure(api_key=api_key)
    
    # --- PERBAIKAN: GANTI MODEL KE GEMINI-PRO (LEBIH STABIL) ---
    # Jika gemini-pro masih error, coba ganti string ini jadi "models/gemini-pro"
    model_name = "gemini-pro" 
    
    generation_config = {
        "temperature": 0.2, 
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        # Gemini Pro (versi lama) kadang rewel dengan response_mime_type JSON
        # Jadi kita hapus baris mime_type, dan kita parsing manual teksnya.
    }
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

        product_context = relevant_df[['Kode', 'Nama', 'Merk']].to_csv(index=False)

        # PROMPT YANG DIOPTIMALKAN UNTUK GEMINI PRO
        prompt = f"""
        Anda adalah parser PO Sales. Tugas: Ekstrak item dari CHAT dan cari Kode Barangnya di DATABASE.
        
        ATURAN:
        1. "Goute Cushion" baris baru "01: 12pcs" -> "Goute Cushion 01".
        2. "Serum:12pcs" -> Qty 12.
        3. Perbaiki Typo (Trii -> Tree).
        
        DATABASE:
        {product_context}
        
        CHAT:
        {raw_text}
        
        OUTPUT HARUS JSON ARRAY MURNI SEPERTI INI (JANGAN ADA TEKS LAIN):
        [
            {{"kode": "KODE_DB", "nama_barang": "NAMA_DB", "qty_input": "12", "keterangan": "..."}}
        ]
        """

        response = model.generate_content(prompt)
        
        # Bersihkan response (kadang AI nambahin ```json di awal)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)

    except Exception as e:
        st.error(f"AI Error ({model_name}): {e}")
        return []

# ==========================================
# 4. USER INTERFACE
# ==========================================
with st.sidebar:
    st.success("✅ API Key Terhubung")
    
    # --- FITUR DEBUGGING ---
    if st.checkbox("🔍 Cek Model Tersedia"):
        try:
            genai.configure(api_key=API_KEY_RAHASIA)
            st.write("Daftar Model yang Bisa Dipakai:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    st.code(m.name)
        except Exception as e:
            st.error(f"Gagal list model: {e}")

    if st.button("Hapus Cache"):
        st.cache_data.clear()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Input PO")
    raw_text = st.text_area("Paste Chat Sales:", height=450, placeholder="Kim kosmetik\nWhitelab\nSerum:12pcs...")
    
    process_btn = st.button("🚀 PROSES DENGAN AI", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Analisa AI")
    
    if process_btn:
        if not raw_text:
            st.warning("⚠️ Teks input kosong.")
        else:
            with st.spinner("🤖 AI Sedang Bekerja (Model: Gemini Pro)..."):
                relevant_df = get_relevant_products(raw_text, df_db)
                ai_results = process_with_ai(API_KEY_RAHASIA, raw_text, relevant_df)
                
                if ai_results:
                    st.success("Analisa Selesai!")
                    df_res = pd.DataFrame(ai_results)
                    
                    df_display = df_res.rename(columns={
                        "kode": "Kode", 
                        "nama_barang": "Nama Barang", 
                        "qty_input": "Qty",
                        "keterangan": "Ket"
                    })
                    
                    st.dataframe(df_display, hide_index=True, use_container_width=True)
                    
                    txt_output = ""
                    first_line = raw_text.split('\n')[0]
                    txt_output += f"Customer: {first_line}\n"
                    
                    for _, row in df_display.iterrows():
                        ket = f"({row['Ket']})" if row['Ket'] else ""
                        txt_output += f"{row['Kode']} | {row['Nama Barang']} | {row['Qty']} {ket}\n"
                    
                    st.text_area("Siap Copy:", value=txt_output, height=200)
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_display.to_excel(writer, index=False, sheet_name='PO')
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name="PO_AI_Result.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                else:
                    st.error("AI Gagal. Cek koneksi atau kuota API.")
