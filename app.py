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
st.title("🧠 AI Fakturis Pro (Auto-Model)")
st.markdown("""
**Status:** API Key Terhubung.
**Teknologi:** Google Gemini (Auto-Detect Available Model).
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
        df['Clean_Text'] = df['Nama'] + " " + df['Merk']
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df_db = load_data()

# ==========================================
# 3. AI ENGINE (SMART SELECTOR)
# ==========================================
def get_relevant_products(raw_text, df):
    text_lower = raw_text.lower()
    found_brands = []
    all_brands = df['Merk'].dropna().unique()
    
    for brand in all_brands:
        if str(brand).lower() in text_lower:
            found_brands.append(brand)
    
    aliases = {
        "kim": "KIM", "whitelab": "WHITELAB", "bonavie": "BONAVIE", 
        "goute": "GOUTE", "syb": "SYB", "yu chun mei": "YU CHUN MEI", 
        "ycm": "YU CHUN MEI"
    }
    for alias, real in aliases.items():
        if alias in text_lower:
            found_brands.append(real)
            
    if not found_brands: return df 
    return df[df['Merk'].isin(found_brands)]

def find_best_model():
    """Fungsi ini mencari model yang tersedia di akun pengguna"""
    try:
        # Prioritas model dari yang terbaru ke yang lama
        preferred_order = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro",
            "models/gemini-pro"
        ]
        
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Cek apakah model prioritas ada di list available
        for model in preferred_order:
            if model in available_models:
                return model
        
        # Jika prioritas tidak ada, ambil apa saja yang ada (biasanya gemini-pro)
        if available_models:
            return available_models[0]
            
        return "models/gemini-pro" # Fallback terakhir
    except Exception as e:
        return "models/gemini-pro"

def process_with_ai(api_key, raw_text, relevant_df):
    genai.configure(api_key=api_key)
    
    # --- AUTO DETECT MODEL ---
    model_name = find_best_model()
    # -------------------------
    
    generation_config = {
        "temperature": 0.1, 
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

        product_context = relevant_df[['Kode', 'Nama', 'Merk']].to_csv(index=False)

        prompt = f"""
        Peran: Kamu adalah sistem parser Purchase Order (PO).
        Tugas: Ekstrak item dari CHAT, perbaiki typo, dan cari Kode Barangnya di DATABASE.
        
        DATABASE PRODUK:
        {product_context}
        
        INPUT CHAT:
        {raw_text}
        
        ATURAN PENTING:
        1. "Goute Cushion" baris bawahnya "01: 12pcs" -> Artinya "Goute Cushion 01". (Gabungkan header dengan varian).
        2. Format ":12pcs" atau "x12" adalah jumlah (Qty).
        3. Typo: "Trii" -> "Tree", "Creme" -> "Cream".
        4. "All" atau "Campur" -> Jangan dipecah jika tidak yakin, ambil item yang paling umum atau beri kode "MANUAL_CHECK".
        
        FORMAT OUTPUT (Hanya JSON Array Valid):
        [
            {{"kode": "KODE_DARI_DB", "nama_barang": "NAMA_DARI_DB", "qty_input": "12", "keterangan": "..."}}
        ]
        """

        response = model.generate_content(prompt)
        
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json), model_name

    except Exception as e:
        st.error(f"AI Error ({model_name}): {e}")
        return [], model_name

# ==========================================
# 4. USER INTERFACE
# ==========================================
with st.sidebar:
    st.success("✅ API Key Terhubung")
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
            with st.spinner("🤖 AI Sedang Mencari Model & Menganalisa..."):
                relevant_df = get_relevant_products(raw_text, df_db)
                ai_results, used_model = process_with_ai(API_KEY_RAHASIA, raw_text, relevant_df)
                
                if ai_results:
                    st.success(f"Sukses! Menggunakan Model: `{used_model}`")
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
                    st.error("Gagal mendapatkan hasil. Cek log error di atas.")
