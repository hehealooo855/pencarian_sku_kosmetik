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
**Teknologi:** Google Gemini 1.5 Flash (LLM).
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
# 3. AI ENGINE (GEMINI 1.5 FLASH)
# ==========================================
def get_relevant_products(raw_text, df):
    """
    RAG Logic: Hanya ambil produk dari Brand yang disebut di chat
    untuk menghemat token dan meningkatkan akurasi.
    """
    text_lower = raw_text.lower()
    found_brands = []
    
    # Ambil daftar brand unik dari database
    all_brands = df['Merk'].dropna().unique()
    
    # Cek brand mana yang muncul di chat
    for brand in all_brands:
        if str(brand).lower() in text_lower:
            found_brands.append(brand)
    
    # Tambahkan Brand Alias manual jika perlu (untuk brand yang sering disingkat)
    aliases = {
        "kim": "KIM", "whitelab": "WHITELAB", "bonavie": "BONAVIE", 
        "goute": "GOUTE", "syb": "SYB", "yu chun mei": "YU CHUN MEI", 
        "ycm": "YU CHUN MEI"
    }
    for alias, real in aliases.items():
        if alias in text_lower:
            found_brands.append(real)
            
    # Jika tidak ada brand terdeteksi, kembalikan semua (atau kosongkan strategi)
    if not found_brands:
        return df # Fallback: Kirim semua (hati-hati token limit)
    
    # Filter DataFrame
    return df[df['Merk'].isin(found_brands)]

def process_with_ai(api_key, raw_text, relevant_df):
    genai.configure(api_key=api_key)
    
    # Konfigurasi Model
    generation_config = {
        "temperature": 0.2, # Rendah agar tidak halusinasi
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Ubah data produk jadi String CSV ringkas untuk konteks AI
    product_context = relevant_df[['Kode', 'Nama', 'Merk']].to_csv(index=False)

    # THE ULTIMATE PROMPT
    prompt = f"""
    Anda adalah asisten input order (Fakturis) yang sangat teliti.
    
    TUGAS:
    Ekstrak item pesanan dari "INPUT CHAT" dan cocokkan dengan "DATABASE PRODUK" secara tepat.
    
    ATURAN KONTEKS (PENTING):
    1. Chat sering menggunakan struktur Induk-Anak. 
       Contoh:
       "Goute Cushion" (Header)
       "01: 12pcs" (Item Anak -> Artinya "Goute Cushion 01")
    2. Jika ada kata "All" atau "Campur", pecahkan menjadi varian yang mungkin ada di database jika masuk akal, atau biarkan sebagai catatan.
    3. Hati-hati dengan Brand. Jangan tertukar antar brand.
    4. Perbaiki Typo secara otomatis (misal "Trii" -> "Tree", "Creme" -> "Cream").
    5. Kenali format qty aneh: ":12pcs", "x12", "12pc", "@12".
    
    DATABASE PRODUK (Hanya pilih dari sini):
    {product_context}
    
    INPUT CHAT:
    {raw_text}
    
    OUTPUT FORMAT (JSON Array):
    [
        {{"kode": "KODE_DARI_DB", "nama_barang": "NAMA_PERSIS_DARI_DB", "qty_input": "12", "keterangan": "Bonus/Note jika ada"}}
    ]
    
    Jika item tidak ditemukan di database sama sekali, beri kode "UNKNOWN".
    """

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"AI Error: {e}")
        return []

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
            with st.spinner("🤖 AI sedang membaca konteks, memperbaiki typo, dan mencocokkan database..."):
                # 1. Filter DB biar irit token & fokus
                relevant_df = get_relevant_products(raw_text, df_db)
                
                # 2. Kirim ke Gemini (Pakai Kunci Otomatis)
                ai_results = process_with_ai(API_KEY_RAHASIA, raw_text, relevant_df)
                
                if ai_results:
                    st.success("Analisa Selesai!")
                    df_res = pd.DataFrame(ai_results)
                    
                    # Rename untuk tampilan
                    df_display = df_res.rename(columns={
                        "kode": "Kode", 
                        "nama_barang": "Nama Barang", 
                        "qty_input": "Qty",
                        "keterangan": "Ket"
                    })
                    
                    st.dataframe(df_display, hide_index=True, use_container_width=True)
                    
                    # Fitur Copy
                    txt_output = ""
                    # Cek nama toko (Baris pertama biasanya)
                    first_line = raw_text.split('\n')[0]
                    txt_output += f"Customer: {first_line}\n"
                    
                    for _, row in df_display.iterrows():
                        ket = f"({row['Ket']})" if row['Ket'] else ""
                        txt_output += f"{row['Kode']} | {row['Nama Barang']} | {row['Qty']} {ket}\n"
                    
                    st.text_area("Siap Copy:", value=txt_output, height=200)
                    
                    # Download Excel
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
                    st.error("AI tidak mengembalikan data. Coba cek format input atau koneksi.")
