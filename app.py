import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 0. KONFIGURASI HALAMAN (Wajib Paling Atas)
# ==========================================
st.set_page_config(
    page_title="Sistem Faktur Pintar", 
    page_icon="🛍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. KONFIGURASI KAMUS DATA
# ==========================================
AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
    "jelly mask": ["Cucumber", "Blueberry", "DNA Salmon", "Mugwort", "Watermelon"],
}

BRAND_ALIASES = {
    "sekawan": "AINIE", "javinci": "JAVINCI", "thai": "THAI", 
    "syb": "SYB", "diosys": "DIOSYS", "satto": "SATTO", 
    "esene": "ESENE", "y2000": "Y2000", "hanasui": "HANASUI", 
    "implora": "IMPLORA", "vlagio": "VLAGIO", "honor": "HONOR"
}

KEYWORD_REPLACEMENTS = {
    "zaitun": "olive oil", "kemiri": "candlenut", "n.black": "natural black",    
    "n black": "natural black", "d.brwon": "dark brown", "d.brown": "dark brown",
    "brwon": "brown", "coffe": "coffee", "cerry": "cherry", 
    "shunsine": "sunshine", "temulawak": "temulawak", "hand body": "lotion",
    "hb": "lotion", "lulur": "body scrub"
}

# ==========================================
# 2. LOAD DATA (Smart Header)
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    # ------------------------------------------------------------------
    # GANTI LINK DI BAWAH INI DENGAN LINK GOOGLE SHEET ANDA
    # ------------------------------------------------------------------
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        df_raw = pd.read_csv(sheet_url, header=None)
        header_idx = -1
        for i, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower().tolist()
            if any("kode barang" in x for x in row_str):
                header_idx = i
                break
        
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

        df = df.rename(columns={
            col_map['kode']: 'Kode Barang',
            col_map['nama']: 'Nama Barang',
            col_map['merk']: 'Merek'
        })
        
        df = df[['Kode Barang', 'Nama Barang', 'Merek']].copy()
        df = df.dropna(subset=['Nama Barang'])
        df['Kode Barang'] = df['Kode Barang'].astype(str).str.strip().replace('nan', '-')
        df['Merek'] = df['Merek'].astype(str).str.strip().replace('nan', '')
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        df['Full_Text'] = df['Merek'] + ' ' + df['Nama Barang']
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df

    except Exception: return None

df = load_data()

# ==========================================
# 3. TRAINING AI
# ==========================================
@st.cache_resource
def train_model(data):
    if data is None or data.empty: return None, None
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])

# ==========================================
# 4. LOGIKA PENCARIAN & PARSING (Engine)
# ==========================================
def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""
    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    final_scores = similarity_scores.copy()

    if brand_filter:
        brand_mask = df['Merek'].str.lower().str.contains(brand_filter.lower(), regex=False, na=False).to_numpy()
        final_scores = final_scores * brand_mask

    if "100ml" in query_clean:
        for idx, row in df.iterrows():
            if "100ml" in row['Clean_Text']: final_scores[idx] += 0.4
            elif "45ml" in row['Clean_Text']: final_scores[idx] -= 0.5
    
    # Logika Besar/Kecil
    if "kecil" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(30ml|50gr|50ml|45ml|60ml|kecil|mini|sachet)\b', row['Clean_Text']): final_scores[idx] += 0.25
            elif re.search(r'\b(200ml|500ml|besar|jumbo|1000ml)\b', row['Clean_Text']): final_scores[idx] -= 0.15
    if "besar" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(200ml|250ml|500ml|1000ml|besar|jumbo)\b', row['Clean_Text']): final_scores[idx] += 0.25
            elif re.search(r'\b(30ml|50gr|kecil|mini|sachet)\b', row['Clean_Text']): final_scores[idx] -= 0.15

    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    if best_score > 0.1:
        row = df.iloc[best_idx]
        return row['Nama Barang'], best_score, row['Merek'], row['Kode Barang']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    current_brand = ""      
    current_category = ""   
    current_parent = ""     
    global_bonus = ""       
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    db_brands = df['Merek'].str.lower().unique().tolist() if df is not None else []
    db_brands = [b for b in db_brands if len(str(b)) > 1]

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        words = line.lower().split()
        replaced_words = [KEYWORD_REPLACEMENTS.get(w.strip(",.-"), w) for w in words]
        line_processed = " ".join(replaced_words)

        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak|dos)', line_processed, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line_processed)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        clean_keyword = re.sub(r'^[\s\-\.]+', '', line_processed.replace(qty_str, "").replace(bonus_str, "")).strip()
        
        if not qty_match:
            lower_key = clean_keyword.lower()
            detected_alias = None
            context_suffix = ""
            
            for alias, real_brand in BRAND_ALIASES.items():
                if lower_key == alias or lower_key.startswith(alias + " "):
                    detected_alias = real_brand
                    context_suffix = lower_key.replace(alias, "").strip()
                    break
            if not detected_alias:
                for brand in db_brands:
                    if lower_key == brand or lower_key.startswith(brand + " "):
                        detected_alias = brand
                        context_suffix = lower_key.replace(brand, "").strip()
                        break
            
            if detected_alias:
                current_brand = detected_alias 
                current_category = context_suffix 
                current_parent = ""   
                global_bonus = bonus_str if bonus_str else "" 
                continue 
            else:
                if "tambahan order" in lower_key:
                    current_brand = ""
                    current_category = ""
                    current_parent = ""
                    global_bonus = ""
                elif len(lower_key) > 3: 
                    current_category = clean_keyword 
                    current_parent = clean_keyword
            continue 

        items_to_process = []
        if "semua varian" in clean_keyword.lower():
            found = False
            full_chk = f"{current_category} {clean_keyword}".lower()
            for key, vars in AUTO_VARIANTS.items():
                if key in full_chk:
                    base = clean_keyword.lower().replace("semua varian", "").strip()
                    prefix = f"{current_brand} {current_category} {base}".strip()
                    for v in vars: items_to_process.append(f"{prefix} {v}")
                    found = True; break
            if not found: items_to_process.append(f"{current_brand} {current_category} {clean_keyword}")
        elif "," in clean_keyword:
            parts = clean_keyword.split(',')
            local_prefix = " ".join(parts[0].split()[:-1]) if len(parts[0].split()) > 1 else current_category
            items_to_process.append(f"{current_brand} {current_category} {parts[0]}")
            for p in parts[1:]: items_to_process.append(f"{current_brand} {local_prefix} {p}")
        else:
            if current_parent and len(clean_keyword.split()) <= 2:
                items_to_process.append(f"{current_parent} {clean_keyword}")
            else:
                items_to_process.append(f"{current_brand} {current_category} {clean_keyword}")

        final_bonus = bonus_str if bonus_str else global_bonus
        for q in items_to_process:
            nama, score, merk, kode = search_sku(q, brand_filter=current_brand)
            results.append({
                "Kode Barang": kode, "Nama Barang (Sistem)": nama, "Merk": merk if merk else current_brand,
                "Qty": qty_str, "Bonus": final_bonus, "Akurasi": score
            })
    return store_name, results

# ==========================================
# 5. USER INTERFACE (GRAFIS & GAMBAR)
# ==========================================

# --- A. SIDEBAR ---
with st.sidebar:
    # Gambar Logo (Gunakan URL gambar Logo perusahaan Anda jika ada, atau placeholder ini)
    st.image("https://cdn-icons-png.flaticon.com/512/2897/2897785.png", width=100)
    st.markdown("### **CV. INDAH JAYA LESTARI**")
    st.markdown("---")
    st.markdown("**Status Database:**")
    if df is not None:
        st.success(f"🟢 Terhubung ({len(df)} SKU)")
    else:
        st.error("🔴 Terputus")
    
    st.markdown("---")
    if st.button("🔄 Refresh Data Sheet"):
        st.cache_data.clear()
        st.rerun()
    st.info("Tekan tombol ini jika ada update barang baru di Google Sheet.")

# --- B. HEADER UTAMA ---
# Banner Gambar Profesional
st.image("https://images.unsplash.com/photo-1553413077-190dd305871c?auto=format&fit=crop&w=1200&q=80", use_column_width=True)

st.title("🛍️ Sistem Faktur Pintar")
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    font-size: 20px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# --- C. AREA KERJA UTAMA ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 1. Input Pesanan (WhatsApp)")
    st.caption("Paste chat order dari Sales di bawah ini:")
    raw_text = st.text_area("Input Chat", height=450, placeholder="Contoh:\nIvana Martubung\nSYB\nEye mask semua varian per lusin...")
    
    process_btn = st.button("🚀 PROSES ORDER SEKARANG")

with col2:
    st.markdown("### 2. Hasil Generator Faktur")
    
    if process_btn and raw_text:
        with st.spinner('Sedang menganalisa orderan...'):
            store_name, data = parse_po_complex(raw_text)
        
        if data:
            # Tampilkan Kartu Informasi Toko
            st.info(f"🏪 **Pelanggan:** {store_name}")
            
            df_res = pd.DataFrame(data)
            
            # Tampilan Data Editor yang Elegan
            st.data_editor(
                df_res[["Kode Barang", "Nama Barang (Sistem)", "Qty", "Bonus", "Merk", "Akurasi"]],
                column_config={
                    "Akurasi": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
                    "Kode Barang": st.column_config.TextColumn("KODE SKU", width="medium"),
                    "Nama Barang (Sistem)": st.column_config.TextColumn("Nama Barang", width="large"),
                    "Merk": st.column_config.TextColumn("Brand", width="small"),
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Area Copy Paste dengan Format Khusus
            st.markdown("#### 📋 Salin untuk Faktur")
            st.caption("Blok teks di bawah dan paste ke program kasir.")
            
            copy_text = f"Customer: {store_name}\n----------------------------------\n"
            for item in data:
                bonus_txt = f"(Bonus {item['Bonus']})" if item['Bonus'] else ""
                # Format Copy: KODE - NAMA - QTY
                copy_text += f"{item['Kode Barang']} \t {item['Nama Barang (Sistem)']} \t {item['Qty']} {bonus_txt}\n"
            
            st.code(copy_text, language="text")
            
        else:
            st.warning("⚠️ Tidak ada item yang dapat dikenali. Pastikan format ada Qty (pcs/lsn/box).")
    else:
        # Placeholder Image jika belum ada hasil
        st.markdown("Belum ada data yang diproses.")
        st.image("https://cdn-icons-png.flaticon.com/512/7486/7486744.png", width=150, caption="Menunggu Input...")
