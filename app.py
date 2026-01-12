import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Fakturis Enterprise", page_icon="🏢", layout="wide")
st.title("🏢 AI Fakturis Enterprise (Fixed Logic)")
st.markdown("Perbaikan: **Akurasi Satuan (ML vs GR)**, **Deteksi Header Diosys**, & **Mapping Warna**.")

# ==========================================
# 1. KAMUS & LOGIKA PINTAR
# ==========================================

AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
}

# Mapping Nama Sales -> Nama Database (BRAND LOCK)
BRAND_ALIASES = {
    "sekawan": "AINIE",  
    "javinci": "JAVINCI",
    "thai": "THAI",
    "syb": "SYB",
    "diosys": "DIOSYS",
    "satto": "SATTO",
    "vlagio": "VLAGIO"
}

# Kamus Typo & Singkatan
KEYWORD_REPLACEMENTS = {
    "zaitun": "olive oil",         
    "kemiri": "candlenut",         
    "n.black": "natural black",    # Penting buat Diosys
    "n black": "natural black",
    "d.brwon": "dark brown",       # Typo umum
    "d.brown": "dark brown",
    "brwon": "brown",
    "coffe": "coffee",
    "cerry": "cherry",
    "shunsine": "sunshine",
    "temulawak": "temulawak",
    "hand body": "lotion",
    "hb": "lotion"
}

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    # Link Google Sheet Anda
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        df = pd.read_csv(sheet_url)
        # Pastikan kolom Merk dan Nama Barang jadi String
        df['Merek'] = df['Merek'].astype(str).str.strip()
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        
        # Buat kolom pencarian
        df['Full_Text'] = df['Merek'] + ' ' + df['Nama Barang']
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df
    except Exception as e:
        st.error(f"Gagal memuat database: {e}")
        return None

df = load_data()

# ==========================================
# 3. TRAIN AI
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
# 4. ENGINE PENCARIAN (DENGAN LOGIKA BARU)
# ==========================================
def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()

    # --- A. LOGIKA BRAND LOCK (WAJIB) ---
    if brand_filter:
        # Filter ketat: Score jadi 0 jika merk tidak sesuai
        brand_mask = df['Merek'].str.lower().str.contains(brand_filter.lower(), regex=False, na=False).to_numpy()
        final_scores = final_scores * brand_mask

    # --- B. LOGIKA HUKUMAN SATUAN (ML vs GR) ---
    # Ini memperbaiki kasus Thai Zaitun 125ml vs Sabun 100gr
    if "ml" in query_clean:
        # Jika user minta ML, hukum barang yang mengandung GR (sabun padat/cream)
        for idx, row in df.iterrows():
            if re.search(r'\b\d+gr\b', row['Clean_Text']): # Ada angka+gr (misal 100gr)
                final_scores[idx] -= 0.6 # Hukuman berat

    if "gr" in query_clean:
        # Jika user minta GR, hukum barang yang mengandung ML (cairan)
        for idx, row in df.iterrows():
            if re.search(r'\b\d+ml\b', row['Clean_Text']): 
                final_scores[idx] -= 0.6

    # --- C. LOGIKA BOOSTING SPESIFIK ---
    # Diosys 100ml
    if "100ml" in query_clean:
        for idx, row in df.iterrows():
            if "100ml" in row['Clean_Text']: final_scores[idx] += 0.5
            elif "45ml" in row['Clean_Text']: final_scores[idx] -= 0.5
    
    # Zaitun / Olive Oil
    if "olive oil" in query_clean:
        for idx, row in df.iterrows():
            if "olive oil" in row['Clean_Text']: final_scores[idx] += 0.3
            # Bonus extra kalau ada 125ml
            if "125ml" in query_clean and "125ml" in row['Clean_Text']: final_scores[idx] += 0.5

    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    # Ambang batas dinaikkan sedikit agar tidak asal tebak
    if best_score > 0.15:
        # Validasi ganda: Jika Brand Lock aktif, pastikan hasil memang dari brand itu
        if brand_filter:
            result_brand = df.iloc[best_idx]['Merek']
            if brand_filter.lower() not in result_brand.lower():
                 return "⚠️ Brand Mismatch (Cek Database)", 0.0, ""

        # Format output: KODE | NAMA BARANG
        kode = str(df.iloc[best_idx]['Kode Barang']).replace("nan","-")
        nama = df.iloc[best_idx]['Nama Barang']
        return f"{kode} | {nama}", best_score, df.iloc[best_idx]['Merek']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, ""

# ==========================================
# 5. PARSER PO (LOGIKA HEADER DIPERBAIKI)
# ==========================================
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    global_bonus = ""       
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    db_brands = df['Merek'].str.lower().unique().tolist() if df is not None else []
    # Filter merk kosong/nan
    db_brands = [str(b) for b in db_brands if b != 'nan' and len(str(b)) > 1]

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        # 1. Ganti Keyword (Sinonim)
        words = line.lower().split()
        replaced_words = []
        for w in words:
            clean_w = w.strip(",.-")
            replaced_words.append(KEYWORD_REPLACEMENTS.get(clean_w, w))
        line_processed = " ".join(replaced_words)

        # 2. Ekstraksi Angka
        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak)', line_processed, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line_processed)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        # Bersihkan teks (PENTING: Hapus qty/bonus agar deteksi brand akurat)
        clean_line = line_processed
        if qty_str: clean_line = clean_line.replace(qty_str, "")
        if bonus_str: clean_line = clean_line.replace(bonus_match.group(0), "")
        
        # Hapus simbol aneh sisa regex
        clean_keyword = re.sub(r'[^\w\s]', '', clean_line).strip() 
        
        # 3. DETEKSI HEADER (BRAND/KATEGORI)
        is_item = bool(qty_match)
        
        if not is_item:
            lower_key = clean_keyword.lower()
            detected_alias = None
            context_suffix = ""
            
            # Cek Alias
            for alias, real_brand in BRAND_ALIASES.items():
                # Pakai startswith dengan spasi agar "diosys 100ml" kena, tapi "diosy" tidak
                if lower_key == alias or lower_key.startswith(alias + " "):
                    detected_alias = real_brand
                    context_suffix = lower_key.replace(alias, "").strip()
                    break

            # Cek DB Langsung
            if not detected_alias:
                for brand in db_brands:
                    if lower_key == brand or lower_key.startswith(brand + " "):
                        detected_alias = brand
                        context_suffix = lower_key.replace(brand, "").strip()
                        break
            
            if detected_alias:
                current_brand = detected_alias 
                current_category = context_suffix 
                global_bonus = bonus_str if bonus_str else "" 
                continue 
            else:
                # Kategori Baru (bukan brand)
                if "tambahan order" in lower_key:
                    current_brand = ""
                    current_category = ""
                    global_bonus = ""
                elif len(lower_key) > 3: 
                    current_category = clean_keyword 
            continue 

        # 4. PROSES ITEM
        items_to_process = []
        
        if "semua varian" in clean_keyword.lower():
            # (Logika semua varian sama seperti sebelumnya)
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
            # PENTING: Gabungkan Brand + Kategori (100ml) + Nama Item
            final_query = f"{current_brand} {current_category} {clean_keyword}"
            items_to_process.append(final_query.strip())
            
        # 5. Cari SKU
        final_bonus = bonus_str if bonus_str else global_bonus
        
        for query in items_to_process:
            sku_res, score, detected_merek = search_sku(query, brand_filter=current_brand)
            
            results.append({
                "Brand Lock": current_brand if current_brand else "-",
                "Hasil Pencarian SKU": sku_res,
                "Qty": qty_str,
                "Bonus": final_bonus,
                "Input": query,
                "Akurasi": score
            })
            
    return store_name, results

# ==========================================
# 6. UI UTAMA
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Input PO WhatsApp")
    raw_text = st.text_area("Paste Chat di sini:", height=500, placeholder="Paste PO Anda...")
    process_btn = st.button("🚀 PROSES DATA", type="primary")

with col2:
    st.subheader("📊 Hasil Analisa Faktur")
    
    if process_btn and raw_text:
        store_name, data = parse_po_complex(raw_text)
        
        st.success(f"🏪 **Nama Toko:** {store_name}")
        
        if data:
            df_res = pd.DataFrame(data)
            
            # Format tampilan agar lebih mudah dibaca
            st.data_editor(
                df_res[["Hasil Pencarian SKU", "Qty", "Bonus", "Brand Lock", "Akurasi"]],
                column_config={
                    "Akurasi": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
                    "Hasil Pencarian SKU": st.column_config.TextColumn("KODE | NAMA BARANG", width="large")
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
        else:
            st.warning("Tidak ada item yang terdeteksi.")
