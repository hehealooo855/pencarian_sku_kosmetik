import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Fakturis Master", page_icon="💎", layout="wide")
st.title("💎 AI Fakturis Master (Google Sheet Connected)")
st.markdown("Sistem Faktur Otomatis: Database terhubung ke Google Sheet.")

# ==========================================
# 1. KONFIGURASI PINTAR
# ==========================================

# A. AUTO VARIANTS (Kamus Varian Otomatis)
AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
    "jelly mask": ["Cucumber", "Blueberry", "DNA Salmon", "Mugwort", "Watermelon"],
}

# B. BRAND ALIASES (Mapping Nama Sales -> Nama Database)
BRAND_ALIASES = {
    "sekawan": "AINIE",  
    "javinci": "JAVINCI",
    "thai": "THAI",
    "syb": "SYB",
    "diosys": "DIOSYS",
    "satto": "SATTO",
    "esene": "ESENE",
    "y2000": "Y2000",
    "hanasui": "HANASUI",
    "implora": "IMPLORA"
}

# C. KEYWORD REPLACEMENTS (Kamus Typo & Istilah)
KEYWORD_REPLACEMENTS = {
    "zaitun": "olive oil",         
    "kemiri": "candlenut",         
    "n.black": "natural black",    
    "n black": "natural black",
    "d.brwon": "dark brown",       
    "d.brown": "dark brown",
    "brwon": "brown",
    "coffe": "coffee",
    "cerry": "cherry",
    "shunsine": "sunshine",
    "temulawak": "temulawak",
    "hand body": "lotion",
    "hb": "lotion",
    "lulur": "body scrub"
}

# ==========================================
# 2. LOAD DATA (SMART HEADER DETECTION)
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    # -------------------------------------------------------------
    # 1. MASUKKAN LINK GOOGLE SHEET ANDA DI BAWAH INI (SEKALI SAJA)
    # Pastikan akhiran link diganti menjadi: /export?format=csv
    # -------------------------------------------------------------
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        # Baca dulu mentah-mentah (tanpa header)
        df_raw = pd.read_csv(sheet_url, header=None)
        
        # Cari baris keberapa yang mengandung kata "Kode Barang"
        # Ini mengatasi masalah Kop Surat CV INDAH JAYA yang ikut ter-copy
        header_idx = -1
        for i, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower().tolist()
            # Cek apakah ada kata 'kode barang' dan 'nama barang' di baris ini
            if any("kode barang" in x for x in row_str):
                header_idx = i
                break
        
        if header_idx == -1:
            st.error("Gagal menemukan Header 'Kode Barang' di Google Sheet. Pastikan Anda copy-paste header tabelnya juga.")
            return None

        # Load ulang data dimulai dari baris header yang benar
        df = pd.read_csv(sheet_url, header=header_idx)
        
        # Bersihkan nama kolom (hapus spasi depan/belakang)
        df.columns = df.columns.str.strip()
        
        # Normalisasi Nama Kolom (Jaga-jaga beda ketik dikit)
        # Kita cari kolom yang mengandung kata kunci tertentu
        col_map = {}
        for col in df.columns:
            c_low = col.lower()
            if "kode" in c_low and "barang" in c_low: col_map['kode'] = col
            if "nama" in c_low and "barang" in c_low: col_map['nama'] = col
            if "merek" in c_low or "merk" in c_low: col_map['merk'] = col

        if len(col_map) < 3:
            st.error(f"Kolom tidak lengkap. Ditemukan: {list(col_map.keys())}. Butuh: Kode Barang, Nama Barang, Merek.")
            return None

        # Rename kolom ke standar kita
        df = df.rename(columns={
            col_map['kode']: 'Kode Barang',
            col_map['nama']: 'Nama Barang',
            col_map['merk']: 'Merek'
        })
        
        # Ambil kolom penting saja
        df = df[['Kode Barang', 'Nama Barang', 'Merek']].copy()
        
        # Bersihkan Data (Hapus baris kosong/nan)
        df = df.dropna(subset=['Nama Barang'])
        df['Kode Barang'] = df['Kode Barang'].astype(str).str.strip().replace('nan', '-')
        df['Merek'] = df['Merek'].astype(str).str.strip().replace('nan', '')
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        
        # Buat Kolom Teks Gabungan untuk AI
        df['Full_Text'] = df['Merek'] + ' ' + df['Nama Barang']
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        
        return df

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca Google Sheet: {e}")
        return None

df = load_data()

# ==========================================
# 3. TRAINING MODEL AI
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
# 4. ENGINE PENCARIAN (BRAND LOCK + BOOSTING)
# ==========================================
def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()

    # LOGIKA BRAND LOCK
    if brand_filter:
        brand_mask = df['Merek'].str.lower().str.contains(brand_filter.lower(), regex=False, na=False).to_numpy()
        final_scores = final_scores * brand_mask

    # LOGIKA BOOSTING (UKURAN & JENIS)
    if "100ml" in query_clean:
        for idx, row in df.iterrows():
            if "100ml" in row['Clean_Text']: final_scores[idx] += 0.4
            elif "45ml" in row['Clean_Text']: final_scores[idx] -= 0.5
            
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
        row_data = df.iloc[best_idx]
        return row_data['Nama Barang'], best_score, row_data['Merek'], row_data['Kode Barang']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 5. PARSER PO (LOGIKA UTAMA)
# ==========================================
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    current_parent = ""     
    global_bonus = ""       
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    db_brands = df['Merek'].str.lower().unique().tolist() if df is not None else []
    db_brands = [b for b in db_brands if len(str(b)) > 1] # Filter merk kosong

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        # 1. Ganti Keyword (Sinonim)
        words = line.lower().split()
        replaced_words = []
        for w in words:
            clean_w = w.strip(",.-")
            if clean_w in KEYWORD_REPLACEMENTS:
                replaced_words.append(KEYWORD_REPLACEMENTS[clean_w])
            else:
                replaced_words.append(w)
        line_processed = " ".join(replaced_words)

        # 2. Ambil Angka
        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak|dos)', line_processed, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line_processed)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        disc_match = re.search(r'\(?([\d\+\.\s]+%)\)?', line_processed)
        disc_str = disc_match.group(1) if disc_match else ""
        
        clean_line = line_processed
        if qty_str: clean_line = clean_line.replace(qty_str, "")
        if bonus_str: clean_line = clean_line.replace(bonus_match.group(0), "")
        if disc_str: clean_line = clean_line.replace(disc_match.group(0), "")
        clean_keyword = re.sub(r'^[\s\-\.]+', '', clean_line).strip()
        
        # 3. Cek Header vs Item
        is_item = bool(qty_match)
        
        if not is_item:
            lower_key = clean_keyword.lower()
            detected_alias = None
            context_suffix = ""
            
            # Cek Alias
            for alias, real_brand in BRAND_ALIASES.items():
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

        # 4. Proses Item (Split & Expand)
        items_to_process = []
        
        if "semua varian" in clean_keyword.lower():
            found_in_dict = False
            full_check_str = f"{current_category} {clean_keyword}".lower()
            for key, variants in AUTO_VARIANTS.items():
                if key in full_check_str:
                    base_name = clean_keyword.lower().replace("semua varian", "").strip()
                    prefix = f"{current_brand} {current_category} {base_name}".strip()
                    for var in variants:
                        items_to_process.append(f"{prefix} {var}")
                    found_in_dict = True
                    break
            if not found_in_dict:
                items_to_process.append(f"{current_brand} {current_category} {clean_keyword}")

        elif "," in clean_keyword:
            parts = clean_keyword.split(',')
            first_part_words = parts[0].split()
            local_prefix = " ".join(first_part_words[:-1]) if len(first_part_words) > 1 else current_category
            items_to_process.append(f"{current_brand} {current_category} {parts[0]}".strip())
            for part in parts[1:]:
                items_to_process.append(f"{current_brand} {local_prefix} {part.strip()}".strip())
                
        else:
            final_query = ""
            is_short_variant = len(clean_keyword.split()) <= 2
            if current_parent and is_short_variant:
                final_query = f"{current_parent} {clean_keyword}"
            else:
                final_query = f"{current_brand} {current_category} {clean_keyword}"
            items_to_process.append(final_query.strip())
            
        # 5. Cari SKU
        final_bonus = bonus_str if bonus_str else global_bonus
        
        for query in items_to_process:
            nama_res, score, detected_merk, kode_barang = search_sku(query, brand_filter=current_brand)
            
            results.append({
                "Kode Barang": kode_barang,
                "Nama Barang (Sistem)": nama_res,
                "Merk": detected_merk if detected_merk else current_brand,
                "Qty": qty_str,
                "Bonus": final_bonus,
                "Input": query,
                "Akurasi": score
            })
            
    return store_name, results

# ==========================================
# 6. UI WEBSITE
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Input PO")
    st.info("Database: Terhubung ke Google Sheet.")
    raw_text = st.text_area("Paste Chat Sales:", height=400)
    
    if st.button("🔄 Refresh Database", help="Klik jika baru saja update Google Sheet"):
        st.cache_data.clear()
        st.rerun()
        
    process_btn = st.button("🚀 PROSES DATA", type="primary")

with col2:
    st.subheader("📊 Hasil Generator Order")
    
    if process_btn and raw_text:
        store_name, data = parse_po_complex(raw_text)
        
        if data:
            df_res = pd.DataFrame(data)
            
            # Tampilan Tabel
            st.success(f"Toko: **{store_name}**")
            st.data_editor(
                df_res[["Kode Barang", "Nama Barang (Sistem)", "Qty", "Bonus", "Merk", "Akurasi"]],
                column_config={
                    "Akurasi": st.column_config.ProgressColumn(
                        "Confidence", format="%.2f", min_value=0, max_value=1
                    ),
                    "Kode Barang": st.column_config.TextColumn("KODE SKU", width="medium"),
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Text Area untuk Copy
            st.markdown("### 📋 Copy Hasil")
            copy_text = f"PO: {store_name}\n"
            for item in data:
                bonus_txt = f"({item['Bonus']})" if item['Bonus'] else ""
                # Format: KODE | NAMA | QTY
                copy_text += f"{item['Kode Barang']} | {item['Nama Barang (Sistem)']} | {item['Qty']} {bonus_txt}\n"
            
            st.text_area("Siap Salin:", value=copy_text, height=200)
            
        else:
            st.warning("Tidak ada item terdeteksi.")
