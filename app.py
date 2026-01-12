import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Fakturis Master", page_icon="💎", layout="wide")
st.title("💎 AI Fakturis Master (SKU Connected)")
st.markdown("Sistem Faktur Otomatis: Input Chat Sales -> Keluar **Kode Barang** & Nama Resmi.")

# ==========================================
# 1. KONFIGURASI PINTAR (KAMUS & LOGIKA)
# ==========================================

# A. AUTO VARIANTS (Untuk 'Semua Varian')
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
    "y2000": "Y2000"
}

# C. KEYWORD REPLACEMENTS (Kamus Terjemahan & Typo)
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
    "hb": "lotion"
}

# ==========================================
# 2. LOAD DATA CANGGIH (DENGAN KODE BARANG)
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    # LINK GOOGLE SHEET (Export CSV)
    # Pastikan file "Daftar Nama Barang" sudah diupload ke Google Sheet
    # Dan link-nya diganti di bawah ini:
    sheet_url = 'https://docs.google.com/spreadsheets/d/1A2b3C4d5E6fG7h8i9j/export?format=csv'
    
    try:
        # Trik Membaca File:
        # File Anda punya header "CV INDAH JAYA..." di baris atas.
        # Header asli (Kode Barang, Nama Barang) biasanya ada di baris ke-4 atau ke-5.
        # Kita baca dulu raw, lalu cari baris yang mengandung "Kode Barang".
        
        df_raw = pd.read_csv(sheet_url, header=None)
        
        # Cari index baris header
        header_row_idx = df_raw[df_raw.astype(str).apply(lambda x: x.str.contains('Kode Barang', case=False)).any(axis=1)].index[0]
        
        # Reload ulang dengan header yang benar
        df = pd.read_csv(sheet_url, header=header_row_idx)
        
        # Bersihkan nama kolom (kadang ada spasi)
        df.columns = df.columns.str.strip()
        
        # Ambil hanya kolom penting (Sesuaikan nama kolom dengan file asli Anda)
        # Berdasarkan file yang Anda upload: "Kode Barang", "Nama Barang", "Merek"
        cols_to_use = ['Kode Barang', 'Nama Barang', 'Merek']
        
        # Validasi kolom ada atau tidak
        for col in cols_to_use:
            if col not in df.columns:
                # Fallback jika nama kolom beda (misal 'Merk' vs 'Merek')
                if col == 'Merek' and 'Merk' in df.columns:
                    df.rename(columns={'Merk': 'Merek'}, inplace=True)
                else:
                    st.error(f"Kolom '{col}' tidak ditemukan di file. Cek header file CSV Anda.")
                    return None

        df = df[cols_to_use].copy()
        
        # Bersihkan Data
        df['Kode Barang'] = df['Kode Barang'].astype(str).str.strip().replace('nan', '-')
        df['Merek'] = df['Merek'].astype(str).str.strip().replace('nan', '')
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        
        # Gabungkan Teks untuk AI Belajar
        df['Full_Text'] = df['Merek'] + ' ' + df['Nama Barang']
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        
        return df
    except Exception as e:
        st.error(f"Gagal memuat database. Error: {e}")
        return None

df = load_data()

# ==========================================
# 3. OTAK AI (TRAINING MODEL)
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
# 4. ENGINE PENCARIAN (DENGAN BRAND LOCK)
# ==========================================
def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()

    # A. LOGIKA BRAND LOCK
    if brand_filter:
        # Filter ketat: Hanya izinkan merk yang mengandung string brand_filter
        brand_mask = df['Merek'].str.lower().str.contains(brand_filter.lower(), regex=False, na=False).to_numpy()
        final_scores = final_scores * brand_mask

    # B. LOGIKA BOOSTING UKURAN & KONTEKS KHUSUS
    # Kasus Diosys 100ml vs 45ml
    if "100ml" in query_clean:
        for idx, row in df.iterrows():
            if "100ml" in row['Clean_Text']: final_scores[idx] += 0.4
            elif "45ml" in row['Clean_Text']: final_scores[idx] -= 0.5
            
    # Kasus Umum (Besar/Kecil)
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
        # Return lengkap dengan Kode Barang
        result_str = f"{row_data['Nama Barang']}"
        return result_str, best_score, row_data['Merek'], row_data['Kode Barang']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 5. PARSER PO (PEMROSES TEKS CHAT)
# ==========================================
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    current_parent = ""     
    global_bonus = ""       
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    # Ambil list merk unik dari DB untuk deteksi header
    db_brands = df['Merek'].str.lower().unique().tolist() if df is not None else []
    # Bersihkan list brands (hapus nan/kosong)
    db_brands = [b for b in db_brands if len(str(b)) > 1]

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        # 1. REPLACE SINONIM (Zaitun -> Olive Oil)
        words = line.lower().split()
        replaced_words = []
        for w in words:
            clean_w = w.strip(",.-")
            if clean_w in KEYWORD_REPLACEMENTS:
                replaced_words.append(KEYWORD_REPLACEMENTS[clean_w])
            else:
                replaced_words.append(w)
        line_processed = " ".join(replaced_words)

        # 2. EKSTRAKSI ANGKA
        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak)', line_processed, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line_processed)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        disc_match = re.search(r'\(?([\d\+\.\s]+%)\)?', line_processed)
        disc_str = disc_match.group(1) if disc_match else ""
        
        # 3. BERSIHKAN TEKS
        clean_line = line_processed
        if qty_str: clean_line = clean_line.replace(qty_str, "")
        if bonus_str: clean_line = clean_line.replace(bonus_match.group(0), "")
        if disc_str: clean_line = clean_line.replace(disc_match.group(0), "")
        
        clean_keyword = re.sub(r'^[\s\-\.]+', '', clean_line).strip()
        
        # 4. LOGIKA JUDUL VS ITEM
        is_item = False
        if qty_match: is_item = True 
        
        if not is_item:
            lower_key = clean_keyword.lower()
            
            # Cek Alias Brand (Sekawan -> Ainie)
            detected_alias = None
            context_suffix = ""
            
            # Cek kamus alias
            for alias, real_brand in BRAND_ALIASES.items():
                if lower_key == alias or lower_key.startswith(alias + " "):
                    detected_alias = real_brand
                    context_suffix = lower_key.replace(alias, "").strip()
                    break

            # Cek DB langsung
            if not detected_alias:
                for brand in db_brands:
                    # Match exact or startswith
                    if lower_key == brand or lower_key.startswith(brand + " "):
                        detected_alias = brand
                        context_suffix = lower_key.replace(brand, "").strip()
                        break
            
            if detected_alias:
                current_brand = detected_alias 
                current_category = context_suffix # Wariskan (misal: "100ml")
                current_parent = ""   
                global_bonus = bonus_str if bonus_str else "" 
                continue 
            
            else:
                if "tambahan order" in lower_key:
                    current_brand = ""
                    current_category = ""
                    current_parent = ""
                    global_bonus = ""
                else:
                    if len(lower_key) > 3: 
                        current_category = clean_keyword 
                        current_parent = clean_keyword
            continue 

        # 5. PROSES ITEM
        items_to_process = []
        
        # Cek Semua Varian
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

        # Cek Split Koma
        elif "," in clean_keyword:
            parts = clean_keyword.split(',')
            first_part_words = parts[0].split()
            local_prefix = " ".join(first_part_words[:-1]) if len(first_part_words) > 1 else current_category
            
            items_to_process.append(f"{current_brand} {current_category} {parts[0]}".strip())
            for part in parts[1:]:
                items_to_process.append(f"{current_brand} {local_prefix} {part.strip()}".strip())
                
        # Item Tunggal
        else:
            final_query = ""
            is_short_variant = len(clean_keyword.split()) <= 2
            
            if current_parent and is_short_variant:
                final_query = f"{current_parent} {clean_keyword}"
            else:
                final_query = f"{current_brand} {current_category} {clean_keyword}"
            
            items_to_process.append(final_query.strip())
            
        # 6. EKSEKUSI
        final_bonus = bonus_str if bonus_str else global_bonus
        
        for query in items_to_process:
            nama_res, score, detected_merk, kode_barang = search_sku(query, brand_filter=current_brand)
            
            results.append({
                "Kode Barang": kode_barang,  # <-- KOLOM BARU YANG PALING PENTING
                "Nama Barang (Sistem)": nama_res,
                "Merk": detected_merk if detected_merk else current_brand,
                "Qty": qty_str,
                "Bonus": final_bonus,
                "Diskon": disc_str,
                "Input Asli": query,
                "Akurasi": score
            })
            
    return store_name, results

# ==========================================
# 6. UI (TAMPILAN WEBSITE)
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Input PO")
    st.info("Kini mendukung Kode Barang dari Database 'Daftar Nama Barang.xls'")
    raw_text = st.text_area("Paste Chat di sini:", height=500, placeholder="Paste PO Anda...")
    process_btn = st.button("🚀 PROSES DATA", type="primary")

with col2:
    st.subheader("📊 Hasil Generator Order")
    
    if process_btn and raw_text:
        store_name, data = parse_po_complex(raw_text)
        
        st.success(f"🏪 **Nama Toko:** {store_name}")
        
        if data:
            df_res = pd.DataFrame(data)
            
            # REORDER KOLOM AGAR KODE BARANG DI DEPAN
            cols = ["Kode Barang", "Nama Barang (Sistem)", "Qty", "Bonus", "Diskon", "Merk", "Akurasi"]
            df_res = df_res[cols]
            
            st.data_editor(
                df_res,
                column_config={
                    "Akurasi": st.column_config.ProgressColumn(
                        "Confidence", format="%.2f", min_value=0, max_value=1
                    ),
                    "Kode Barang": st.column_config.TextColumn(
                        "KODE SKU", help="Masukkan kode ini ke program kasir", width="medium"
                    ),
                    "Nama Barang (Sistem)": st.column_config.TextColumn(
                        "Nama Barang", width="large"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
            
            # FITUR COPY KHUSUS
            st.markdown("### 📋 Copy Text untuk Laporan")
            copy_text = f"PO: {store_name}\n"
            for item in data:
                # Format: KODE - NAMA - QTY - BONUS
                bonus_txt = f"(Bonus {item['Bonus']})" if item['Bonus'] else ""
                copy_text += f"{item['Kode Barang']} | {item['Nama Barang (Sistem)']} | {item['Qty']} {bonus_txt}\n"
            
            st.text_area("Hasil Teks:", value=copy_text, height=200)
            
        else:
            st.warning("Tidak ada item yang terdeteksi.")
