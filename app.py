import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Fakturis Pro", page_icon="🤖", layout="wide")

st.title("🤖 AI Fakturis Pro (Context Aware)")
st.markdown("Sistem pintar yang mengerti Konteks Brand, Header Kategori, dan Multi-varian.")

# --- 1. CONFIGURATION & MAPPING ---
# DICTIONARY UNTUK "SEMUA VARIAN"
# Anda bisa menambahkan daftar varian otomatis di sini jika sales mengetik "Semua Varian"
AUTO_VARIANTS = {
    "lip mask": ["Peach", "Strawberry", "Blueberry"], # Contoh, sesuaikan dengan stok nyata
    "eye mask": ["Gold", "Osmanthus", "Seaweed"],
    "bibit pemutih": ["100ml"]
}

# --- 2. LOAD DATA ---
@st.cache_data(ttl=600)
def load_data():
    # GANTI LINK GOOGLE SHEET ANDA DI SINI
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        df = pd.read_csv(sheet_url)
        df['Merk'] = df['Merk'].astype(str).str.strip()
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        df['Full_Text'] = df['Merk'] + ' ' + df['Nama Barang']
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df
    except Exception as e:
        st.error(f"Error Database: {e}")
        return None

df = load_data()

# --- 3. LATIH MODEL (Per Brand untuk Akurasi Tinggi) ---
@st.cache_resource
def get_vectorizer(data_texts):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(data_texts)
    return vectorizer, matrix

# Kita latih Global Model dulu
if df is not None:
    global_vec, global_matrix = get_vectorizer(df['Clean_Text'])

# --- 4. FUNGSI PENCARIAN CANGGIH (BRAND LOCKING) ---
def cari_barang_terbaik(query, brand_context=None, df_source=df):
    """
    Mencari barang. Jika brand_context ada (misal: 'SYB'), 
    pencarian hanya dilakukan di dalam brand tersebut.
    """
    if not query or len(query) < 2: return "TIDAK DITEMUKAN", 0.0, ""

    # FILTER DATABASE BERDASARKAN BRAND (JIKA ADA)
    # Ini solusi agar 'Acne Sachet' tidak lari ke merk lain
    active_df = df_source
    active_matrix = global_matrix
    active_vec = global_vec
    
    # Jika brand terdeteksi, kita persempit pencarian (Filtering)
    if brand_context:
        # Cari baris yang merkn-nya mengandung kata brand context
        mask = df_source['Merk'].str.contains(brand_context, case=False, na=False)
        if mask.sum() > 0:
            active_df = df_source[mask].reset_index(drop=True)
            # Kita perlu re-train vectorizer kecil khusus untuk brand ini agar cepat & akurat
            active_vec, active_matrix = get_vectorizer(active_df['Clean_Text'])
        else:
            # Jika brand tidak ditemukan di DB, fallback ke global
            pass

    # Lakukan Pencarian
    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vector = active_vec.transform([query_clean])
    similarity_scores = cosine_similarity(query_vector, active_matrix).flatten()
    
    # Logika Besar/Kecil (Boosting)
    final_scores = similarity_scores.copy()
    if "kecil" in query_clean:
        for idx, row in active_df.iterrows():
            if re.search(r'\b(30ml|50gr|50ml|kecil)\b', row['Clean_Text']): final_scores[idx] += 0.2
            elif re.search(r'\b(200ml|500ml|besar)\b', row['Clean_Text']): final_scores[idx] -= 0.1
    
    # Ambil Pemenang
    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    if best_score > 0.1: # Threshold sedikit dilonggarkan karena sudah ada filter brand
        return f"{active_df.iloc[best_idx]['Merk']} - {active_df.iloc[best_idx]['Nama Barang']}", best_score, active_df.iloc[best_idx]['Merk']
    else:
        return "⚠️ TIDAK DITEMUKAN", 0.0, ""

# --- 5. LOGIKA PARSING STRUKTUR PO (The Brain) ---
def process_po_block(raw_text):
    lines = raw_text.split('\n')
    
    # 1. DETEKSI BRAND DI HEADER (Baris 1 & 2)
    # Kita cari apakah 2 baris pertama mengandung nama Merk yang ada di Database
    detected_brand = None
    header_lines_count = 0
    
    # Ambil list unik merk dari database untuk pengecekan
    all_brands = df['Merk'].str.lower().unique().tolist()
    
    for i in range(min(3, len(lines))):
        line_clean = lines[i].lower().strip()
        for brand in all_brands:
            # Cek exact match atau contained (misal 'syb' ada di 'syb kosmetik')
            if brand in line_clean.split(): 
                detected_brand = brand
                header_lines_count = i + 1 # Tandai baris ini sebagai header
                break
        if detected_brand: break
            
    results = []
    
    # Variabel untuk menyimpan Konteks Kategori (seperti "Peel off mask")
    current_category_context = ""
    
    # Mulai proses baris per baris setelah header
    for line in lines[header_lines_count:]:
        line = line.strip()
        if not line or line == "-": continue

        # A. Cek Apakah ini Baris Kategori/Header? (Tidak ada angka QTY)
        # Jika baris tidak ada angka (qty), kita anggap itu Header Kategori
        if not re.search(r'\d+', line):
            current_category_context = line # Simpan context, misal "Peel off mask"
            continue # Lanjut ke baris berikutnya
            
        # B. Ekstraksi Qty, Bonus, Diskon
        qty_match = re.search(r'(\d+)\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack)', line, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        disc_match = re.search(r'\(?([\d\+\.\s]+%)\)?', line)
        disc_str = disc_match.group(1) if disc_match else ""

        # Bersihkan text dari angka-angka untuk jadi keyword pencarian
        clean_item_text = line
        if qty_str: clean_item_text = clean_item_text.replace(qty_str, "")
        if bonus_str: clean_item_text = clean_item_text.replace(bonus_match.group(0), "")
        if disc_str: clean_item_text = clean_item_text.replace(disc_match.group(0), "")
        clean_item_text = re.sub(r'^[\s-]*', '', clean_item_text).strip()

        # C. LOGIKA EKSPANSI (Multi Varian & Komma)
        
        items_to_search = []
        
        # C.1 Cek "Semua Varian"
        if "semua varian" in clean_item_text.lower():
            # Cari prefix, misal "Lip mask semua varian" -> keyword "Lip mask"
            base_keyword = clean_item_text.lower().replace("semua varian", "").strip()
            # Gabungkan dengan Context Header jika ada
            full_keyword = f"{current_category_context} {base_keyword}".strip()
            
            # Cek di kamus AUTO_VARIANTS
            found_auto = False
            for key, variants in AUTO_VARIANTS.items():
                if key in full_keyword.lower():
                    for var in variants:
                        items_to_search.append(f"{full_keyword} {var}")
                    found_auto = True
                    break
            
            if not found_auto:
                # Jika tidak ada di kamus, cari apa adanya (biar user sadar harus manual)
                items_to_search.append(full_keyword)
                
        # C.2 Cek Daftar Komma (Cucumber, Blueberry, dst)
        elif "," in clean_item_text:
            # Misal: "Jelly mask cucumber, blueberry, dna salmon"
            # Strategi: Ambil kata pertama sebagai Prefix (Jelly mask)
            parts = clean_item_text.split(',')
            
            # Asumsi part pertama mengandung Prefix + Varian 1
            # Ini heuristik: Biasanya 2 kata pertama adalah prefix
            words = parts[0].split()
            prefix = " ".join(words[:-1]) # "Jelly mask"
            variant1 = words[-1]          # "cucumber"
            
            items_to_search.append(f"{current_category_context} {prefix} {variant1}")
            
            # Varian sisanya
            for p in parts[1:]:
                items_to_search.append(f"{current_category_context} {prefix} {p.strip()}")
                
        # C.3 Barang Tunggal Biasa
        else:
            # Gabungkan dengan context header (misal: "Peel off mask" + "Acne sachet")
            full_search_query = f"{current_category_context} {clean_item_text}".strip()
            items_to_search.append(full_search_query)

        # D. LAKUKAN PENCARIAN KE DB UNTUK SETIAP ITEM
        for item_query in items_to_search:
            sku_res, score, found_brand = cari_barang_terbaik(item_query, detected_brand, df)
            
            results.append({
                "Kategori/Context": current_category_context if current_category_context else "-",
                "Input Item": item_query,
                "Hasil Pencarian SKU": sku_res,
                "Qty": qty_str,
                "Bonus": bonus_str,
                "Akurasi": score
            })

    return detected_brand, results

# --- UI UTAMA ---
col1, col2 = st.columns([1, 2])

with col1:
    st.info("💡 **Tips:** Sistem otomatis mendeteksi Brand di baris awal. Baris tanpa angka dianggap Judul Kategori.")
    raw_input = st.text_area("Paste PO Di Sini", height=400, placeholder="ivana martubung\nsyb\n...\nPeel off mask\nAcne sachet 2kotak")
    run_btn = st.button("Proses Analisa", type="primary")

with col2:
    if run_btn and raw_input:
        detected_brand, data_hasil = process_po_block(raw_input)
        
        if detected_brand:
            st.success(f"🔒 **Brand Lock Aktif:** Mencari khusus produk **{detected_brand.upper()}**")
        else:
            st.warning("🔓 Brand tidak terdeteksi spesifik di header. Mencari di semua database.")
            
        if data_hasil:
            res_df = pd.DataFrame(data_hasil)
            st.data_editor(
                res_df,
                column_config={
                    "Akurasi": st.column_config.ProgressColumn(
                        "Kecocokan", format="%.2f", min_value=0, max_value=1
                    )
                },
                use_container_width=True,
                height=600
            )
        else:
            st.error("Tidak ada data yang bisa diproses.")
