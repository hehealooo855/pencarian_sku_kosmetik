import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Fakturis Enterprise", page_icon="🏢", layout="wide")
st.title("🏢 AI Fakturis Enterprise (Auto-Correct)")
st.markdown("Fitur Baru: **Alias Brand** (Sekawan->Ainie), **Sinonim** (Zaitun->Olive Oil), & **Pewarisan Ukuran**.")

# --- 1. KAMUS & KONFIGURASI PINTAR ---

# A. AUTO VARIANTS (Untuk 'Semua Varian')
AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
}

# B. BRAND ALIASES (Mapping Nama Sales -> Nama Database)
# Kiri: Apa yang sales ketik | Kanan: Apa yang tertulis di Kolom Merk Database
BRAND_ALIASES = {
    "sekawan": "AINIE",  # <-- Request Anda Terpenuhi
    "javinci": "JAVINCI",
    "thai": "THAI",
    "syb": "SYB",
    "diosys": "DIOSYS"
}

# C. KEYWORD REPLACEMENTS (Kamus Terjemahan & Typo)
# Kiri: Kata Sales | Kanan: Kata Database
KEYWORD_REPLACEMENTS = {
    "zaitun": "olive oil",         # <-- Request Anda Terpenuhi
    "kemiri": "candlenut",         # Jaga-jaga kalau database pakai bahasa inggris
    "n.black": "natural black",    # Memperbaiki singkatan Diosys
    "n black": "natural black",
    "d.brwon": "dark brown",       # Memperbaiki typo sales
    "d.brown": "dark brown",
    "brwon": "brown",
    "coffe": "coffee",
    "cerry": "cherry",
    "shunsine": "sunshine",
    "temulawak": "temulawak",      # Jika ada translasi khusus
}

# --- 2. LOAD DATA ---
@st.cache_data(ttl=600)
def load_data():
    # GANTI LINK DI SINI DENGAN LINK CSV GOOGLE SHEET ANDA
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        df = pd.read_csv(sheet_url)
        df['Merk'] = df['Merk'].astype(str).str.strip()
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        df['Full_Text'] = df['Merk'] + ' ' + df['Nama Barang']
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df
    except Exception as e:
        st.error(f"Gagal memuat database: {e}")
        return None

df = load_data()

# --- 3. LATIH MODEL AI ---
@st.cache_resource
def train_model(data):
    if data is None or data.empty: return None, None
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])

# --- 4. FUNGSI PENCARIAN SKU (LOGIKA BOOSTING UKURAN) ---
def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()

    # A. LOGIKA BRAND LOCK
    if brand_filter:
        # Cek apakah merk di database mengandung kata kunci brand_filter
        brand_mask = df['Merk'].str.lower().str.contains(brand_filter.lower(), regex=False, na=False).to_numpy()
        final_scores = final_scores * brand_mask

    # B. LOGIKA BOOSTING UKURAN (Sangat Penting untuk Diosys)
    # Jika sales minta 100ml, kita bantai skor yang 45ml
    if "100ml" in query_clean:
        for idx, row in df.iterrows():
            if "100ml" in row['Clean_Text']: 
                final_scores[idx] += 0.4  # Boost Kuat
            elif "45ml" in row['Clean_Text'] or "45 ml" in row['Clean_Text']: 
                final_scores[idx] -= 0.5  # Hukuman Berat (Agar 45ml tenggelam)

    # Logika umum Besar/Kecil
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
        return f"{df.iloc[best_idx]['Merk']} - {df.iloc[best_idx]['Nama Barang']}", best_score, df.iloc[best_idx]['Merk']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, ""

# --- 5. LOGIKA PARSING PO (DENGAN SINONIM & ALIAS) ---
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    current_parent = ""     
    global_bonus = ""       
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    # Ambil daftar merk asli database
    db_brands = df['Merk'].str.lower().unique().tolist() if df is not None else []

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        # 1. PRE-PROCESS SINONIM (Zaitun -> Olive Oil)
        # Kita ganti kata-kata sales dengan kata database sebelum diproses
        words = line.lower().split()
        replaced_words = []
        for w in words:
            # Bersihkan tanda baca nempel (misal "zaitun,")
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
        
        # 4. DETEKSI HEADER (BRAND / KATEGORI)
        is_item = False
        if qty_match: is_item = True 
        
        if not is_item:
            lower_key = clean_keyword.lower()
            
            # Cek Alias Brand dulu (Sekawan -> Ainie)
            found_brand_header = False
            
            # Cek apakah baris ini dimulai dengan kunci Alias?
            detected_alias = None
            for alias, real_brand in BRAND_ALIASES.items():
                if lower_key == alias or lower_key.startswith(alias + " "):
                    detected_alias = real_brand
                    # Simpan sisa teks header sebagai Konteks (Penting untuk "Diosys 100ml")
                    # Misal: "Diosys 100ml" -> Brand: DIOSYS, Sisa: "100ml"
                    context_suffix = lower_key.replace(alias, "").strip()
                    break

            # Jika tidak ketemu di alias, cek di DB langsung
            if not detected_alias:
                for brand in db_brands:
                    if lower_key == brand or lower_key.startswith(brand + " "):
                        detected_alias = brand
                        context_suffix = lower_key.replace(brand, "").strip()
                        break
            
            if detected_alias:
                current_brand = detected_alias # Kunci ke Database Name (misal: AINIE)
                current_category = context_suffix # Wariskan "100ml" ke anak-anaknya
                current_parent = ""   
                
                if bonus_str: global_bonus = bonus_str
                else: global_bonus = "" 
                continue # Lanjut baris berikutnya
            
            else:
                # Kategori Baru / Parent Baru
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

        # Cek Koma (Splitter)
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
                # Disini kuncinya: current_category berisi "100ml" jika header tadi "Diosys 100ml"
                final_query = f"{current_brand} {current_category} {clean_keyword}"
            
            items_to_process.append(final_query.strip())
            
        # 6. EKSEKUSI PENCARIAN
        final_bonus = bonus_str if bonus_str else global_bonus
        
        for query in items_to_process:
            sku_res, score, detected_merk = search_sku(query, brand_filter=current_brand)
            
            results.append({
                "Brand Lock": current_brand if current_brand else "Auto",
                "Input Original": query, 
                "Hasil Pencarian SKU": sku_res,
                "Qty": qty_str,
                "Bonus": final_bonus,
                "Diskon": disc_str,
                "Akurasi": score
            })
            
    return store_name, results

# --- 6. UI UTAMA ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Input PO WhatsApp")
    st.info("Otomatis mendeteksi: Sekawan -> Ainie, Zaitun -> Olive Oil, Diosys 100ml vs 45ml.")
    raw_text = st.text_area("Paste Chat di sini:", height=500, placeholder="Paste PO Anda...")
    process_btn = st.button("🚀 PROSES DATA", type="primary")

with col2:
    st.subheader("📊 Hasil Analisa Faktur")
    
    if process_btn and raw_text:
        store_name, data = parse_po_complex(raw_text)
        
        st.success(f"🏪 **Nama Toko:** {store_name}")
        
        if data:
            df_res = pd.DataFrame(data)
            
            st.data_editor(
                df_res,
                column_config={
                    "Akurasi": st.column_config.ProgressColumn(
                        "Kecocokan", format="%.2f", min_value=0, max_value=1
                    ),
                    "Hasil Pencarian SKU": st.column_config.TextColumn(
                        "Nama Barang di Sistem", width="large"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
        else:
            st.warning("Tidak ada item yang terdeteksi.")
