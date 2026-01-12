import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Fakturis Ultimate", page_icon="💎", layout="wide")
st.title("💎 AI Fakturis Ultimate (Zero Mistake Mode)")
st.markdown("Fitur: **Hard Unit Filter**, **Aggressive Regex Cleaning**, & **Strict Brand Locking**.")

# ==========================================
# 1. KAMUS DATA & MAPPING
# ==========================================

# A. AUTO VARIANTS (Untuk 'Semua Varian')
AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
}

# B. BRAND ALIASES (Mapping Nama Sales -> Nama Database)
BRAND_ALIASES = {
    "sekawan": "AINIE", "javinci": "JAVINCI", "thai": "THAI", 
    "syb": "SYB", "diosys": "DIOSYS", "satto": "SATTO", 
    "vlagio": "VLAGIO", "honor": "HONOR", "hanasui": "HANASUI",
    "implora": "IMPLORA", "brasov": "BRASOV", "felinz": "FELINZ"
}

# C. KEYWORD REPLACEMENTS (Kamus Typo & Singkatan)
KEYWORD_REPLACEMENTS = {
    "zaitun": "olive oil", "kemiri": "candlenut", 
    "n.black": "natural black", "n black": "natural black", 
    "d.brwon": "dark brown", "d.brown": "dark brown", "d brown": "dark brown",
    "brwon": "brown", "coffe": "coffee", "cerry": "cherry", 
    "shunsine": "sunshine", "temulawak": "temulawak", 
    "hand body": "lotion", "hb": "lotion", "bl": "body lotion"
}

# ==========================================
# 2. LOAD DATA (CANGGIH)
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    # Link Google Sheet (Pastikan format=csv)
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        df = pd.read_csv(sheet_url)
        # Pastikan kolom string
        df['Merk'] = df['Merk'].astype(str).str.strip().replace('nan', '')
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        df['Kode Barang'] = df['Kode Barang'].astype(str).str.strip().replace('nan', '-')
        
        # Buat kolom pencarian bersih
        df['Full_Text'] = df['Merk'] + ' ' + df['Nama Barang']
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df
    except Exception as e:
        st.error(f"Gagal memuat database: {e}")
        return None

df = load_data()

# ==========================================
# 3. TRAIN AI MODEL
# ==========================================
@st.cache_resource
def train_model(data):
    if data is None or data.empty: return None, None
    # Ngram 1-5 untuk menangkap kata super pendek (misal: "Ash", "Red")
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{2,}', ngram_range=(1, 3)) 
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])

# ==========================================
# 4. ENGINE PENCARIAN (STRICT LOGIC)
# ==========================================
def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()

    # --- 1. BRAND LOCK (KACAMATA KUDA) ---
    if brand_filter:
        # Buat Mask: True jika merk sesuai, False jika tidak
        # Gunakan regex=False agar karakter spesial tidak merusak
        brand_mask = df['Merk'].str.lower().str.contains(brand_filter.lower(), regex=False, na=False).to_numpy()
        # Matikan total skor brand lain
        final_scores = final_scores * brand_mask

    # --- 2. HARD UNIT FILTER (HUKUMAN MATI) ---
    # Jika user minta ML, haram hukumnya dikasih GR (Sabun Padat). Dan sebaliknya.
    if re.search(r'\b\d+\s*ml\b', query_clean): # User minta ML
        for idx, row in df.iterrows():
            if re.search(r'\b\d+\s*gr\b', row['Clean_Text']): # Database item GR
                final_scores[idx] = 0.0 # MATIKAN

    if re.search(r'\b\d+\s*gr\b', query_clean): # User minta GR
        for idx, row in df.iterrows():
            if re.search(r'\b\d+\s*ml\b', row['Clean_Text']): # Database item ML
                final_scores[idx] = 0.0 # MATIKAN

    # --- 3. CONTEXT BOOSTING ---
    # Boost Ukuran Diosys
    if "100ml" in query_clean:
        for idx, row in df.iterrows():
            if "100ml" in row['Clean_Text']: final_scores[idx] += 0.5
            elif "45ml" in row['Clean_Text']: final_scores[idx] -= 1.0 # Hukuman diperberat

    # Boost Warna/Varian Pendek
    # Jika query pendek (misal "Red Wine"), kita cari exact match di nama barang
    if len(query.split()) <= 3:
        for idx, row in df.iterrows():
            if query_clean in row['Clean_Text']:
                final_scores[idx] += 0.3

    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    # Threshold Validasi
    min_threshold = 0.15
    if brand_filter: min_threshold = 0.05 # Kalau brand sudah kekunci, turunkan threshold biar varian aneh tetap ketemu

    if best_score > min_threshold:
        row = df.iloc[best_idx]
        
        # Validasi Terakhir: Jika Brand Lock aktif, pastikan Brand Output mengandung Filter
        if brand_filter and brand_filter.lower() not in row['Merk'].lower():
             return "⚠️ Brand Mismatch", 0.0, "", "-"

        return row['Nama Barang'], best_score, row['Merk'], row['Kode Barang']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 5. PARSER PO (CLEANING AGRESIF)
# ==========================================
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    global_bonus = ""       
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    # Daftar Brand DB untuk deteksi
    db_brands = df['Merk'].str.lower().unique().tolist() if df is not None else []
    db_brands = [str(b) for b in db_brands if len(str(b)) > 1]

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        # A. CLEANING RAW (Hapus dalam kurung DULUAN)
        # Hapus (24+3), (12+1), (10+5%) SEBELUM diproses agar tidak mengganggu deteksi Brand
        # Regex: Hapus apapun di dalam kurung (...)
        line_no_brackets = re.sub(r'\([^)]*\)', '', line)
        
        # B. GANTI KEYWORD (Sinonim)
        words = line_no_brackets.lower().split()
        replaced_words = []
        for w in words:
            clean_w = w.strip(",.-")
            replaced_words.append(KEYWORD_REPLACEMENTS.get(clean_w, w))
        line_processed = " ".join(replaced_words)

        # C. EKSTRAKSI ANGKA (Dari line asli yg masih ada kurungnya utk bonus)
        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak)', line, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        # Keyword Bersih untuk AI
        clean_keyword = line_processed.replace(qty_str.lower(), "").strip()
        # Hapus karakter non-alphanumeric sisa
        clean_keyword = re.sub(r'[^\w\s,]', '', clean_keyword).strip()
        
        # D. LOGIKA HEADER vs ITEM
        is_item = bool(qty_match)
        
        if not is_item:
            lower_key = clean_keyword.lower()
            detected_alias = None
            context_suffix = ""
            
            # Cek Alias Brand
            for alias, real_brand in BRAND_ALIASES.items():
                if lower_key == alias or lower_key.startswith(alias + " "):
                    detected_alias = real_brand
                    context_suffix = lower_key.replace(alias, "").strip()
                    break

            # Cek DB Langsung
            if not detected_alias:
                for brand in db_brands:
                    # Match exact word untuk menghindari salah deteksi (misal "The" kena "Thai")
                    # Pakai regex boundary \b
                    if re.search(r'\b' + re.escape(brand) + r'\b', lower_key):
                        detected_alias = brand
                        # Hapus nama brand dari keyword untuk jadi kategori (misal: "Diosys 100ml" -> "100ml")
                        context_suffix = lower_key.replace(brand, "").strip()
                        break
            
            if detected_alias:
                current_brand = detected_alias 
                current_category = context_suffix # Mewariskan "100ml"
                global_bonus = bonus_str if bonus_str else "" 
                continue 
            else:
                # Judul Kategori non-brand
                if "tambahan order" in lower_key:
                    current_brand = ""
                    current_category = ""
                    global_bonus = ""
                elif len(lower_key) > 3: 
                    current_category = clean_keyword 
            continue 

        # E. PROSES ITEM
        items_to_process = []
        
        # Logic Semua Varian
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

        # Logic Split Koma
        elif "," in clean_keyword:
            parts = clean_keyword.split(',')
            local_prefix = " ".join(parts[0].split()[:-1]) if len(parts[0].split()) > 1 else current_category
            items_to_process.append(f"{current_brand} {current_category} {parts[0]}")
            for p in parts[1:]: items_to_process.append(f"{current_brand} {local_prefix} {p}")
        else:
            final_query = f"{current_brand} {current_category} {clean_keyword}"
            items_to_process.append(final_query.strip())
            
        # F. EKSEKUSI SEARCH
        final_bonus = bonus_str if bonus_str else global_bonus
        
        for query in items_to_process:
            nama, score, merk, kode = search_sku(query, brand_filter=current_brand)
            results.append({
                "Kode Barang": kode,
                "Nama Barang": nama,
                "Qty": qty_str,
                "Bonus": final_bonus,
                "Brand Lock": current_brand if current_brand else "-",
                "Input Asli": query,
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
    if st.button("🔄 Refresh Database"):
        st.cache_data.clear()
        st.rerun()
    process_btn = st.button("🚀 PROSES DATA", type="primary")

with col2:
    st.subheader("📊 Hasil Analisa Faktur")
    
    if process_btn and raw_text:
        store_name, data = parse_po_complex(raw_text)
        
        st.success(f"🏪 **Nama Toko:** {store_name}")
        
        if data:
            df_res = pd.DataFrame(data)
            
            st.data_editor(
                df_res[["Kode Barang", "Nama Barang", "Qty", "Bonus", "Brand Lock", "Akurasi"]],
                column_config={
                    "Akurasi": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
                    "Kode Barang": st.column_config.TextColumn("KODE SKU", width="medium"),
                    "Nama Barang": st.column_config.TextColumn("Nama Barang", width="large")
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
            
            # COPY PASTE AREA
            st.markdown("### 📋 Copy Text")
            copy_text = f"Toko: {store_name}\n"
            for item in data:
                bns = f"({item['Bonus']})" if item['Bonus'] else ""
                copy_text += f"{item['Kode Barang']} | {item['Nama Barang']} | {item['Qty']} {bns}\n"
            st.text_area("Hasil:", value=copy_text, height=200)

        else:
            st.warning("Tidak ada item yang terdeteksi.")
