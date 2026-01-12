import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Fakturis Pro", page_icon="🧬", layout="wide")
st.title("🧬 AI Fakturis Pro (Final Correction)")
st.markdown("Fitur: **Smart Number Reader**, **Bahasa Indonesia Native (No Oil translation)**, & **Brand Lock**.")

# ==========================================
# 1. KAMUS DATA & MAPPING
# ==========================================

AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
}

BRAND_ALIASES = {
    "sekawan": "AINIE", "javinci": "JAVINCI", "thai": "THAI", 
    "syb": "SYB", "diosys": "DIOSYS", "satto": "SATTO", 
    "vlagio": "VLAGIO", "honor": "HONOR", "hanasui": "HANASUI",
    "implora": "IMPLORA", "brasov": "BRASOV", "tata": "JAVINCI",
    "body white": "JAVINCI"
}

# --- PERBAIKAN DI SINI: HAPUS TRANSLASI YANG MERUSAK ---
KEYWORD_REPLACEMENTS = {
    # 1. Terjemahan Bentuk Botol Tata
    "bulat": "150ml", "botol bulat": "150ml",
    "gepeng": "100ml", "botol gepeng": "100ml",
    
    # 2. Typo & Singkatan
    "kemiri": "candlenut", # Candlenut biasanya aman karena jarang dipakai
    "n.black": "natural black", "n black": "natural black", 
    "d.brwon": "dark brown", "d.brown": "dark brown",
    "brwon": "brown", "coffe": "coffee", "cerry": "cherry", 
    "temulawak": "temulawak", "hand body": "lotion", "hb": "lotion",
    "hairmask": "hair mask"
    # SAYA HAPUS "minyak": "oil" AGAR SESUAI DATABASE INDONESIA
}

# Daftar Konflik (Anti-Clash)
CONFLICT_MAP = {
    "olive": ["candlenut", "kemiri"],
    "zaitun": ["candlenut", "kemiri"],
    "candlenut": ["olive", "zaitun"],
    "kemiri": ["olive", "zaitun"],
}

# Keyword Wajib
ESSENTIAL_KEYWORDS = ["banded", "bonus", "free", "gratis"]

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    try:
        df_raw = pd.read_csv(sheet_url, header=None)
        header_idx = -1
        for i, row in df_raw.iterrows():
            if any("kode barang" in str(x).lower() for x in row.tolist()):
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
            col_map['merk']: 'Merk'
        })
        
        df = df[['Kode Barang', 'Nama Barang', 'Merk']].copy()
        df['Merk'] = df['Merk'].astype(str).str.strip().replace('nan', '')
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        df['Kode Barang'] = df['Kode Barang'].astype(str).str.strip().replace('nan', '-')
        
        df['Full_Text'] = df['Merk'] + ' ' + df['Nama Barang']
        # Pembersihan teks dasar
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df

    except Exception: return None

df = load_data()

# ==========================================
# 3. TRAIN AI
# ==========================================
@st.cache_resource
def train_model(data):
    if data is None or data.empty: return None, None
    # Token pattern diperbaiki agar bisa membaca kata pendek
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 3)) 
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])

# ==========================================
# 4. ENGINE PENCARIAN
# ==========================================
def extract_numbers_robust(text):
    # Regex ini bisa mengambil "100" dari "100ml"
    return re.findall(r'(\d+)', text)

def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())

    # --- SYNONYM INJECTION ---
    search_query = query_clean
    if "zaitun" in search_query and "olive" not in search_query:
        search_query += " olive oil" 
    
    # AI Cari Kandidat
    query_vec = tfidf_vectorizer.transform([search_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-20:][::-1]
    
    best_candidate = None
    best_score = -10.0
    
    # Ambil angka dari query ASLI (sebelum spasi dibersihkan regex)
    # Agar "100ml" terbaca angka 100-nya
    query_numbers = extract_numbers_robust(query.lower())
    
    for idx in top_indices:
        current_score = similarity_scores[idx]
        if current_score < 0.1: continue 
        
        row = df.iloc[idx]
        db_text = row['Clean_Text']
        db_brand = row['Merk'].lower()
        
        # 1. BRAND LOCK
        if brand_filter and brand_filter.lower() not in db_brand:
            continue
        
        # 2. FILTER ANGKA (SMART VOLUME)
        # Jika user minta "100", di database harus ada angka "100" (baik terpisah atau nempel)
        valid_number = True
        for num in query_numbers:
            if int(num) > 20: 
                # Cek keberadaan angka di teks database
                if num not in db_text: 
                    valid_number = False
                    break
        if not valid_number:
            current_score -= 2.0 # Hukuman Mati (Salah Volume)

        # 3. ANTI-CLASH (Zaitun vs Kemiri)
        conflict_found = False
        for key, enemies in CONFLICT_MAP.items():
            if key in query_clean:
                for enemy in enemies:
                    if enemy in db_text:
                        conflict_found = True; break
        if conflict_found: continue

        # 4. BANDED LOGIC
        for kw in ESSENTIAL_KEYWORDS:
            if kw in query_clean and kw not in db_text:
                current_score -= 2.0
            if kw not in query_clean and kw in db_text:
                current_score -= 0.5 
        
        # 5. VARIANT GUARD (COLLAGEN PENALTY)
        sensitive_keywords = ["collagen", "booster", "serum", "acne", "brightening"]
        for kw in sensitive_keywords:
            if kw in db_text and kw not in query_clean:
                current_score -= 1.0 # Hukuman diperberat (-1.0)
            if kw in db_text and kw in query_clean:
                current_score += 0.5 

        if current_score > best_score:
            best_score = current_score
            best_candidate = row

    # Turunkan threshold sedikit agar barang yang kena penalti ringan masih bisa muncul
    if best_candidate is not None and best_score > 0.05:
        return best_candidate['Nama Barang'], best_score, best_candidate['Merk'], best_candidate['Kode Barang']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 5. PARSER PO
# ==========================================
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    footer_bonus = ""
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    for line in reversed(lines):
        if re.fullmatch(r'\s*\d+\s*\+\s*\d+\s*', line):
            footer_bonus = line.strip()
            break

    db_brands = df['Merk'].str.lower().unique().tolist() if df is not None else []
    db_brands = [str(b) for b in db_brands if len(str(b)) > 1]

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        if line == footer_bonus: continue 

        line_clean = re.sub(r'\([^)]*\)', '', line)
        
        words = line_clean.lower().split()
        replaced_words = []
        for w in words:
            clean_w = w.strip(",.-")
            if clean_w in KEYWORD_REPLACEMENTS:
                replaced_words.append(KEYWORD_REPLACEMENTS[clean_w])
            else:
                replaced_words.append(w)
        
        line_processed = " ".join(replaced_words)
        
        # HANDLE FRASA BENTUK
        line_processed = line_processed.replace("botol bulat", "150ml").replace("bulat", "150ml")
        line_processed = line_processed.replace("botol gepeng", "100ml").replace("gepeng", "100ml")

        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak)', line, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line)
        line_bonus = bonus_match.group(1) if bonus_match else ""
        
        clean_keyword = line_processed.replace(qty_str.lower(), "").strip()
        clean_keyword = re.sub(r'[^\w\s]', '', clean_keyword).strip()
        
        is_item = bool(qty_match)
        if not is_item:
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
                    if re.search(r'\b' + re.escape(brand) + r'\b', lower_key):
                        detected_alias = brand
                        context_suffix = lower_key.replace(brand, "").strip()
                        break
            if detected_alias:
                current_brand = detected_alias 
                current_category = context_suffix 
                continue 
            else:
                if len(lower_key) > 3: current_category = clean_keyword 
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
            items_to_process.append(f"{current_brand} {current_category} {parts[0]}")
            for p in parts[1:]: 
                 local_prefix = parts[0].split()[0] if parts[0] else ""
                 items_to_process.append(f"{current_brand} {local_prefix} {p}")
        else:
            final_query = f"{current_brand} {current_category} {clean_keyword}"
            items_to_process.append(final_query.strip())
            
        final_bonus = line_bonus if line_bonus else footer_bonus
        
        for query in items_to_process:
            nama, score, merk, kode = search_sku(query, brand_filter=current_brand)
            results.append({
                "Kode Barang": kode,
                "Nama Barang": nama,
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
    st.subheader("📝 Input PO")
    raw_text = st.text_area("Paste Chat:", height=500, placeholder="SJJ Petumbukan...")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    process_btn = st.button("🚀 PROSES", type="primary")

with col2:
    st.subheader("📊 Hasil")
    if process_btn and raw_text:
        store_name, data = parse_po_complex(raw_text)
        if data:
            st.success(f"Toko: **{store_name}**")
            df_res = pd.DataFrame(data)
            st.data_editor(
                df_res[["Kode Barang", "Nama Barang", "Qty", "Bonus", "Akurasi"]],
                column_config={
                    "Akurasi": st.column_config.ProgressColumn("Conf", format="%.2f", min_value=0, max_value=1),
                    "Kode Barang": st.column_config.TextColumn("KODE", width="medium"),
                    "Nama Barang": st.column_config.TextColumn("NAMA", width="large")
                }, hide_index=True, use_container_width=True, height=600
            )
            st.markdown("### 📋 Copy")
            copy_text = f"Toko: {store_name}\n"
            for item in data:
                bns = f"({item['Bonus']})" if item['Bonus'] else ""
                copy_text += f"{item['Kode Barang']} | {item['Nama Barang']} | {item['Qty']} {bns}\n"
            st.text_area("Hasil Teks:", value=copy_text, height=200)
        else:
            st.warning("Data tidak ditemukan.")
