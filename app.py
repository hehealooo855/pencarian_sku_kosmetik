import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Fakturis Ultimate", page_icon="💎", layout="wide")
st.title("💎 AI Fakturis Ultimate (Precision Mode)")
st.markdown("Fitur: **Anti-Clash (Zaitun vs Kemiri)**, **Volume Enforcer (150ml)**, & **Footer Bonus Scan**.")

# ==========================================
# 1. KAMUS DATA & MAPPING
# ==========================================

# A. AUTO VARIANTS
AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
}

# B. BRAND ALIASES
BRAND_ALIASES = {
    "sekawan": "AINIE", "javinci": "JAVINCI", "thai": "THAI", 
    "syb": "SYB", "diosys": "DIOSYS", "satto": "SATTO", 
    "vlagio": "VLAGIO", "honor": "HONOR", "hanasui": "HANASUI",
    "implora": "IMPLORA", "brasov": "BRASOV", "felinz": "FELINZ",
    "y2000": "Y2000", "esene": "ESENE", "tata": "JAVINCI" # Tata biasanya masuk Javinci
}

# C. KEYWORD REPLACEMENTS
KEYWORD_REPLACEMENTS = {
    "zaitun": "olive oil", "kemiri": "candlenut", 
    "n.black": "natural black", "n black": "natural black", 
    "d.brwon": "dark brown", "d.brown": "dark brown",
    "brwon": "brown", "coffe": "coffee", "cerry": "cherry", 
    "shunsine": "sunshine", "temulawak": "temulawak", 
    "hand body": "lotion", "hb": "lotion", "bl": "body lotion",
    "hairmask": "hair mask", "minyak": "oil"
}

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
            col_map['merk']: 'Merk'
        })
        
        df = df[['Kode Barang', 'Nama Barang', 'Merk']].copy()
        df['Merk'] = df['Merk'].astype(str).str.strip().replace('nan', '')
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        df['Kode Barang'] = df['Kode Barang'].astype(str).str.strip().replace('nan', '-')
        
        df['Full_Text'] = df['Merk'] + ' ' + df['Nama Barang']
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
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{2,}', ngram_range=(1, 3)) 
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])

# ==========================================
# 4. ENGINE PENCARIAN (PRECISION LOGIC)
# ==========================================
def search_sku(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()

    # --- A. BRAND LOCK ---
    if brand_filter:
        brand_mask = df['Merk'].str.lower().str.contains(brand_filter.lower(), regex=False, na=False).to_numpy()
        final_scores = final_scores * brand_mask

    # --- B. VOLUME ENFORCER (150ml vs 100ml) ---
    # Cari pola angka+ml di query (misal: "150ml")
    vol_match = re.search(r'\b(\d+)\s*(ml|gr|g|liter|l)\b', query_clean)
    if vol_match:
        target_vol = vol_match.group(1) # Ambil angkanya saja, misal "150"
        unit_type = vol_match.group(2)  # Ambil unitnya, misal "ml"
        
        for idx, row in df.iterrows():
            row_txt = row['Clean_Text']
            # Jika di database TIDAK ADA angka volume yang diminta, potong skor
            if target_vol not in row_txt:
                final_scores[idx] -= 0.4 # Hukuman size salah
            
            # Hukuman Salah Satuan (ML vs GR)
            if "ml" in unit_type and re.search(r'\b\d+\s*gr\b', row_txt): final_scores[idx] = 0.0
            if "gr" in unit_type and re.search(r'\b\d+\s*ml\b', row_txt): final_scores[idx] = 0.0

    # --- C. ANTI-CLASH (Zaitun vs Kemiri) ---
    # Mencegah barang tertukar karena kemiripan kata lain
    if "olive" in query_clean or "zaitun" in query_clean:
        for idx, row in df.iterrows():
            if "candlenut" in row['Clean_Text'] or "kemiri" in row['Clean_Text']:
                final_scores[idx] = 0.0 # MATIKAN KEMIRI

    if "candlenut" in query_clean or "kemiri" in query_clean:
        for idx, row in df.iterrows():
            if "olive" in row['Clean_Text'] or "zaitun" in row['Clean_Text']:
                final_scores[idx] = 0.0 # MATIKAN ZAITUN

    # --- D. BOOSTING ---
    if "100ml" in query_clean:
        for idx, row in df.iterrows():
            if "100ml" in row['Clean_Text']: final_scores[idx] += 0.3
            elif "45ml" in row['Clean_Text']: final_scores[idx] -= 0.5
            
    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    threshold = 0.15
    if brand_filter: threshold = 0.05

    if best_score > threshold:
        row = df.iloc[best_idx]
        if brand_filter and brand_filter.lower() not in row['Merk'].lower():
             return "⚠️ Brand Mismatch", 0.0, "", "-"
        return row['Nama Barang'], best_score, row['Merk'], row['Kode Barang']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 5. PARSER PO (FOOTER SCANNING)
# ==========================================
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    global_header_bonus = ""       
    footer_bonus = "" # Bonus yang ditemukan di paling bawah
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    # 0. PRE-SCAN FOOTER BONUS (Cari angka seperti "12+1" yang berdiri sendiri di teks)
    # Regex cari pola angka+angka di baris yang pendek (kemungkinan footer)
    footer_match = re.search(r'(?:\n|^)\s*(\d+\s*\+\s*\d+)\s*(?:\n|$)', text)
    if footer_match:
        footer_bonus = footer_match.group(1)

    db_brands = df['Merk'].str.lower().unique().tolist() if df is not None else []
    db_brands = [str(b) for b in db_brands if len(str(b)) > 1]

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        # SKIP jika baris ini HANYA berisi bonus footer (agar tidak dianggap item)
        if re.fullmatch(r'\d+\s*\+\s*\d+', line.replace(" ","")):
            continue

        # CLEANING
        line_no_brackets = re.sub(r'\([^)]*\)', '', line) # Hapus (...)
        
        words = line_no_brackets.lower().split()
        replaced_words = []
        for w in words:
            clean_w = w.strip(",.-")
            replaced_words.append(KEYWORD_REPLACEMENTS.get(clean_w, w))
        line_processed = " ".join(replaced_words)

        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak)', line, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        # Cek bonus spesifik di baris ini
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line)
        line_bonus = bonus_match.group(1) if bonus_match else ""
        
        clean_keyword = line_processed.replace(qty_str.lower(), "").strip()
        clean_keyword = re.sub(r'[^\w\s,]', '', clean_keyword).strip()
        
        # HEADER DETECTION
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
                # Jika ada bonus di header (misal "Honor (12+1)"), simpan sebagai global header
                global_header_bonus = line_bonus if line_bonus else "" 
                continue 
            else:
                if "tambahan order" in lower_key:
                    current_brand = ""
                    current_category = ""
                    global_header_bonus = ""
                elif len(lower_key) > 3: 
                    current_category = clean_keyword 
            continue 

        # PROSES ITEM
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
            final_query = f"{current_brand} {current_category} {clean_keyword}"
            items_to_process.append(final_query.strip())
            
        # EKSEKUSI SEARCH
        # Prioritas Bonus: 1. Bonus di baris item -> 2. Bonus di Header Brand -> 3. Bonus Footer (Paling Bawah)
        final_bonus = line_bonus if line_bonus else (global_header_bonus if global_header_bonus else footer_bonus)
        
        for query in items_to_process:
            nama, score, merk, kode = search_sku(query, brand_filter=current_brand)
            results.append({
                "Kode Barang": kode,
                "Nama Barang": nama,
                "Qty": qty_str,
                "Bonus": final_bonus,
                "Brand Lock": current_brand if current_brand else "-",
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
    raw_text = st.text_area("Paste Chat di sini:", height=500, placeholder="SJJ Petumbukan...")
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
            
            st.markdown("### 📋 Copy Text")
            copy_text = f"Toko: {store_name}\n"
            for item in data:
                bns = f"({item['Bonus']})" if item['Bonus'] else ""
                copy_text += f"{item['Kode Barang']} | {item['Nama Barang']} | {item['Qty']} {bns}\n"
            st.text_area("Hasil:", value=copy_text, height=200)

        else:
            st.warning("Tidak ada item yang terdeteksi.")
