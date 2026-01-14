import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import SequenceMatcher
import io

# ==========================================
# 1. KONFIGURASI SISTEM
# ==========================================
st.set_page_config(page_title="AI Fakturis Ultimate", page_icon="💎", layout="wide")

# --- DATABASE KAMUS (THE BRAIN) ---
# Kategori yang dikenal sistem untuk Auto-Expand & Sanitizing
KNOWN_CATEGORIES = {
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry", "Tea Tree"],
    "peeling gel": ["Aloe", "Charcoal", "Milk", "Snail", "Lemon"],
    "peeling": ["Aloe", "Charcoal", "Milk", "Snail", "Lemon"],
    "toner badan": ["Red Jelly", "Coklat", "Fresh Skin", "Kelupas", "Ginseng"],
    "toner": ["Rose", "Chamomile", "Aloe", "Lemon"],
    "masker wajah": ["Bengkoang", "Aloe Vera", "Cucumber", "Avocado"],
    "cushion": ["01", "02", "03", "04", "Ivory", "Natural", "Beige"],
    "blush": ["01", "02", "03", "04"],
    "lip matte": ["01", "02", "03", "04", "05"],
    "yu chun mei": ["Night Cream", "Day Cream", "Cleanser", "Serum"],
    "body serum": ["Aha", "Gluta", "Hitam", "Kuning"],
    "hair mask": ["Ginseng", "Strawberry", "Aloe", "Avocado", "Kemiri", "Cocoa", "Choco", "Milky"]
}

# Alias Brand (Nama di Chat -> Nama di Database)
BRAND_MAP = {
    "sekawan": "AINIE", "ainie": "AINIE",
    "javinci": "JAVINCI", "thai": "THAI", 
    "syb": "SYB", "diosys": "DIOSYS", "satto": "SATTO", 
    "vlagio": "VLAGIO", "honor": "HONOR", "hanasui": "HANASUI",
    "implora": "IMPLORA", "brasov": "BRASOV", "tata": "JAVINCI",
    "body white": "JAVINCI", "holly": "HOLLY",
    "yu chun mei": "YU CHUN MEI", "ycm": "YU CHUN MEI",
    "whitelab": "WHITELAB", "bonavie": "BONAVIE", "goute": "GOUTE",
    "kim": "KIM", "kim kosmetik": "KIM", "newlab": "NEWLAB", "autumn": "JAVINCI"
}

# Kamus Perbaikan Kata (Typo -> Baku)
KEYWORD_FIX = {
    # Grammar & Typo
    "cream night": "night cream", "krim malam": "night cream", "malam": "night cream",
    "cream day": "day cream", "krim siang": "day cream", "siang": "day cream",
    "cleanser": "cleanser", "sabun muka": "cleanser", "facial wash": "cleanser",
    "pepaya": "papaya", "bengkuang": "bengkoang",
    "barsoap": "soap", "bar soap": "soap", "sabun": "soap",
    "minyak": "oil", "zaitun": "olive oil",
    "trii": "tree", "pappermint": "peppermint",
    "creme": "cream", "canele": "canale",
    "goublush": "blush", "goublus": "blush",
    "n.black": "natural black", "d.brown": "dark brown",
    
    # Satuan & Bentuk
    "bulat": "150ml", "gepeng": "100ml",
    "50g": "50gr", "50gram": "50gr",
    "all": "semua varian", "campur": "semua varian",
    
    # Surgical Fixes (Kasus Spesifik)
    "manjakani 100ml": "manjakani 110ml", 
    "manjakani 100": "manjakani 110ml"
}

# Aturan Konflik (Jika ada A, jangan pilih B)
CONFLICT_RULES = {
    "olive": ["candlenut", "kemiri", "urang"],
    "zaitun": ["candlenut", "kemiri", "urang"],
    "kemiri": ["olive", "zaitun", "urang"],
    "candlenut": ["olive", "zaitun"],
    "cream": ["serum", "toner", "cleanser", "soap"], 
    "krim": ["serum", "toner", "cleanser", "soap"],
    "serum": ["cream", "night", "day", "krim", "cleanser"],
    "night": ["day", "siang"], "day": ["night", "malam"]
}

# Kata Kunci Wajib (Jika user minta, DB harus ada)
REQUIRED_KEYWORDS = ["banded", "bonus", "free", "gratis"]

# ==========================================
# 2. CORE ENGINE (LOADING & PROCESSING)
# ==========================================
@st.cache_data(ttl=3600)
def load_database():
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    try:
        df_raw = pd.read_csv(sheet_url, header=None)
        # Cari baris header otomatis
        header_idx = -1
        for i, row in df_raw.iterrows():
            if any("kode barang" in str(x).lower() for x in row.tolist()):
                header_idx = i; break
        
        if header_idx == -1: return None

        df = pd.read_csv(sheet_url, header=header_idx)
        df.columns = df.columns.str.strip()
        
        # Mapping Kolom Fleksibel
        col_map = {}
        for col in df.columns:
            c_low = col.lower()
            if "kode" in c_low and "barang" in c_low: col_map['kode'] = col
            if "nama" in c_low and "barang" in c_low: col_map['nama'] = col
            if "merek" in c_low or "merk" in c_low: col_map['merk'] = col

        if len(col_map) < 3: return None
        
        df = df.rename(columns={col_map['kode']: 'Kode', col_map['nama']: 'Nama', col_map['merk']: 'Merk'})
        df = df[['Kode', 'Nama', 'Merk']].dropna(subset=['Nama'])
        
        # CLEANING DATABASE (Standardisasi Awal)
        df['Clean_Text'] = df['Nama'] + " " + df['Merk']
        df['Clean_Text'] = df['Clean_Text'].astype(str).str.lower()
        # Unit Splitter (50g -> 50 g)
        df['Clean_Text'] = df['Clean_Text'].apply(lambda x: re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', x))
        df['Clean_Text'] = df['Clean_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', x))
        
        return df
    except Exception as e: st.error(f"Database Error: {e}"); return None

df_db = load_database()

# --- AI TRAINING ---
@st.cache_resource
def train_ai(data):
    if data is None or data.empty: return None, None
    # N-Gram 1-3 untuk menangkap frasa seperti "tea tree" atau "night cream"
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3)) 
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df_db is not None:
    ai_vectorizer, ai_matrix = train_ai(df_db['Clean_Text'])

# ==========================================
# 3. FUNGSI PENCARIAN CERDAS (HYBRID)
# ==========================================
def extract_numbers(text):
    return re.findall(r'(\d+)', text)

def smart_search(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    # 1. Normalisasi Query
    query_clean = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', query.lower())
    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query_clean).strip()
    
    # 2. Injeksi Sinonim
    search_query = query_clean
    for k, v in KEYWORD_FIX.items():
        if k in search_query: search_query = search_query.replace(k, v) # Direct replace lebih aman
    
    # 3. Zaitun Injection (Khusus)
    if "zaitun" in search_query and "olive" not in search_query:
        search_query += " olive oil"

    # 4. TF-IDF Filtering (Ambil Top 50 Kandidat)
    query_vec = ai_vectorizer.transform([search_query])
    cosine_scores = cosine_similarity(query_vec, ai_matrix).flatten()
    top_indices = cosine_scores.argsort()[-50:][::-1]
    
    candidates = df_db.iloc[top_indices].copy()
    candidates['tfidf_score'] = cosine_scores[top_indices]
    
    best_match = None
    best_score = -100
    user_nums = extract_numbers(query_clean)
    
    # 5. Reranking dengan Logika Bisnis
    for idx, row in candidates.iterrows():
        db_text = row['Clean_Text']
        db_brand = str(row['Merk']).lower()
        
        # Base Score: Kombinasi TF-IDF (Konteks) + Fuzzy (Typo)
        fuzzy_ratio = SequenceMatcher(None, search_query, db_text).ratio()
        final_score = (row['tfidf_score'] * 0.4) + (fuzzy_ratio * 0.6)
        
        # --- RULE 1: BRAND LOCK ---
        if brand_filter and brand_filter.lower() not in db_brand:
            continue # Langsung skip jika brand salah
            
        # --- RULE 2: VOLUME CHECK (CRITICAL) ---
        num_mismatch = False
        for num in user_nums:
            if int(num) > 20: # Hanya cek angka besar (Volume/Gramasi)
                # Regex boundary check (\b50\b match 50, tapi tidak 500)
                if not re.search(r'\b' + num + r'\b', db_text):
                    num_mismatch = True
        if num_mismatch: final_score -= 0.6 # Hukuman berat untuk salah volume

        # --- RULE 3: ANTI-CLASH ---
        clash = False
        for key, enemies in CONFLICT_RULES.items():
            if key in search_query: 
                for enemy in enemies:
                    if enemy in db_text:
                        clash = True; break
        if clash: continue # Skip item yang konflik

        # --- RULE 4: REQUIRED KEYWORDS (Banded) ---
        for kw in REQUIRED_KEYWORDS:
            if kw in search_query and kw not in db_text:
                final_score -= 0.5 
            if kw not in search_query and kw in db_text:
                final_score -= 0.1 

        # --- RULE 5: COLOR CHECK ---
        colors = ["hitam", "kuning", "putih", "ungu", "gold", "pink", "blue", "merah"]
        for c in colors:
            if c in search_query and c not in db_text:
                # Jika user minta warna A, tapi DB punya warna B -> Salah
                has_other_color = any(oc in db_text for oc in colors if oc != c)
                if has_other_color: final_score -= 0.5 

        if final_score > best_score:
            best_score = final_score
            best_match = row

    # Threshold dinamis
    threshold = 0.35 if not brand_filter else 0.20
    
    if best_match is not None and best_score > threshold:
        return best_match['Nama'], round(best_score, 2), best_match['Merk'], best_match['Kode']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 4. PARSER PO UTAMA (THE MONSTER PARSER)
# ==========================================
def sanitize_category(text):
    """Membersihkan judul kategori yang kotor (misal: 'cushion satins airy...' -> 'cushion')"""
    text_lower = text.lower()
    for cat_key in KNOWN_CATEGORIES.keys():
        if cat_key in text_lower:
            return cat_key # Kembalikan kategori murni
    return text # Kalau tidak dikenal, kembalikan aslinya

def parse_po_sales(text):
    lines = text.split('\n')
    results = []
    
    curr_brand = ""
    curr_cat = ""
    footer_bonus = ""
    header_bonus = ""
    
    # 1. Pre-Scan Footer Bonus
    for line in reversed(lines):
        if re.fullmatch(r'\s*\d+\s*[\+\*]\s*\d+\s*', line):
            footer_bonus = line.strip(); break

    # Siapkan list brand dari DB untuk deteksi otomatis
    db_brands_list = [str(b).lower() for b in df_db['Merk'].unique() if len(str(b)) > 2]

    for line in lines:
        line = line.strip()
        if not line or line == "-" or line == footer_bonus: continue
        
        # --- A. SUPER CLEANING ---
        # 1. Hapus bullet points
        line_clean = re.sub(r'^[\-\.\s]+', '', line) 
        # 2. Hapus noise transaksi
        line_clean = re.sub(r'\b(cash|tunai|kredit|tempo)\b', '', line_clean, flags=re.IGNORECASE)
        # 3. Hapus isi kurung
        line_clean = re.sub(r'[\(\)]', ' ', line_clean)
        # 4. FIX COLON (:12 -> x 12)
        line_clean = re.sub(r':\s*(\d+)', r' x \1', line_clean)
        
        # --- B. DETEKSI ITEM ---
        # Pola Regex: Mencari angka kuantitas
        qty_pattern = r'(?:^|\s)(?:x|@|@x)?\s*(\d+)\s*(?:x|pcs|pc|lsn|box|ktk|pak|btl)?(?:\s*[\+\*]\s*\d+)?(?:$|\s)'
        qty_match = re.findall(qty_pattern, line_clean, re.IGNORECASE)
        
        qty_str = ""
        is_item = False
        
        if qty_match:
            valid_nums = [m for m in qty_match if m.strip()]
            if valid_nums:
                qty_str = valid_nums[0]
                is_item = True
        
        # Cek Bonus (12+1)
        line_bonus_match = re.search(r'\d+\s*[\+\*]\s*\d+', line_clean)
        line_bonus = line_bonus_match.group(0) if line_bonus_match else ""
        if line_bonus: is_item = True

        # Bersihkan nama barang dari angka qty
        clean_name = re.sub(qty_pattern, ' ', line_clean, flags=re.IGNORECASE)
        clean_name = re.sub(r'\d+\s*[\+\*]\s*\d+', ' ', clean_name) # Hapus bonus pattern
        clean_name = re.sub(r'[^\w\s]', ' ', clean_name).strip() # Hapus simbol sisa
        
        # Translate kata per kata
        words = clean_name.lower().split()
        final_words = [KEYWORD_FIX.get(w, w) for w in words]
        clean_name = " ".join(final_words)

        # --- C. LOGIKA HEADER / ITEM ---
        if not is_item:
            # Kemungkinan ini adalah HEADER (Brand atau Kategori)
            lower_name = clean_name.lower()
            detected_brand = None
            
            # Cek Alias Brand
            for alias, real in BRAND_MAP.items():
                if lower_name == alias or lower_name.startswith(alias + " "):
                    detected_brand = real
                    clean_name = lower_name.replace(alias, "").strip()
                    break
            
            # Cek Database Brand
            if not detected_brand:
                for db_b in db_brands_list:
                    if db_b in lower_name: 
                        detected_brand = next((v for k,v in BRAND_MAP.items() if k == db_b), db_b.upper())
                        clean_name = lower_name.replace(db_b, "").strip()
                        break
            
            if detected_brand:
                curr_brand = detected_brand
                # Sanitasi kategori sisa (misal: "Goute Cushion Satins..." -> "Cushion")
                curr_cat = sanitize_category(clean_name)
                header_bonus = line_bonus
                continue
            else:
                # Bukan brand, mungkin Sub-Kategori (misal: "Powder Mask")
                if len(clean_name) > 2:
                    curr_cat = sanitize_category(clean_name)
            continue

        # --- D. LOGIKA AUTO-EXPAND ("ALL VARIAN") ---
        items_to_process = []
        
        if "semua varian" in clean_name.lower():
            found_collection = False
            # Cek gabungan kategori (misal: "Powder Mask")
            check_str = f"{curr_cat} {clean_name}".lower()
            
            # Coba cari di kamus kategori
            effective_cat = curr_cat 
            for key, variants in KNOWN_CATEGORIES.items():
                # Jika nama item sendiri mengandung kategori (misal: "Peeling Gel All")
                if key in clean_name.lower():
                    effective_cat = key 
                    for v in variants: items_to_process.append(f"{curr_brand} {effective_cat} {v}")
                    found_collection = True; break
                # Jika kategori induk cocok
                elif key in check_str:
                    for v in variants: items_to_process.append(f"{curr_brand} {key} {v}")
                    found_collection = True; break
            
            if not found_collection:
                # Fallback: Cari apa adanya
                items_to_process.append(f"{curr_brand} {curr_cat} {clean_name}")
        else:
            # Item biasa (Single)
            items_to_process.append(f"{curr_brand} {curr_cat} {clean_name}")

        # --- E. EKSEKUSI PENCARIAN ---
        active_bonus = line_bonus if line_bonus else (header_bonus if header_bonus else footer_bonus)
        
        for q in items_to_process:
            # Bersihkan spasi ganda hasil penggabungan
            q = re.sub(r'\s+', ' ', q).strip()
            
            res_name, res_score, res_brand, res_code = smart_search(q, brand_filter=curr_brand)
            results.append({
                "Kode": res_code,
                "Nama Barang": res_name,
                "Qty": qty_str if qty_str else "1",
                "Bonus": active_bonus,
                "Akurasi": res_score,
                "Input Asli": q
            })

    return lines[0] if lines else "Unknown", results

# ==========================================
# 5. USER INTERFACE (CLEAN & PROFESSIONAL)
# ==========================================
st.sidebar.title("⚙️ Kontrol Panel")
if st.sidebar.button("🔄 Reset / Refresh Data"):
    st.cache_data.clear()
    st.rerun()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Input PO (Chat Sales)")
    raw_text = st.text_area("Paste Pesanan Disini:", height=450, placeholder="Contoh:\nKim Kosmetik\nWhitelab\nSerum:12pcs...")
    if st.button("🚀 PROSES SEKARANG", type="primary", use_container_width=True):
        if raw_text:
            with st.spinner("Sedang membedah pesanan..."):
                store, res = parse_po_sales(raw_text)
                st.session_state['results'] = (store, res)

with col2:
    st.subheader("📊 Hasil Identifikasi")
    if 'results' in st.session_state:
        store, data = st.session_state['results']
        if data:
            st.success(f"Customer Terdeteksi: **{store}**")
            df_res = pd.DataFrame(data)
            
            # Tampilan Tabel Interaktif
            st.dataframe(
                df_res[["Kode", "Nama Barang", "Qty", "Bonus", "Akurasi"]],
                column_config={
                    "Akurasi": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                    "Kode": st.column_config.TextColumn("Kode SKU"),
                    "Nama Barang": st.column_config.TextColumn("Nama Database", width="large")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Area Copy-Paste
            st.markdown("### 📋 Siap Copy")
            txt_output = f"Customer: {store}\n"
            for item in data:
                b_txt = f"({item['Bonus']})" if item['Bonus'] else ""
                txt_output += f"{item['Kode']} | {item['Nama Barang']} | {item['Qty']} {b_txt}\n"
            
            st.text_area("Hasil Teks:", value=txt_output, height=150, label_visibility="collapsed")
            
            # Fitur Download Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, index=False, sheet_name='PO')
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f"PO_{store.replace(' ', '_')}.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
            
        else:
            st.warning("⚠️ Tidak ada item yang terdeteksi. Pastikan format ada angkanya.")
