import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz, utils
import re

# ==========================================
# 0. KONFIGURASI & SETUP
# ==========================================
st.set_page_config(page_title="AI Fakturis Enterprise", page_icon="🏢", layout="wide")
st.title("🏢 AI Fakturis Enterprise (Hybrid Engine)")
st.markdown("Teknologi: **TF-IDF (Filter)** + **RapidFuzz (Selection)** + **Context Aware**.")

# ==========================================
# 1. KAMUS DATA (Hanya yang krusial)
# ==========================================
# Kita buang kamus typo kecil (cerry, brwon) karena Fuzzy Logic akan menanganinya otomatis!

AUTO_VARIANTS = {
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry", "Tea Tree"],
    "peeling gel": ["Aloe", "Charcoal", "Milk", "Snail", "Lemon"],
    "toner badan": ["Red Jelly", "Coklat", "Fresh Skin", "Kelupas"],
    "masker wajah": ["Bengkoang", "Aloe Vera", "Cucumber", "Avocado"]
}

BRAND_ALIASES = {
    "sekawan": "AINIE", "ainie": "AINIE",
    "javinci": "JAVINCI", "thai": "THAI", 
    "syb": "SYB", "diosys": "DIOSYS", "satto": "SATTO", 
    "vlagio": "VLAGIO", "honor": "HONOR", "hanasui": "HANASUI",
    "implora": "IMPLORA", "brasov": "BRASOV", "tata": "JAVINCI",
    "body white": "JAVINCI", "holly": "HOLLY"
}

# Kamus Konsep/Sinonim (Bukan Typo)
CONCEPT_MAP = {
    "bulat": "150ml", "gepeng": "100ml",
    "all": "semua varian", "campur": "semua varian",
    "manjakani 100ml": "manjakani 110ml", "manjakani 100": "manjakani 110ml",
    "pepaya": "papaya", "bengkuang": "bengkoang",
    "barsoap": "soap", "sabun": "soap",
    "minyak": "oil", "zaitun": "olive oil" # Kita kembalikan ini untuk membantu Fuzzy
}

# Konflik Mutlak (Jika ada A, tidak boleh ada B)
HARD_CONFLICTS = {
    "olive": ["candlenut", "kemiri", "urang"],
    "zaitun": ["candlenut", "kemiri", "urang"],
    "kemiri": ["olive", "zaitun", "urang"],
    "candlenut": ["olive", "zaitun"],
}

ESSENTIAL_KEYWORDS = ["banded", "bonus", "free"]

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
                header_idx = i; break
        if header_idx == -1: return None

        df = pd.read_csv(sheet_url, header=header_idx)
        df.columns = df.columns.str.strip()
        
        # Mapping Kolom Otomatis
        col_map = {}
        for col in df.columns:
            c_low = col.lower()
            if "kode" in c_low and "barang" in c_low: col_map['kode'] = col
            if "nama" in c_low and "barang" in c_low: col_map['nama'] = col
            if "merek" in c_low or "merk" in c_low: col_map['merk'] = col

        if len(col_map) < 3: return None
        
        df = df.rename(columns={col_map['kode']: 'Kode', col_map['nama']: 'Nama', col_map['merk']: 'Merk'})
        df = df[['Kode', 'Nama', 'Merk']].dropna(subset=['Nama'])
        
        # CLEANING DATABASE (Standardisasi)
        # Pisahkan angka dari huruf (50g -> 50 g, 100ml -> 100 ml)
        df['Clean_Text'] = df['Nama'] + " " + df['Merk']
        df['Clean_Text'] = df['Clean_Text'].astype(str).str.lower()
        df['Clean_Text'] = df['Clean_Text'].apply(lambda x: re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', x))
        df['Clean_Text'] = df['Clean_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', x))
        
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df = load_data()

# ==========================================
# 3. HYBRID ENGINE (TF-IDF + FUZZY)
# ==========================================
@st.cache_resource
def train_tfidf(data):
    if data is None or data.empty: return None, None
    # Ngram 1-3 menangkap "tea tree", "olive oil" sebagai satu kesatuan
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3)) 
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_tfidf(df['Clean_Text'])

def extract_numbers(text):
    return re.findall(r'(\d+)', text)

def search_hybrid(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    # 1. PRE-PROCESS QUERY
    # Pisahkan angka nempel (50g -> 50 g)
    query_clean = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', query.lower())
    # Bersihkan simbol
    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query_clean).strip()
    
    # Injeksi Sinonim (Tanpa menghapus aslinya)
    search_query = query_clean
    for k, v in CONCEPT_MAP.items():
        if k in search_query: search_query += f" {v}"

    # 2. STEP 1: TF-IDF (Penyaringan Kasar)
    # Ambil 50 kandidat teratas berdasarkan kata yang sama
    query_vec = tfidf_vectorizer.transform([search_query])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Ambil index top 50
    top_indices = cosine_scores.argsort()[-50:][::-1]
    
    candidates = df.iloc[top_indices].copy()
    # Tambahkan skor TF-IDF ke dataframe sementara
    candidates['tfidf_score'] = cosine_scores[top_indices]
    
    # 3. STEP 2: LOGIC FILTER & SCORING
    best_match = None
    best_score = -100
    
    user_nums = extract_numbers(query_clean)
    
    for idx, row in candidates.iterrows():
        db_text = row['Clean_Text']
        db_brand = str(row['Merk']).lower()
        
        # Base Score dari Fuzzy Match (WRatio sangat pintar menangani singkatan/typo)
        # Kita bandingkan query asli user (yang bersih) dengan database
        fuzzy_score = fuzz.WRatio(query_clean, db_text) / 100.0
        
        # Gabungkan skor TF-IDF dan Fuzzy
        final_score = (row['tfidf_score'] * 0.4) + (fuzzy_score * 0.6)
        
        # --- PENALTI DAN BONUS (THE BRAIN) ---
        
        # A. BRAND LOCK (Wajib)
        if brand_filter and brand_filter.lower() not in db_brand:
            continue # Skip beda brand
            
        # B. CEK ANGKA (CRITICAL)
        # Jika user tulis "100", DB harus ada "100". 
        # Pengecualian: User tidak tulis angka (aman).
        num_mismatch = False
        for num in user_nums:
            if int(num) > 20: # Abaikan angka kecil (mungkin varian)
                # Cek exact word match di DB
                if not re.search(r'\b' + num + r'\b', db_text):
                    num_mismatch = True
        
        if num_mismatch: 
            final_score -= 0.5 # Hukuman berat (misal minta 100ml dikasih 50ml)

        # C. ANTI-CLASH (Musuh Bebuyutan)
        clash = False
        for key, enemies in HARD_CONFLICTS.items():
            if key in query_clean:
                for enemy in enemies:
                    if enemy in db_text:
                        clash = True; break
        if clash: continue # Langsung buang

        # D. BANDED CHECK
        for kw in ESSENTIAL_KEYWORDS:
            if kw in query_clean and kw not in db_text:
                final_score -= 0.4 # User minta banded, DB gak ada
            if kw not in query_clean and kw in db_text:
                final_score -= 0.1 # DB banded, user gak minta (masih boleh tampil tapi prioritas turun dikit)

        # E. VARIANT COLOR CHECK (Hitam vs Kuning)
        # Jika user sebut warna, pastikan DB juga punya atau DB netral.
        # Jika DB punya warna lain -> HUKUM.
        colors = ["hitam", "kuning", "putih", "ungu", "gold", "pink", "blue"]
        for c in colors:
            if c in query_clean and c not in db_text:
                # Cek apakah DB punya warna lain?
                has_other_color = any(oc in db_text for oc in colors if oc != c)
                if has_other_color:
                    final_score -= 0.5 # Salah warna

        # Simpan pemenang
        if final_score > best_score:
            best_score = final_score
            best_match = row

    # Threshold dinamis
    threshold = 0.35 if not brand_filter else 0.25
    
    if best_match is not None and best_score > threshold:
        return best_match['Nama'], round(best_score, 2), best_match['Merk'], best_match['Kode']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 4. PARSER PO (REGEX MONSTER)
# ==========================================
def parse_po(text):
    lines = text.split('\n')
    results = []
    
    curr_brand = ""
    curr_cat = ""
    footer_bonus = ""
    header_bonus = ""
    
    # 1. Cek Footer Bonus (Baris terakhir yang cuma angka)
    for line in reversed(lines):
        if re.fullmatch(r'\s*\d+\s*[\+\*]\s*\d+\s*', line):
            footer_bonus = line.strip()
            break

    db_brands = [str(b).lower() for b in df['Merk'].unique() if len(str(b)) > 2]

    for line in lines[1:]:
        line = line.strip()
        if not line or line == "-" or line == footer_bonus: continue
        
        # 2. Pre-Cleaning
        # Hapus kata transaksi
        line_clean = re.sub(r'\b(cash|tunai|kredit|tempo)\b', '', line, flags=re.IGNORECASE)
        # Hapus kurung
        line_clean = re.sub(r'[\(\)]', ' ', line_clean)
        
        # 3. DETEKSI ITEM (REGEX MONSTER)
        # Mendeteksi: 12pcs, 12 pcs, x12, 12x, @12, @x12, 12+1
        qty_pattern = r'(?:^|\s)(?:x|@|@x)?\s*(\d+)\s*(?:x|pcs|pc|lsn|box|ktk|pak|btl)?(?:\s*[\+\*]\s*\d+)?(?:$|\s)'
        qty_match = re.findall(qty_pattern, line_clean, re.IGNORECASE)
        
        qty_str = ""
        is_item = False
        
        # Ambil angka pertama yang valid sebagai Qty utama
        if qty_match:
            # Filter hasil kosong dari regex
            valid_nums = [m for m in qty_match if m.strip()]
            if valid_nums:
                qty_str = valid_nums[0] # Ambil angka pertama
                is_item = True
        
        # Cek Bonus Spesifik di baris ini (misal 12+1)
        line_bonus_match = re.search(r'\d+\s*[\+\*]\s*\d+', line_clean)
        line_bonus = line_bonus_match.group(0) if line_bonus_match else ""
        if line_bonus: is_item = True # Kalau ada bonus, pasti item

        # Bersihkan Qty dari teks untuk pencarian nama
        clean_name = re.sub(qty_pattern, ' ', line_clean, flags=re.IGNORECASE)
        # Bersihkan bonus pattern
        clean_name = re.sub(r'\d+\s*[\+\*]\s*\d+', ' ', clean_name)
        
        # Normalisasi spasi dan simbol
        clean_name = re.sub(r'[^\w\s]', ' ', clean_name).strip()
        # Normalisasi kata kunci (Translate)
        words = clean_name.lower().split()
        final_words = []
        for w in words:
            final_words.append(KEYWORD_REPLACEMENTS.get(w, w))
        clean_name = " ".join(final_words)

        # 4. LOGIKA HIERARKI (Header vs Item)
        if not is_item:
            # Cek apakah ini Brand Header?
            lower_name = clean_name.lower()
            detected_brand = None
            
            # Cek Alias
            for alias, real in BRAND_ALIASES.items():
                if lower_name == alias or lower_name.startswith(alias + " "):
                    detected_brand = real
                    clean_name = lower_name.replace(alias, "").strip() # Sisa teks jadi kategori
                    break
            
            # Cek DB
            if not detected_brand:
                for db_b in db_brands:
                    if db_b in lower_name: # Partial match allowed for header
                        detected_brand = next((v for k,v in BRAND_ALIASES.items() if k == db_b), db_b.upper())
                        clean_name = lower_name.replace(db_b, "").strip()
                        break
            
            if detected_brand:
                curr_brand = detected_brand
                curr_cat = clean_name # Sisa teks header jadi kategori (misal: "Honor (12+1)" -> Brand: Honor, Cat: "")
                header_bonus = line_bonus
                continue
            else:
                # Bukan brand, mungkin sub-kategori (misal "Powder Mask")
                if len(clean_name) > 2:
                    curr_cat = clean_name
            continue

        # 5. CONSTRUCT QUERY
        # Gabungkan: Brand + Kategori + Nama Item
        # Misal: SYB + Powder Mask + Tea Tree
        
        # Handle "Semua Varian"
        expand_list = []
        if "semua varian" in clean_name.lower():
            found_collection = False
            # Cek kategori mana yang dimaksud
            check_str = f"{curr_cat} {clean_name}".lower()
            for key, variants in AUTO_VARIANTS.items():
                if key in check_str:
                    for v in variants:
                        query = f"{curr_brand} {curr_cat} {v}".strip()
                        expand_list.append(query)
                    found_collection = True
                    break
            if not found_collection: # Default fallback
                expand_list.append(f"{curr_brand} {curr_cat} {clean_name}")
        else:
            # Item biasa
            expand_list.append(f"{curr_brand} {curr_cat} {clean_name}")

        # 6. SEARCH
        active_bonus = line_bonus if line_bonus else (header_bonus if header_bonus else footer_bonus)
        
        for q in expand_list:
            res_name, res_score, res_brand, res_code = search_hybrid(q, brand_filter=curr_brand)
            results.append({
                "Kode": res_code,
                "Nama Barang": res_name,
                "Qty": qty_str if qty_str else "1",
                "Bonus": active_bonus,
                "Score": res_score,
                "Input": q
            })

    return lines[0], results

# ==========================================
# 5. USER INTERFACE
# ==========================================
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### 📥 Input PO Sales")
    raw_text = st.text_area("Paste chat di sini:", height=400, placeholder="Contoh:\nToko Jaya\nSyb\n-powder mask\nTea tree x20\n...")
    if st.button("🚀 PROSES DATA", type="primary", use_container_width=True):
        if raw_text:
            store, res = parse_po(raw_text)
            st.session_state['results'] = (store, res)

with col2:
    st.markdown("### 📤 Hasil Analisa")
    if 'results' in st.session_state:
        store, data = st.session_state['results']
        if data:
            st.success(f"Customer: **{store}**")
            df_res = pd.DataFrame(data)
            
            st.dataframe(
                df_res[["Kode", "Nama Barang", "Qty", "Bonus", "Score"]],
                column_config={
                    "Score": st.column_config.ProgressColumn("Akurasi", min_value=0, max_value=1, format="%.2f"),
                    "Qty": st.column_config.TextColumn("Jumlah"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Text Copy Format
            txt = f"Customer: {store}\n"
            for item in data:
                b_txt = f"({item['Bonus']})" if item['Bonus'] else ""
                txt += f"{item['Kode']} | {item['Nama Barang']} | {item['Qty']} {b_txt}\n"
            
            st.text_area("Siap Copy:", value=txt, height=200)
        else:
            st.error("Tidak ada item yang terdeteksi. Cek format input.")
