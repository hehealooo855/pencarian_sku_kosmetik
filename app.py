import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import SequenceMatcher

# ==========================================
# 0. KONFIGURASI & SETUP
# ==========================================
st.set_page_config(page_title="AI Fakturis Pro", page_icon="🧬", layout="wide")
st.title("🧬 AI Fakturis Pro (Flexi-Parser)")
st.markdown("Fitur: **Colon Parser (:12pcs)**, **Smart Context**, **Auto-Switch Category**.")

# ==========================================
# 1. KAMUS DATA
# ==========================================
AUTO_VARIANTS = {
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry", "Tea Tree"],
    "peeling gel": ["Aloe", "Charcoal", "Milk", "Snail", "Lemon"],
    "peeling": ["Aloe", "Charcoal", "Milk", "Snail", "Lemon"],
    "toner badan": ["Red Jelly", "Coklat", "Fresh Skin", "Kelupas", "Ginseng"],
    "toner": ["Rose", "Chamomile", "Aloe", "Lemon"],
    "masker wajah": ["Bengkoang", "Aloe Vera", "Cucumber", "Avocado"],
    "cushion": ["01", "02", "03", "04", "Ivory", "Natural", "Beige"], # Tambahan untuk Goute
    "blush": ["01", "02", "03", "04"], # Tambahan untuk Goublush
    "yu chun mei": ["Night Cream", "Day Cream", "Cleanser", "Serum"]
}

BRAND_ALIASES = {
    "sekawan": "AINIE", "ainie": "AINIE",
    "javinci": "JAVINCI", "thai": "THAI", 
    "syb": "SYB", "diosys": "DIOSYS", "satto": "SATTO", 
    "vlagio": "VLAGIO", "honor": "HONOR", "hanasui": "HANASUI",
    "implora": "IMPLORA", "brasov": "BRASOV", "tata": "JAVINCI",
    "body white": "JAVINCI", "holly": "HOLLY",
    "yu chun mei": "YU CHUN MEI", "ycm": "YU CHUN MEI",
    # Tambahan Brand Baru dari PO Sales
    "whitelab": "WHITELAB", "bonavie": "BONAVIE", "goute": "GOUTE",
    "kim": "KIM", "kim kosmetik": "KIM"
}

KEYWORD_REPLACEMENTS = {
    "cream night": "night cream", 
    "krim malam": "night cream",
    "cream malam": "night cream",
    "malam": "night cream",
    "cream day": "day cream",
    "krim siang": "day cream", 
    "cream siang": "day cream",
    "siang": "day cream",
    "cleanser": "cleanser", 
    "sabun muka": "cleanser",
    "facial wash": "cleanser",
    "bulat": "150ml", "gepeng": "100ml",
    "all": "semua varian", "campur": "semua varian",
    "manjakani 100ml": "manjakani 110ml", "manjakani 100": "manjakani 110ml",
    "pepaya": "papaya", "bengkuang": "bengkoang",
    "barsoap": "soap", "bar soap": "soap", "sabun": "soap",
    "minyak": "oil", "zaitun": "olive oil",
    "50g": "50gr", "50gram": "50gr",
    "trii": "tree", "pappermint": "peppermint",
    # Fix Typo Bonavie
    "creme": "cream", "canele": "canale" 
}

HARD_CONFLICTS = {
    "olive": ["candlenut", "kemiri", "urang"],
    "zaitun": ["candlenut", "kemiri", "urang"],
    "kemiri": ["olive", "zaitun", "urang"],
    "candlenut": ["olive", "zaitun"],
    "cream": ["serum", "toner", "cleanser", "soap"], 
    "krim": ["serum", "toner", "cleanser", "soap"],
    "serum": ["cream", "night", "day", "krim", "cleanser"],
    "night": ["day", "siang"],
    "malam": ["day", "siang"],
    "day": ["night", "malam"],
    "siang": ["night", "malam"]
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
        
        col_map = {}
        for col in df.columns:
            c_low = col.lower()
            if "kode" in c_low and "barang" in c_low: col_map['kode'] = col
            if "nama" in c_low and "barang" in c_low: col_map['nama'] = col
            if "merek" in c_low or "merk" in c_low: col_map['merk'] = col

        if len(col_map) < 3: return None
        
        df = df.rename(columns={col_map['kode']: 'Kode', col_map['nama']: 'Nama', col_map['merk']: 'Merk'})
        df = df[['Kode', 'Nama', 'Merk']].dropna(subset=['Nama'])
        
        df['Clean_Text'] = df['Nama'] + " " + df['Merk']
        df['Clean_Text'] = df['Clean_Text'].astype(str).str.lower()
        df['Clean_Text'] = df['Clean_Text'].apply(lambda x: re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', x))
        df['Clean_Text'] = df['Clean_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', x))
        
        return df
    except Exception as e: st.error(f"DB Error: {e}"); return None

df = load_data()

# ==========================================
# 3. HYBRID ENGINE
# ==========================================
@st.cache_resource
def train_tfidf(data):
    if data is None or data.empty: return None, None
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3)) 
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_tfidf(df['Clean_Text'])

def extract_numbers(text):
    return re.findall(r'(\d+)', text)

def search_hybrid(query, brand_filter=None):
    if not query or len(query) < 2: return None, 0.0, "", ""

    query_clean = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', query.lower())
    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query_clean).strip()
    
    # KEYWORD REPLACEMENT (Translator)
    for k, v in KEYWORD_REPLACEMENTS.items():
        if k in query_clean: 
            query_clean = query_clean.replace(k, v)
    
    search_query = query_clean

    query_vec = tfidf_vectorizer.transform([search_query])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_scores.argsort()[-50:][::-1]
    
    candidates = df.iloc[top_indices].copy()
    candidates['tfidf_score'] = cosine_scores[top_indices]
    
    best_match = None
    best_score = -100
    
    user_nums = extract_numbers(query_clean)
    
    for idx, row in candidates.iterrows():
        db_text = row['Clean_Text']
        db_brand = str(row['Merk']).lower()
        
        fuzzy_ratio = SequenceMatcher(None, search_query, db_text).ratio()
        final_score = (row['tfidf_score'] * 0.4) + (fuzzy_ratio * 0.6)
        
        if brand_filter and brand_filter.lower() not in db_brand:
            continue
            
        num_mismatch = False
        for num in user_nums:
            if int(num) > 20: 
                if not re.search(r'\b' + num + r'\b', db_text):
                    num_mismatch = True
        
        if num_mismatch: final_score -= 0.5 

        # --- LOGIC ANTI-CLASH ---
        clash = False
        for key, enemies in HARD_CONFLICTS.items():
            if key in search_query: 
                for enemy in enemies:
                    if enemy in db_text:
                        clash = True; break
        if clash: continue 

        for kw in ESSENTIAL_KEYWORDS:
            if kw in search_query and kw not in db_text:
                final_score -= 0.4 
            if kw not in search_query and kw in db_text:
                final_score -= 0.1 

        colors = ["hitam", "kuning", "putih", "ungu", "gold", "pink", "blue"]
        for c in colors:
            if c in search_query and c not in db_text:
                has_other_color = any(oc in db_text for oc in colors if oc != c)
                if has_other_color: final_score -= 0.5 

        if final_score > best_score:
            best_score = final_score
            best_match = row

    threshold = 0.35 if not brand_filter else 0.20 
    
    if best_match is not None and best_score > threshold:
        return best_match['Nama'], round(best_score, 2), best_match['Merk'], best_match['Kode']
    else:
        return "❌ TIDAK DITEMUKAN", 0.0, "", "-"

# ==========================================
# 4. PARSER PO (FLEXIBLE UPDATE)
# ==========================================
def parse_po(text):
    lines = text.split('\n')
    results = []
    
    curr_brand = ""
    curr_cat = ""
    footer_bonus = ""
    header_bonus = ""
    
    for line in reversed(lines):
        if re.fullmatch(r'\s*\d+\s*[\+\*]\s*\d+\s*', line):
            footer_bonus = line.strip()
            break

    db_brands = [str(b).lower() for b in df['Merk'].unique() if len(str(b)) > 2]

    for line in lines[1:]:
        line = line.strip()
        if not line or line == "-" or line == footer_bonus: continue
        
        line_clean = re.sub(r'^[\-\.\s]+', '', line) 
        line_clean = re.sub(r'\b(cash|tunai|kredit|tempo)\b', '', line_clean, flags=re.IGNORECASE)
        line_clean = re.sub(r'[\(\)]', ' ', line_clean)
        
        # --- FIX UTAMA: COLON PARSER (:12pcs -> x 12pcs) ---
        # Ini mengubah "serum:12pcs" menjadi "serum x 12pcs" agar tertangkap regex monster
        line_clean = re.sub(r':\s*(\d+)', r' x \1', line_clean)
        
        # Regex Monster
        qty_pattern = r'(?:^|\s)(?:x|@|@x)?\s*(\d+)\s*(?:x|pcs|pc|lsn|box|ktk|pak|btl)?(?:\s*[\+\*]\s*\d+)?(?:$|\s)'
        qty_match = re.findall(qty_pattern, line_clean, re.IGNORECASE)
        
        qty_str = ""
        is_item = False
        
        if qty_match:
            valid_nums = [m for m in qty_match if m.strip()]
            if valid_nums:
                qty_str = valid_nums[0]
                is_item = True
        
        line_bonus_match = re.search(r'\d+\s*[\+\*]\s*\d+', line_clean)
        line_bonus = line_bonus_match.group(0) if line_bonus_match else ""
        if line_bonus: is_item = True

        clean_name = re.sub(qty_pattern, ' ', line_clean, flags=re.IGNORECASE)
        clean_name = re.sub(r'\d+\s*[\+\*]\s*\d+', ' ', clean_name)
        
        clean_name = re.sub(r'[^\w\s]', ' ', clean_name).strip()
        words = clean_name.lower().split()
        final_words = []
        for w in words:
            final_words.append(KEYWORD_REPLACEMENTS.get(w, w))
        clean_name = " ".join(final_words)

        # LOGIKA HIERARKI (Header vs Item)
        if not is_item:
            lower_name = clean_name.lower()
            detected_brand = None
            
            for alias, real in BRAND_ALIASES.items():
                if lower_name == alias or lower_name.startswith(alias + " "):
                    detected_brand = real
                    clean_name = lower_name.replace(alias, "").strip()
                    break
            
            if not detected_brand:
                for db_b in db_brands:
                    if db_b in lower_name: 
                        detected_brand = next((v for k,v in BRAND_ALIASES.items() if k == db_b), db_b.upper())
                        clean_name = lower_name.replace(db_b, "").strip()
                        break
            
            if detected_brand:
                curr_brand = detected_brand
                curr_cat = clean_name
                header_bonus = line_bonus
                continue
            else:
                # Jika bukan brand, berarti nama kategori (misal "Goublush" atau "Cushion")
                if len(clean_name) > 2:
                    curr_cat = clean_name
            continue

        # LOGIKA KONSTRUKSI QUERY
        expand_list = []
        if "semua varian" in clean_name.lower():
            found_collection = False
            check_str = f"{curr_cat} {clean_name}".lower()
            
            effective_cat = curr_cat 
            for key, variants in AUTO_VARIANTS.items():
                if key in clean_name.lower():
                    effective_cat = key 
                    for v in variants: expand_list.append(f"{curr_brand} {effective_cat} {v}".strip())
                    found_collection = True; break
                elif key in check_str:
                    for v in variants: expand_list.append(f"{curr_brand} {key} {v}".strip())
                    found_collection = True; break
            
            if not found_collection:
                expand_list.append(f"{curr_brand} {curr_cat} {clean_name}")
        else:
            # Gabungkan Brand + Kategori (jika ada) + Nama Item
            # Ini menangani kasus "01:36pcs" -> "Goute Cushion 01"
            expand_list.append(f"{curr_brand} {curr_cat} {clean_name}")

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
    raw_text = st.text_area("Paste chat di sini:", height=400, placeholder="Contoh:\nKim kosmetik\nWhitelab\nSerum:12pcs")
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
            
            txt = f"Customer: {store}\n"
            for item in data:
                b_txt = f"({item['Bonus']})" if item['Bonus'] else ""
                txt += f"{item['Kode']} | {item['Nama Barang']} | {item['Qty']} {b_txt}\n"
            
            st.text_area("Siap Copy:", value=txt, height=200)
        else:
            st.error("Tidak ada item yang terdeteksi. Cek format input.")
