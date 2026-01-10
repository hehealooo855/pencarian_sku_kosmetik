import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Fakturis Enterprise", page_icon="🏢", layout="wide")

st.title("🏢 AI Fakturis Enterprise (Strict Brand)")
st.markdown("Fitur Baru: **Brand Lock**. Jika header 'SYB', sistem 100% mengabaikan brand lain (Avione, dll).")

# --- 1. KAMUS "SEMUA VARIAN" (UPDATE SESUAI KATALOG ANDA) ---
# Tambahkan varian produk di sini agar sales yang malas ngetik tetap terdeteksi
AUTO_VARIANTS = {
    # Varian SYB
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
    "jelly mask": ["Cucumber", "Blueberry", "DNA Salmon", "Mugwort", "Watermelon"],
    # Varian Lain
    "hairmask": ["Susu", "Ginseng", "Coklat", "Strawberry"],
}

# --- 2. LOAD DATA ---
@st.cache_data(ttl=600)
def load_data():
    # GANTI LINK DI SINI DENGAN LINK GOOGLE SHEET CSV ANDA
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        df = pd.read_csv(sheet_url)
        df['Merk'] = df['Merk'].astype(str).str.strip()
        df['Nama Barang'] = df['Nama Barang'].astype(str).str.strip()
        df['Full_Text'] = df['Merk'] + ' ' + df['Nama Barang']
        # Membersihkan teks database
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

# --- 4. FUNGSI PENCARIAN SKU (DENGAN BRAND FILTER) ---
def search_sku(query, brand_filter=None):
    """
    Mencari barang.
    Jika brand_filter diisi (misal 'SYB'), maka barang merk lain SKOR-nya DI-NOL-KAN.
    """
    if not query or len(query) < 2: return None, 0.0, ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()

    # === LOGIKA BRAND LOCK (KACAMATA KUDA) ===
    if brand_filter:
        # Buat "Topeng" (Mask). True jika merk sesuai, False jika tidak.
        # Kita pakai str.contains agar jika filter "SYB", merk "SYB Cosmetics" tetap masuk
        # Tapi "Avione" akan jadi False.
        brand_mask = df['Merk'].astype(str).str.lower().str.contains(brand_filter.lower(), regex=False).to_numpy()
        
        # Kalikan skor dengan mask. 
        # Jika False (0), skor jadi 0. Jika True (1), skor tetap.
        final_scores = final_scores * brand_mask

    # Logika Boosting untuk Besar/Kecil
    if "kecil" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(30ml|50gr|50ml|60ml|kecil|mini|sachet|sct)\b', row['Clean_Text']): final_scores[idx] += 0.25
            elif re.search(r'\b(200ml|500ml|besar|jumbo|1000ml)\b', row['Clean_Text']): final_scores[idx] -= 0.15
            
    if "besar" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(200ml|250ml|500ml|1000ml|besar|jumbo)\b', row['Clean_Text']): final_scores[idx] += 0.25
            elif re.search(r'\b(30ml|50gr|kecil|mini|sachet|sct)\b', row['Clean_Text']): final_scores[idx] -= 0.15

    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    # Threshold 0.1 (Cukup rendah agar typo parah tetap ketemu, tapi aman karena sudah filter brand)
    if best_score > 0.1:
        return f"{df.iloc[best_idx]['Merk']} - {df.iloc[best_idx]['Nama Barang']}", best_score, df.iloc[best_idx]['Merk']
    else:
        # Jika skor 0 (artinya barang SYB dengan nama itu tidak ada), kembalikan Not Found
        # Jangan kembali ke Avione!
        return "❌ TIDAK DITEMUKAN (Cek Database)", 0.0, ""

# --- 5. LOGIKA PARSING PO ---
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    current_brand = ""      
    current_category = ""   
    current_parent = ""     
    global_bonus = ""       
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    # Ambil daftar merk dari database untuk deteksi header
    known_brands = df['Merk'].str.lower().unique().tolist() if df is not None else []

    for line in lines[1:]: 
        line = line.strip()
        if not line or line == "-": continue
        
        # A. Regex Qty, Bonus, Diskon
        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak)', line, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        disc_match = re.search(r'\(?([\d\+\.\s]+%)\)?', line)
        disc_str = disc_match.group(1) if disc_match else ""
        
        # B. Bersihkan Teks
        clean_line = line
        if qty_str: clean_line = clean_line.replace(qty_str, "")
        if bonus_str: clean_line = clean_line.replace(bonus_match.group(0), "")
        if disc_str: clean_line = clean_line.replace(disc_match.group(0), "")
        
        clean_keyword = re.sub(r'^[\s\-\.]+', '', clean_line).strip()
        
        # C. Deteksi Header (Brand / Kategori)
        is_item = False
        if qty_match: is_item = True 
        
        if not is_item:
            lower_key = clean_keyword.lower()
            
            # Cek Ganti Brand
            found_brand_header = False
            for brand in known_brands:
                # Cek exact word match atau startswith agar akurat
                # Misal: "SYB" match "syb", "syb kosmetik"
                if lower_key == brand or lower_key.startswith(brand + " "):
                    current_brand = brand 
                    current_category = "" 
                    current_parent = ""   
                    
                    if bonus_str: global_bonus = bonus_str
                    else: global_bonus = "" 
                    
                    found_brand_header = True
                    break
            
            if not found_brand_header:
                if "tambahan order" in lower_key:
                    current_brand = "" # Reset brand kalau masuk tambahan order campuran
                    current_category = ""
                    current_parent = ""
                    global_bonus = ""
                else:
                    # Kategori Baru
                    if len(lower_key) > 3: 
                        current_category = clean_keyword 
                        current_parent = clean_keyword
            
            continue 

        # D. Proses Item
        items_to_process = []
        
        # D.1 Cek "Semua Varian"
        if "semua varian" in clean_keyword.lower():
            found_in_dict = False
            # Gabungkan kategori + item untuk cek kamus
            # Misal Context: "Jelly mask", Item: "semua varian" -> Key: "jelly mask"
            full_check_str = f"{current_category} {clean_keyword}".lower()

            for key, variants in AUTO_VARIANTS.items():
                if key in full_check_str:
                    # Hapus kata "semua varian"
                    base_name = clean_keyword.lower().replace("semua varian", "").strip()
                    # Prefix pencarian: Brand + Kategori + Nama Item
                    prefix = f"{current_brand} {current_category} {base_name}".strip()
                    
                    for var in variants:
                        items_to_process.append(f"{prefix} {var}")
                    found_in_dict = True
                    break
            if not found_in_dict:
                items_to_process.append(f"{current_brand} {current_category} {clean_keyword}")

        # D.2 Cek Koma (Splitter)
        elif "," in clean_keyword:
            parts = clean_keyword.split(',')
            first_part_words = parts[0].split()
            if len(first_part_words) > 1:
                local_prefix = " ".join(first_part_words[:-1]) 
            else:
                local_prefix = current_category 
            
            # Item 1
            full_query_1 = f"{current_brand} {current_category} {parts[0]}".strip()
            items_to_process.append(full_query_1)
            
            # Item 2...n
            for part in parts[1:]:
                full_query_n = f"{current_brand} {local_prefix} {part.strip()}".strip()
                items_to_process.append(full_query_n)
                
        # D.3 Item Tunggal
        else:
            final_query = ""
            is_short_variant = len(clean_keyword.split()) <= 2
            
            if current_parent and is_short_variant:
                final_query = f"{current_parent} {clean_keyword}"
            else:
                final_query = f"{current_brand} {current_category} {clean_keyword}"
            
            items_to_process.append(final_query.strip())
            
        # E. Pencarian
        final_bonus = bonus_str if bonus_str else global_bonus
        
        for query in items_to_process:
            # === PENTING: KITA KIRIM CURRENT_BRAND SEBAGAI FILTER ===
            sku_res, score, detected_merk = search_sku(query, brand_filter=current_brand)
            
            results.append({
                "Brand Filter": current_brand if current_brand else "Auto",
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
    st.info("Tips: Pastikan baris judul brand (cth: 'SYB') dieja dengan benar sesuai database agar Brand Lock aktif.")
    raw_text = st.text_area("Paste Chat di sini:", height=500, placeholder="ivana martubung\nsyb\nEye mask semua varian per lusin...")
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
