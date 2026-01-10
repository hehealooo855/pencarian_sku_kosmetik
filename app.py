import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Fakturis Enterprise", page_icon="🏢", layout="wide")

st.title("🏢 AI Fakturis Enterprise")
st.markdown("Mendukung: Multi-Brand per PO, Pewarisan Konteks, Split Koma, & Deteksi Promo Otomatis.")

# --- 1. KAMUS "SEMUA VARIAN" ---
# Isi bagian ini sesuai katalog real Anda agar sales tidak perlu ngetik satu-satu
AUTO_VARIANTS = {
    "eye mask": ["Gold", "Osmanthus", "Seaweed", "Black Pearl"], 
    "lip mask": ["Peach", "Strawberry", "Blueberry"],
    "sheet mask": ["Aloe", "Pomegranate", "Honey", "Olive", "Blueberry"],
    "powder mask": ["Greentea", "Lavender", "Peppermint", "Strawberry"],
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
        # Membersihkan teks database agar mudah dicocokkan
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
    # Menggunakan N-gram 2-5 agar sangat toleran terhadap typo (misal: 'delipatory' -> 'depilatory')
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])

# --- 4. FUNGSI PENCARIAN SKU ---
def search_sku(query):
    """Mencari barang di database berdasarkan query teks"""
    if not query or len(query) < 2: return None, 0.0, ""

    query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    final_scores = similarity_scores.copy()
    
    # Logika Boosting untuk Besar/Kecil
    if "kecil" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(30ml|50gr|50ml|60ml|kecil|mini|sachet)\b', row['Clean_Text']): final_scores[idx] += 0.25
            elif re.search(r'\b(200ml|500ml|besar|jumbo|1000ml)\b', row['Clean_Text']): final_scores[idx] -= 0.15
            
    if "besar" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(200ml|250ml|500ml|1000ml|besar|jumbo)\b', row['Clean_Text']): final_scores[idx] += 0.25
            elif re.search(r'\b(30ml|50gr|kecil|mini|sachet)\b', row['Clean_Text']): final_scores[idx] -= 0.15

    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    # Threshold 0.1 agar AI berani menebak typo parah
    if best_score > 0.1:
        return f"{df.iloc[best_idx]['Merk']} - {df.iloc[best_idx]['Nama Barang']}", best_score, df.iloc[best_idx]['Merk']
    return "❌ TIDAK DITEMUKAN", 0.0, ""

# --- 5. LOGIKA PARSING PO (THE MASTERPIECE) ---
def parse_po_complex(text):
    lines = text.split('\n')
    results = []
    
    # Variabel "Ingatan" (Context Memory)
    current_brand = ""      # Misal: "Honor", "SYB"
    current_category = ""   # Misal: "Peel off mask"
    current_parent = ""     # Misal: "Aku ayu body cream" (untuk kasus Tambahan order)
    global_bonus = ""       # Misal: (12+1) yang berlaku untuk satu blok
    
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    # Daftar Brand untuk deteksi Header Brand
    known_brands = df['Merk'].str.lower().unique().tolist() if df is not None else []

    for line in lines[1:]: # Skip baris pertama (Nama Toko)
        line = line.strip()
        if not line or line == "-": continue
        
        # A. DETEKSI ANGKA (QTY, BONUS, DISKON)
        # ------------------------------------------------
        # Regex Qty: Angka diikuti satuan, ATAU kata "per" diikuti angka/satuan
        qty_match = re.search(r'(per\s*)?(\d+)?\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol|pack|kotak)', line, re.IGNORECASE)
        qty_str = qty_match.group(0) if qty_match else ""
        
        # Regex Bonus: (12+1) atau 12+1
        bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line)
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        # Regex Diskon: Ada %
        disc_match = re.search(r'\(?([\d\+\.\s]+%)\)?', line)
        disc_str = disc_match.group(1) if disc_match else ""
        
        # B. BERSIHKAN TEKS UNTUK JADI KEYWORD
        # ------------------------------------------------
        clean_line = line
        if qty_str: clean_line = clean_line.replace(qty_str, "")
        if bonus_str: clean_line = clean_line.replace(bonus_match.group(0), "")
        if disc_str: clean_line = clean_line.replace(disc_match.group(0), "")
        
        # Hapus karakter non-huruf di awal (seperti - atau .)
        clean_keyword = re.sub(r'^[\s\-\.]+', '', clean_line).strip()
        
        # C. LOGIKA PENENTUAN: INI ITEM ATAU JUDUL?
        # ------------------------------------------------
        is_item = False
        if qty_match: is_item = True # Kalau ada Qty, pasti item
        
        # Jika bukan item, kemungkinan ini Judul Brand atau Judul Kategori
        if not is_item:
            lower_key = clean_keyword.lower()
            
            # Cek apakah ini ganti Brand? (Misal: "Honor", "Vlagio", "SYB")
            found_brand_header = False
            for brand in known_brands:
                if brand in lower_key:
                    current_brand = brand # Update Brand Aktif
                    current_category = "" # Reset Kategori
                    current_parent = ""   # Reset Parent
                    
                    # Cek jika di header brand ada promo global. Contoh: "Honor (12+1)"
                    if bonus_str: global_bonus = bonus_str
                    else: global_bonus = "" # Reset jika tidak ada
                    
                    found_brand_header = True
                    break
            
            if not found_brand_header:
                # Jika bukan Brand, berarti ini Judul Kategori/Parent (Misal: "Peel off mask")
                # Tapi kalau ini "Tambahan order", reset semuanya
                if "tambahan order" in lower_key:
                    current_brand = ""
                    current_category = ""
                    current_parent = ""
                    global_bonus = ""
                else:
                    # Ini adalah kategori baru (misal: "Peel off mask" atau "Aku ayu body cream")
                    if len(lower_key) > 3: # Validasi panjang minimal
                        current_category = clean_keyword 
                        # Jika baris ini tidak punya qty tapi terlihat seperti produk lengkap, jadikan parent
                        # Contoh: "Aku ayu body cream botol kecil" -> Item bawahnya cuma "Goatmilk"
                        current_parent = clean_keyword
            
            continue # Lanjut ke baris berikutnya (karena ini cuma judul)

        # D. JIKA INI ADALAH ITEM (ADA QTY)
        # ------------------------------------------------
        items_to_process = []
        
        # D.1 Cek "Semua Varian"
        if "semua varian" in clean_keyword.lower():
            # Cari di kamus
            found_in_dict = False
            for key, variants in AUTO_VARIANTS.items():
                if key in current_category.lower() or key in clean_keyword.lower():
                    # Prefix adalah gabungan brand + kategori + nama item (tanpa kata semua varian)
                    base_name = clean_keyword.lower().replace("semua varian", "").strip()
                    prefix = f"{current_brand} {current_category} {base_name}".strip()
                    
                    for var in variants:
                        items_to_process.append(f"{prefix} {var}")
                    found_in_dict = True
                    break
            if not found_in_dict:
                # Jika tidak ada di kamus, biarkan apa adanya
                items_to_process.append(f"{current_brand} {current_category} {clean_keyword}")

        # D.2 Cek Koma (Splitter) - Contoh: "cucumber, blueberry, dna salmon"
        elif "," in clean_keyword:
            parts = clean_keyword.split(',')
            
            # Heuristik Cerdas: Kata sebelum koma pertama biasanya mengandung "Induk Kata"
            # Contoh: "Jelly mask cucumber" -> Induk "Jelly mask", Varian "cucumber"
            first_part_words = parts[0].split()
            if len(first_part_words) > 1:
                # Ambil semua kecuali kata terakhir sebagai induk lokal
                local_prefix = " ".join(first_part_words[:-1]) 
            else:
                local_prefix = current_category # Fallback ke kategori atas
            
            # Tambahkan part pertama (utuh)
            full_query_1 = f"{current_brand} {current_category} {parts[0]}".strip()
            items_to_process.append(full_query_1)
            
            # Tambahkan part sisanya (digabung dengan induk)
            for part in parts[1:]:
                # Gabung: Brand + Induk Lokal + Varian
                full_query_n = f"{current_brand} {local_prefix} {part.strip()}".strip()
                items_to_process.append(full_query_n)
                
        # D.3 Item Biasa
        else:
            # Strategi Penggabungan Nama:
            # 1. Jika ada Parent (Aku ayu body cream) -> Parent + Item (Goatmilk)
            # 2. Jika tidak -> Brand + Kategori + Item
            
            final_query = ""
            
            # Cek apakah item ini cuma varian pendek? (misal "Goatmilk", "Violet")
            is_short_variant = len(clean_keyword.split()) <= 2
            
            if current_parent and is_short_variant:
                final_query = f"{current_parent} {clean_keyword}"
            else:
                final_query = f"{current_brand} {current_category} {clean_keyword}"
            
            items_to_process.append(final_query.strip())
            
        # E. EKSEKUSI PENCARIAN KE AI
        # ------------------------------------------------
        # Cek Pewarisan Bonus (Jika item tidak punya bonus, tapi Header Brand punya)
        final_bonus = bonus_str if bonus_str else global_bonus
        
        for query in items_to_process:
            sku_res, score, detected_merk = search_sku(query)
            
            results.append({
                "Konteks": f"{current_brand} | {current_category}",
                "Input Original": query, # Apa yang dicari mesin
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
    raw_text = st.text_area("Paste Chat di sini:", height=500, placeholder="Paste contoh PO Anda di sini...")
    process_btn = st.button("🚀 PROSES DATA", type="primary")

with col2:
    st.subheader("📊 Hasil Analisa Faktur")
    
    if process_btn and raw_text:
        store_name, data = parse_po_complex(raw_text)
        
        st.info(f"🏪 **Nama Toko:** {store_name}")
        
        if data:
            df_res = pd.DataFrame(data)
            
            # Format Tabel agar cantik
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
            
            # Tombol Copy (Experimental)
            text_result = f"Toko: {store_name}\n"
            for item in data:
                text_result += f"{item['Hasil Pencarian SKU']} | Qty: {item['Qty']} | Bonus: {item['Bonus']}\n"
            
            st.text_area("Hasil Teks (Untuk Copy Manual jika perlu)", value=text_result, height=150)
            
        else:
            st.warning("Tidak ada item yang terdeteksi. Pastikan format ada Qty (pcs/lsn/kotak).")
