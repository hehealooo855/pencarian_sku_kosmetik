import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Fakturis Pintar", page_icon="🤖", layout="wide")

st.title("🤖 AI Fakturis Pintar")
st.markdown("Copy-paste PO dari WhatsApp Group, sistem akan merapikan dan mencari SKU-nya.")

# --- LOAD DATA DARI GOOGLE SHEET ---
@st.cache_data(ttl=600)
def load_data():
    # -----------------------------------------------------------
    # GANTI LINK DI BAWAH INI DENGAN LINK ANDA
    # -----------------------------------------------------------
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv' 
    # ^^^ JANGAN LUPA GANTI ID SHEET DI ATAS ^^^

    try:
        df = pd.read_csv(sheet_url)
        # Pre-processing
        df['Full_Text'] = df['Merk'].astype(str) + ' ' + df['Nama Barang'].astype(str)
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df
    except Exception as e:
        st.error("Gagal koneksi ke Google Sheet. Pastikan link benar.")
        return None

df = load_data()

# --- LATIH MODEL AI ---
@st.cache_resource
def train_model(data):
    if data is None or data.empty:
        return None, None
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])

# --- FUNGSI PARSING PO (INTI KECERDASAN BARU) ---
def parse_po_line(line):
    """
    Memisahkan Nama Barang dari Qty, Bonus, dan Diskon menggunakan Regex
    """
    line = line.strip()
    if not line or line == "-": return None

    # 1. Ambil Qty (Angka diikuti pcs/lsn/lusin/box)
    qty_match = re.search(r'(\d+)\s*(pcs|pc|lsn|lusin|box|kotak|btl|botol)', line, re.IGNORECASE)
    qty = qty_match.group(0) if qty_match else "-"
    
    # 2. Ambil Bonus (Pola 12+1, 6+1, dst)
    # Mencari pola angka+angka, tapi hindari yang ada % (karena itu diskon)
    bonus_match = re.search(r'\(?(\d+\s*\+\s*\d+)\)?(?!%)', line)
    bonus = bonus_match.group(1) if bonus_match else "-"

    # 3. Ambil Diskon (Pola ada persen %)
    disc_match = re.search(r'\(?([\d\+\.\s]+%)\)?', line)
    disc = disc_match.group(1) if disc_match else "-"

    # 4. Bersihkan Teks untuk Pencarian Barang
    # Kita hapus bagian qty, bonus, diskon dari teks asli agar AI tidak bingung
    clean_name = line
    if qty_match: clean_name = clean_name.replace(qty_match.group(0), "")
    if bonus_match: clean_name = clean_name.replace(bonus_match.group(0), "")
    if disc_match: clean_name = clean_name.replace(disc_match.group(0), "")
    
    # Hapus karakter sisa (-) di awal
    clean_name = re.sub(r'^[\s-]*', '', clean_name)
    
    return {
        "original": line,
        "search_query": clean_name,
        "qty": qty,
        "bonus": bonus,
        "disc": disc
    }

def cari_barang_terbaik(query_clean):
    """Mencari 1 barang terbaik dari database"""
    if not query_clean or len(query_clean) < 3: return "TIDAK DITEMUKAN", 0.0

    query_vec = tfidf_vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Logika Besar/Kecil (Sama seperti sebelumnya)
    final_scores = similarity_scores.copy()
    if "kecil" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(30ml|50gr|50ml|60ml|kecil|mini)\b', row['Clean_Text']): final_scores[idx] += 0.2
            elif re.search(r'\b(200ml|500ml|besar|jumbo)\b', row['Clean_Text']): final_scores[idx] -= 0.1
    if "besar" in query_clean:
        for idx, row in df.iterrows():
            if re.search(r'\b(200ml|500ml|besar|jumbo)\b', row['Clean_Text']): final_scores[idx] += 0.2
            elif re.search(r'\b(30ml|50gr|kecil|mini)\b', row['Clean_Text']): final_scores[idx] -= 0.1

    best_idx = final_scores.argmax()
    best_score = final_scores[best_idx]
    
    if best_score > 0.15: # Threshold
        return f"{df.iloc[best_idx]['Merk']} - {df.iloc[best_idx]['Nama Barang']}", best_score
    else:
        return "TIDAK DITEMUKAN (Cek Manual)", 0.0

# --- USER INTERFACE UTAMA ---
if df is not None:
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("1. Input PO WhatsApp")
        raw_text = st.text_area("Paste Chat di sini:", height=300, 
                                placeholder="SJJ petumbukan\n-Tata hairmask susu 1000ml 12pcs\n-Aha gluta hitam 200ml...")
        process_btn = st.button("🚀 Proses PO", type="primary")

    with col_output:
        st.subheader("2. Hasil Faktur")
        
        if process_btn and raw_text:
            lines = raw_text.split('\n')
            
            # Asumsi Baris 1 adalah Nama Toko
            nama_toko = lines[0].strip()
            st.info(f"🏬 **Nama Toko:** {nama_toko}")
            
            hasil_data = []
            
            # Proses baris ke-2 sampai habis
            for line in lines[1:]:
                parsed = parse_po_line(line)
                if parsed and len(parsed['search_query']) > 2:
                    # Cari di Database
                    sku_result, score = cari_barang_terbaik(parsed['search_query'])
                    
                    hasil_data.append({
                        "Input Sales": parsed['search_query'],
                        "Hasil Pencarian SKU": sku_result,
                        "Qty": parsed['qty'],
                        "Bonus": parsed['bonus'],
                        "Diskon": parsed['disc'],
                        "Akurasi": f"{int(score*100)}%"
                    })
            
            if hasil_data:
                res_df = pd.DataFrame(hasil_data)
                
                # Tampilkan Tabel yang bisa diedit (Data Editor)
                edited_df = st.data_editor(
                    res_df, 
                    column_config={
                        "Akurasi": st.column_config.ProgressColumn(
                            "Kecocokan", format="%s", min_value=0, max_value=100
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.success("✅ Selesai! Silakan salin data di atas ke sistem faktur.")
            else:
                st.warning("Tidak ada item yang terdeteksi.")

