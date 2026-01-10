import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Pencarian SKU Pintar", page_icon="🔍")

st.title("🔍 Smart SKU Search (Google Sheet Version)")
st.markdown("Database terhubung langsung ke Google Sheet. Data update otomatis setiap 10 menit.")

# --- LOAD DATA DARI GOOGLE SHEET ---
# ttl=600 artinya data akan di-refresh dari Google Sheet setiap 600 detik (10 menit)
@st.cache_data(ttl=600)
def load_data():
    # -----------------------------------------------------------
    # GANTI LINK DI BAWAH INI DENGAN LINK "EXPORT CSV" ANDA
    # Pastikan akhiran linknya adalah "/export?format=csv"
    # -----------------------------------------------------------
    sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRqUOC7mKPH8FYtrmXUcFBa3zYQfh2sdC5sPFUFafInQG4wE-6bcBI3OEPLKCVuMdm2rZYgXzkBCcnS/pub?gid=0&single=true&output=csv'
    
    try:
        # Pandas bisa membaca langsung dari URL CSV
        df = pd.read_csv(sheet_url)
        
        # Pre-processing (Membersihkan data agar mudah dicari AI)
        df['Full_Text'] = df['Merk'].astype(str) + ' ' + df['Nama Barang'].astype(str)
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df
    except Exception as e:
        st.error("Gagal mengambil data dari Google Sheet.")
        st.error(f"Error detail: {e}")
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
    
    # --- UI PENCARIAN ---
    query = st.text_input("Masukkan PO Sales:", placeholder="Contoh: skin1004 cleansing oil kecil")

    if query and tfidf_matrix is not None:
        # LOGIKA PENCARIAN
        query_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
        query_vec = tfidf_vectorizer.transform([query_clean])
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        final_scores = similarity_scores.copy()
        
        # Logika Besar/Kecil
        if "kecil" in query_clean:
            for idx, row in df.iterrows():
                if re.search(r'\b(30ml|50gr|50ml|60ml|kecil|mini)\b', row['Clean_Text']):
                    final_scores[idx] += 0.2
                elif re.search(r'\b(200ml|500ml|besar|jumbo)\b', row['Clean_Text']):
                    final_scores[idx] -= 0.1
                    
        if "besar" in query_clean:
            for idx, row in df.iterrows():
                if re.search(r'\b(200ml|250ml|500ml|1000ml|besar|jumbo)\b', row['Clean_Text']):
                    final_scores[idx] += 0.2
                elif re.search(r'\b(30ml|50gr|kecil|mini)\b', row['Clean_Text']):
                    final_scores[idx] -= 0.1

        # Ambil Top 5
        top_indices = final_scores.argsort()[-5:][::-1]
        
        st.write("---")
        st.subheader("Hasil Rekomendasi:")
        
        found_any = False
        for index in top_indices:
            score = final_scores[index]
            if score > 0.15: # Threshold
                found_any = True
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(label="Akurasi", value=f"{int(score*100)}%")
                with col2:
                    merk = df.iloc[index]['Merk']
                    barang = df.iloc[index]['Nama Barang']
                    st.success(f"**{merk}**")
                    st.write(f"{barang}")
                st.divider()
        
        if not found_any:
            st.warning("Barang tidak ditemukan. Coba kata kunci lain.")
            
    # Tampilkan tombol refresh manual jika admin baru saja update data
    if st.button("Refresh Database Sekarang"):
        st.cache_data.clear()
        st.experimental_rerun()

else:
    st.info("Sedang menghubungkan ke Google Sheet...")
