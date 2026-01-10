import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Pencarian SKU Pintar", page_icon="🔍")

# Judul dan Deskripsi
st.title("🔍 Smart SKU Search")
st.markdown("Ketik pesan dari sales (PO) apa adanya, sistem akan mencari SKU yang paling cocok.")

# --- LOAD DATA (DENGAN CACHE AGAR CEPAT) ---
@st.cache_data
def load_data():
    # GANTI NAMA FILE DI SINI SESUAI FILE ANDA DI FOLDER
    file_path = 'Laporan_Sales_Profesional_2026-01-09.xlsx - Sales Data.csv'
    
    try:
        df = pd.read_csv(file_path)
        # Pre-processing
        df['Full_Text'] = df['Merk'].astype(str) + ' ' + df['Nama Barang'].astype(str)
        df['Clean_Text'] = df['Full_Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()))
        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

df = load_data()

# --- LATIH MODEL AI (DENGAN CACHE) ---
@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(data)
    return vectorizer, matrix

if df is not None:
    tfidf_vectorizer, tfidf_matrix = train_model(df['Clean_Text'])
    
    # --- UI PENCARIAN ---
    query = st.text_input("Masukkan PO Sales:", placeholder="Contoh: skin1004 cleansing oil kecil")

    if query:
        # LOGIKA PENCARIAN (Sama seperti sebelumnya)
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
                    st.metric(label="Kecocokan", value=f"{int(score*100)}%")
                with col2:
                    merk = df.iloc[index]['Merk']
                    barang = df.iloc[index]['Nama Barang']
                    st.success(f"**{merk}**")
                    st.write(f"{barang}")
                st.divider()
        
        if not found_any:
            st.warning("Barang tidak ditemukan. Coba kata kunci lain.")

else:
    st.info("Mohon masukkan file CSV ke dalam folder yang sama dengan script ini.")
