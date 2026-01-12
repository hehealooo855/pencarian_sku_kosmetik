import re
from thefuzz import process, fuzz

# ==========================================
# 1. DATABASE PRODUK (SOURCE OF TRUTH)
# ==========================================
# Database ini harus lengkap agar pencocokan akurat
PRODUCT_DATABASE = {
    # --- THAI ---
    "TPA00001": "THAI PAPAYA LIGHT SOAP 130GR",
    "TPA00223": "THAI GOAT MILK & OLIVE OIL SOAP 100GR",
    "TH-OIL-ZAI": "THAI MINYAK ZAITUN (OLIVE OIL) 125ML",
    "TH-OIL-KEM": "THAI MINYAK KEMIRI 125ML",
    
    # --- JAVINCI ---
    "1200005": "BODY WHITE AHA GLUTA-HYA BRIGHT BODY TONE-UP SERUM 200ML (HITAM)",
    "1108.01": "BODY WHITE AHA GLUTA HYA UVBRIGHT BODY TONE-UP SERUM 100ML",
    
    # --- DIOSYS ---
    "DIO-100-NB": "DIOSYS COLOR 100ML NATURAL BLACK",
    "DIO-100-DB": "DIOSYS COLOR 100ML DARK BROWN",
    "DIO-100-BR": "DIOSYS COLOR 100ML BROWN",
    "DIO-100-CF": "DIOSYS COLOR 100ML COFFEE",
    "DIO-100-RW": "DIOSYS COLOR 100ML RED WINE",
    "DIO-100-GB": "DIOSYS COLOR 100ML GOLDEN BLONDE",
    "DIO-100-CH": "DIOSYS COLOR 100ML CHERRY",
    "DIO-100-LB": "DIOSYS COLOR 100ML LIGHT BLONDE",
    "DIO-100-BL": "DIOSYS COLOR 100ML BLEACHING",
    
    # --- ARTIST INC (Penyebab Error sebelumnya) ---
    "AIR8-ET": "ARTIST INC.REJUVEN-8 BACK TO BALANCE ESSENCE TONER 100ML"
}

# Mapping untuk pencarian
PRODUCT_NAMES = list(PRODUCT_DATABASE.values())
PRODUCT_MAP = {v: k for k, v in PRODUCT_DATABASE.items()} 

# ==========================================
# 2. LOGIKA PARSING CERDAS (CONTEXT AWARE)
# ==========================================

class SmartPOParser:
    def __init__(self, db_names, db_map):
        self.db_names = db_names
        self.db_map = db_map

    def clean_text(self, text):
        """Membersihkan simbol bullet point dan spasi."""
        text = re.sub(r'^[-*•\s]+', '', text) 
        return text.strip()

    def extract_quantity(self, text):
        """
        Mendeteksi apakah baris ini memiliki Quantity (berarti ini ITEM).
        Mengembalikan: (Nama Bersih, Quantity String, Bonus String)
        """
        # Regex mencari angka diikuti pcs/btl/kotak, dll
        qty_regex = r'(\d+)\s*(pcs|pc|btl|box|kotak|lsn)'
        qty_match = re.search(qty_regex, text, re.IGNORECASE)
        
        # Regex mencari bonus dalam kurung (misal: 12+1 atau 24+3)
        bonus_match = re.search(r'\((.*?)\)', text)
        
        qty_str = qty_match.group(0) if qty_match else None
        bonus_str = bonus_match.group(1) if bonus_match else ""
        
        # Bersihkan nama produk dari qty dan bonus untuk pencarian bersih
        clean_name = text
        if qty_str: clean_name = clean_name.replace(qty_str, "")
        if bonus_match: clean_name = clean_name.replace(bonus_match.group(0), "")
        
        return clean_name.strip(), qty_str, bonus_str

    def find_best_match(self, query):
        """Mencari produk menggunakan Fuzzy Matching"""
        # Token Set Ratio sangat bagus untuk kata yang diacak atau ada kata tambahan
        # Cutoff 70 berarti minimal kemiripan 70%
        match, score = process.extractOne(query, self.db_names, scorer=fuzz.token_set_ratio)
        
        if score >= 70: 
            return match, score
        return None, score

    def process_po_text(self, raw_text):
        lines = raw_text.strip().split('\n')
        results = []
        
        current_header = "" # INI KUNCINYA: Menyimpan konteks (misal: "Diosys 100ml")
        outlet_name = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            # Baris 1 dianggap nama toko
            if i == 0:
                outlet_name = line
                continue
            
            # Abaikan baris tagihan/komentar
            if line.startswith("#"): continue

            # 1. Analisa Baris: Apakah ini Item atau Header?
            item_text, qty, bonus = self.extract_quantity(line)
            item_text = self.clean_text(item_text)
            
            # Jika TIDAK ada quantity, kita anggap ini Header/Kategori baru
            if not qty:
                # Cek apakah ini header murni atau header dengan info bonus (24+3)
                current_header = item_text
                # Bersihkan info dalam kurung dari header (misal Diosys 100ml (24+3) -> Diosys 100ml)
                current_header = re.sub(r'\((.*?)\)', '', current_header).strip()
                continue 
            
            # 2. Bangun Query Pencarian
            # Gabungkan Header + Item. 
            # Contoh: Header="Diosys 100ml", Item="N.black" -> Query="Diosys 100ml N.black"
            
            # Normalisasi singkatan umum sales sebelum matching (Fix Typo)
            replacements = {
                "n.black": "natural black", "n black": "natural black",
                "d.brwon": "dark brown", "d.brown": "dark brown",
                "brwon": "brown", "coffe": "coffee", "cerry": "cherry",
                "zaitun": "minyak zaitun", "kemiri": "minyak kemiri"
            }
            
            lower_item = item_text.lower()
            for k, v in replacements.items():
                if k in lower_item:
                    lower_item = lower_item.replace(k, v)
            
            # Gabungkan Header + Item yang sudah diperbaiki typonya
            full_query = f"{current_header} {lower_item}"
            
            # 3. Cari di Database
            product_name, score = self.find_best_match(full_query)
            
            # Format tampilan Qty + Bonus untuk Output
            qty_display = qty
            if bonus: qty_display += f" ({bonus})"
            
            if product_name:
                sku = self.db_map[product_name]
                results.append({
                    "sku": sku,
                    "product": product_name,
                    "qty": qty_display,
                    "score": score
                })
            else:
                # Fallback jika tidak ketemu
                results.append({
                    "sku": "UNKNOWN",
                    "product": f"?? CEK MANUAL: {full_query} ??",
                    "qty": qty_display,
                    "score": 0
                })

        return outlet_name, results

# ==========================================
# 3. CONTOH INPUT DARI ANDA
# ==========================================

input_po_sales = """
Tiga kenza dolok masihul
#tagihan bayar

THAI
-Jinzu papaya 130gr 12pcs (12+1)
-zaitun 125ml 12pcs (12+1)
-kemiri 125ml 12pcs (12+1)

Javinci
Aha gluta tone up banded hitam 200ml 12pcs (12+1)
Aha body suncreen 100ml banded 12pcs (12+1)

Diosys 100ml (24+3)
N.black 12pcs
D.brwon 6pcs
Brwon 6pcs
Coffe 8pcs
Red wine 4pcs
Golden blonde 4pcs
Cerry 4pcs
Light blonde 4pcs
"""



# --- JALANKAN ---
if __name__ == "__main__":
    parser = SmartPOParser(PRODUCT_NAMES, PRODUCT_MAP)
    outlet, parsed_items = parser.process_po_text(input_po_sales)

    # Tampilkan Hasil Output di Terminal
    print(f"PO: {outlet}")
    print("=" * 100)
    print(f"{'SKU':<12} | {'NAMA PRODUK (DATABASE)':<60} | {'QTY'}")
    print("-" * 100)

    for item in parsed_items:
        print(f"{item['sku']:<12} | {item['product']:<60} | {item['qty']}")
