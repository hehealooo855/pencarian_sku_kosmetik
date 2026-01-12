import re
from thefuzz import process, fuzz

# ==========================================
# 1. DATABASE PRODUK (SOURCE OF TRUTH)
# ==========================================
# Ini mensimulasikan database SKU Anda. 
# KUNCI AKURASI: Nama produk di sini harus lengkap.
PRODUCT_DATABASE = {
    # --- THAI ---
    "TPA00001": "THAI PAPAYA LIGHT SOAP 130GR",
    "TPA00223": "THAI GOAT MILK & OLIVE OIL SOAP 100GR",
    "TH-OIL-ZAI": "THAI MINYAK ZAITUN (OLIVE OIL) 125ML", # Item yang sebelumnya salah mapping
    "TH-OIL-KEM": "THAI MINYAK KEMIRI 125ML",
    
    # --- JAVINCI ---
    "1200005": "BODY WHITE AHA GLUTA-HYA BRIGHT BODY TONE-UP SERUM 200ML (HITAM)",
    "1108.01": "BODY WHITE AHA GLUTA HYA UVBRIGHT BODY TONE-UP SERUM 100ML",
    
    # --- DIOSYS (Wajib lengkap variannya) ---
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

# Konversi DB ke list untuk matching
PRODUCT_NAMES = list(PRODUCT_DATABASE.values())
PRODUCT_MAP = {v: k for k, v in PRODUCT_DATABASE.items()} # Reverse map: Name -> SKU

# ==========================================
# 2. LOGIKA PARSING CERDAS
# ==========================================

class SmartPOParser:
    def __init__(self, db_names, db_map):
        self.db_names = db_names
        self.db_map = db_map

    def clean_text(self, text):
        """Membersihkan simbol bullet point dan spasi berlebih."""
        text = re.sub(r'^[-*•\s]+', '', text) # Hapus dash di awal
        return text.strip()

    def extract_quantity(self, text):
        """Mendeteksi qty dan bonus. Contoh: '12pcs (12+1)'"""
        # Pola regex untuk menangkap angka pcs dan kurung bonus
        qty_pattern = r'(\d+)\s*pcs'
        bonus_pattern = r'\((.*?)\)'
        
        qty_match = re.search(qty_pattern, text, re.IGNORECASE)
        bonus_match = re.search(bonus_pattern, text)
        
        qty = qty_match.group(1) if qty_match else None
        bonus = bonus_match.group(1) if bonus_match else ""
        
        # Bersihkan teks dari qty dan bonus untuk keperluan matching nama produk
        clean_name = re.sub(qty_pattern, '', text, flags=re.IGNORECASE)
        clean_name = re.sub(bonus_pattern, '', clean_name)
        
        return clean_name.strip(), qty, bonus

    def find_best_match(self, query):
        """Mencari produk paling mirip di database."""
        # Menggunakan token_set_ratio agar urutan kata tidak terlalu berpengaruh
        # Cutoff 70 berarti kemiripan harus minimal 70%
        match, score = process.extractOne(query, self.db_names, scorer=fuzz.token_set_ratio)
        
        if score > 75: # Ambang batas akurasi
            return match, score
        return None, score

    def process_po_text(self, raw_text):
        lines = raw_text.strip().split('\n')
        results = []
        
        current_header = "" # Menyimpan konteks (misal: "Diosys 100ml")
        outlet_name = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            # Baris pertama diasumsikan nama Outlet
            if i == 0:
                outlet_name = line
                continue
            
            # Abaikan baris tagihan
            if line.startswith("#"):
                continue

            # 1. Ekstrak Quantity & Bonus
            item_text, qty, bonus = self.extract_quantity(line)
            item_text = self.clean_text(item_text)

            # 2. Logika Penentuan Header vs Item
            # Jika TIDAK ada quantity, kemungkinan besar ini adalah HEADER (Kategori/Brand)
            if not qty:
                # Ini adalah Header baru (misal: "THAI", "Javinci", "Diosys 100ml")
                current_header = item_text
                continue 
            
            # 3. Membangun Query Pencarian
            # Jika baris ini punya Qty, kita gabungkan dengan Header untuk konteks
            # Contoh: Header="Diosys 100ml", Item="N.black" -> Query="Diosys 100ml N.black"
            
            # Cek khusus: Jika item_text sudah mengandung nama brand, mungkin tidak perlu header
            # Tapi untuk aman, gabungkan saja.
            search_query = f"{current_header} {item_text}"
            
            # 4. Cari Match di Database
            product_name, score = self.find_best_match(search_query)
            
            if product_name:
                sku = self.db_map[product_name]
                
                # Format Quantity output
                qty_display = f"{qty}pcs"
                if bonus:
                    qty_display += f" ({bonus})"
                
                results.append({
                    "sku": sku,
                    "product": product_name,
                    "qty_bonus": qty_display,
                    "original_input": line
                })
            else:
                # Fallback jika tidak ketemu (Manual Check)
                results.append({
                    "sku": "UNKNOWN",
                    "product": f"?? TIDAK DITEMUKAN: {search_query} ??",
                    "qty_bonus": line,
                    "original_input": line
                })

        return outlet_name, results

# ==========================================
# 3. CONTOH PENGGUNAAN (Sesuai Input Anda)
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

# Inisialisasi Parser
parser = SmartPOParser(PRODUCT_NAMES, PRODUCT_MAP)

# Proses Data
outlet, parsed_items = parser.process_po_text(input_po_sales)

# Tampilkan Hasil
print(f"PO: {outlet}")
print("-" * 80)
print(f"{'SKU':<12} | {'NAMA PRODUK':<55} | {'QTY'}")
print("-" * 80)

for item in parsed_items:
    print(f"{item['sku']:<12} | {item['product']:<55} | {item['qty_bonus']}")
