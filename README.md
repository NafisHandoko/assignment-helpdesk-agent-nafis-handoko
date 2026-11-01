# IT Helpdesk Assistant

Backend API asisten AI internal untuk tim People & IT Operations. Aplikasi ini menggabungkan dua kemampuan dalam satu sistem:

1. **RAG (Retrieval-Augmented Generation)** untuk menjawab pertanyaan kebijakan berdasarkan knowledge base.
2. **Workflow routing sederhana (mini-agent)** untuk mendeteksi ketika permintaan karyawan butuh aksi spesifik, seperti mengecek status tiket IT, lalu mengambil datanya langsung lewat tool call.

Seluruh sistem dibungkus sebagai backend API (FastAPI), dikelola dengan environment variables, dan siap dijalankan dalam container Docker.

Tugas ini adalah individual assignment sesi 17 **End-to-End RAG & AI Agent Development**: ReworkAcademy.

---

## Problem Statement

### Objective

Tim People & IT Operations sebuah perusahaan menerima ratusan pertanyaan repetitif dari karyawan setiap hari. Sebagian berupa pertanyaan kebijakan umum ("berapa jatah cuti saya?", "bagaimana cara reset password VPN?"), sebagian lagi permintaan spesifik yang butuh pengecekan data langsung ("apa status tiket TICKET-0042 saya?").

Tujuan tugas ini adalah membangun prototype backend API asisten AI internal yang dapat menangani kedua jenis permintaan tersebut secara otomatis dalam satu sistem yang sama:

1. Menjawab pertanyaan kebijakan berdasarkan knowledge base menggunakan RAG.
2. Mengenali ketika permintaan user butuh pengecekan data spesifik (status tiket), lalu mengambil data tersebut menggunakan tool call, bukan retrieval semantik.
3. Dikemas dalam container Docker agar siap diserahkan ke tim infrastruktur.

### Dua Jenis Permintaan User

| Jenis | Intent | Contoh | Penanganan |
|---|---|---|---|
| Pertanyaan informasi/kebijakan | `faq` | "berapa jatah cuti tahunan saya?" | RAG: semantic search ke knowledge base, lalu LLM menyusun jawaban |
| Permintaan cek status tiket | `ticket_status` | "apa status tiket TICKET-0005?" | Tool call: cari `ticket_id` di data mock tiket |

### Batasan dan Asumsi

- Sistem **hanya menjawab berdasarkan knowledge base** yang tersedia (`data/it_helpdesk_knowledge_base.csv`). Informasi di luar itu dianggap tidak tersedia.
- Data tiket berasal dari `data/mock_tickets.csv` (data dummy, bukan sistem tiket sungguhan).
- Knowledge base dan data tiket **tidak diubah** dari starter repository.
- Intent detection memakai LLM dengan baseline keyword sebagai jaring pengaman.
- Tidak ada autentikasi/otorisasi (prototype internal, bukan production).
- Aplikasi tidak di-deploy ke cloud; cukup dijalankan secara lokal dalam container.

---

## Desain Workflow

```
                    ┌─────────────────┐
                    │  Query dari user │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Intent detection │
                    │  (router.py)     │
                    └────────┬────────┘
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌──────────────────┐          ┌──────────────────┐
     │ intent = faq     │          │ intent =         │
     │                  │          │ ticket_status    │
     │ RAG (rag.py)     │          │ Tool call        │
     │ semantic search  │          │ (tools.py)       │
     │ ke knowledge     │          │ get_ticket_      │
     │ base + LLM       │          │ status(id)       │
     └────────┬─────────┘          └────────┬─────────┘
              └──────────────┬──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Response        │
                    │ {status, intent,│
                    │  data}          │
                    └─────────────────┘
```

Alur singkatnya: **query masuk → intent detection → RAG atau tool call → response keluar**.

### Kriteria Pembedaan Intent

Router menentukan intent dengan dua lapis:

1. **Lapis utama: LLM.** Query dikirim ke LLM dengan prompt terstruktur yang meminta klasifikasi ke salah satu label: `faq` (pertanyaan informasi/kebijakan umum) atau `ticket_status` (permintaan mengecek status tiket tertentu). LLM menjawab satu kata, lalu divalidasi.
2. **Lapis cadangan: keyword (jaring pengaman).** Jika LLM gagal merespons atau mengembalikan label di luar yang dikenal, router memakai pola kata kunci: query yang menyebut "tiket", "ticket", "nomor tiket", atau mengandung pola `TICKET-XXXX` diarahkan ke `ticket_status`; selain itu dianggap `faq`.

Alasan memakai dua lapis: baseline keyword memastikan sistem tetap berfungsi walaupun LLM error, sekaligus menjadi acuan untuk memeriksa kualitas keputusan LLM.

### Ekstraksi Ticket ID

Ketika intent terdeteksi `ticket_status`, router mengekstrak nomor tiket dengan regex `TICKET-\d{1,4}` (case-insensitive, dinormalisasi ke huruf besar). Jika query menyebut "tiket" tapi tidak ada nomornya, sistem meminta user menyebutkan format nomor tiket (`TICKET-XXXX`): tidak crash.

---

## Struktur Project

```
it-helpdesk-assignment/
├── app/
│   ├── __init__.py    # penanda package
│   ├── main.py        # entry point FastAPI, endpoint /ask & /health
│   ├── router.py      # intent detection + routing ke RAG / tool call
│   ├── rag.py         # embedding, index FAISS, semantic search, jawaban LLM
│   └── tools.py       # tool call: pencarian status tiket
├── data/
│   ├── it_helpdesk_knowledge_base.csv   # knowledge base (tidak diubah)
│   └── mock_tickets.csv                 # data mock tiket (tidak diubah)
├── Dockerfile
├── .env.example       # contoh konfigurasi environment variables
├── .gitignore         # melindungi .env agar tidak ter-commit
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Jalankan secara lokal (tanpa Docker)

```bash
# Buat virtual environment (disarankan)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependency
pip install -r requirements.txt

# Siapkan environment variables
cp .env.example .env
# lalu isi GEMINI_API_KEY (dan OPENROUTER_API_KEY jika ingin fallback)

# Jalankan server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Server berjalan di `http://localhost:8000`. Dokumentasi API interaktif otomatis tersedia di `http://localhost:8000/docs`.

### 2. Jalankan dengan Docker

```bash
# Build image
docker build -t it-helpdesk-assistant .

# Jalankan container, teruskan .env
docker run -p 8000:8000 --env-file .env it-helpdesk-assistant
```

Endpoint `/ask` dapat diakses dari luar container melalui port 8000 yang sudah di-expose.

### 3. Menguji endpoint

```bash
# Cek kesehatan service
curl http://localhost:8000/health

# Pertanyaan kebijakan (RAG)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "berapa jatah cuti tahunan saya?"}'

# Cek status tiket (tool call)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "apa status tiket TICKET-0005?"}'
```

Response selalu terstruktur:

```json
{
  "status": "success",
  "intent": "faq",
  "data": "Halo! Karyawan tetap mendapat jatah 12 hari cuti tahunan..."
}
```

---

## Hasil Pengujian

Pengujian dilakukan terhadap endpoint `/ask` dengan 5 skenario sesuai instruksi: 2 query RAG, 2 query tool call (termasuk 1 tiket yang tidak ada), dan 1 query ambigu.

| # | Query | Intent Terdeteksi | Komponen Tujuan | Hasil |
|---|---|---|---|---|
| 1 | "Bagaimana cara mengajukan cuti?" | `faq` | RAG | ✅ Sukses: menjelaskan prosedur lewat portal HR, 3 hari kerja, persetujuan atasan |
| 2 | "Apakah laptop kantor bisa dibawa pulang?" | `faq` | RAG | ✅ Sukses: menjawab jujur belum ada info, sarankan hubungi IT Support |
| 3 | "apa status tiket TICKET-0002?" | `ticket_status` | Tool call | ✅ Sukses: status, kategori, dan pembaruan terakhir ditampilkan |
| 4 | "status tiket TICKET-7777 gimana?" | `ticket_status` | Tool call | ✅ Sukses: tiket tidak ditemukan, tidak crash |
| 5 | "kamu bisa ngapain aja?" | `faq` | RAG | ✅ Sukses: menjelaskan kemampuan, tidak mengarang |

### Observasi

- **Router mengarahkan ke komponen yang tepat pada semua skenario yang jelas intent-nya.** Pertanyaan kebijakan (langsung maupun parafrase seperti "gimana cara reset password VPN ya?") masuk ke RAG; permintaan status tiket (termasuk `ticket-0003` huruf kecil) masuk ke tool call.
- **Edge case nomor tiket tidak ditemukan ditangani dengan baik.** Tool mengembalikan pesan informatif, aplikasi tetap merespons `status: success` dengan data berupa penjelasan "tidak ditemukan".
- **Query ambigu tidak membuat sistem mengarang.** Query di luar cakupan tetap diarahkan ke RAG (default aman), lalu LLM menjawab bahwa informasi tidak tersedia dan menyarankan menghubungi IT Support.
- **Query "tiket" tanpa nomor** (misal "status tiket saya bagaimana?") terdeteksi sebagai `ticket_status`, lalu router meminta user menyebutkan format nomor tiket: perilaku yang ramah dan tidak crash.

---

## Dukungan Dual-Provider API

Karena Gemini API tidak tersedia dari semua region (misalnya IP server cloud diblokir), aplikasi memakai pola dual-provider:

| Jalur | Embedding | LLM | Kapan dipakai |
|---|---|---|---|
| A. Gemini API langsung | `gemini-embedding-001` (768d) | `gemini-3.6-flash` | `GEMINI_API_KEY` tersedia dan region didukung (mis. laptop) |
| B. OpenRouter (fallback) | `google/gemini-embedding-001` (3072d) | `google/gemini-3.6-flash` | Gemini langsung gagal (mis. region server diblokir) |

Deteksi jalur dilakukan otomatis saat aplikasi startup. Satu jalur dipakai konsisten untuk seluruh proses (embed dokumen, embed query, generate jawaban) agar ruang vektor seragam. API key tidak pernah di-hardcode; semuanya lewat environment variables.

---

## Keterbatasan

1. **Data mock, bukan sistem nyata.** Status tiket diambil dari `mock_tickets.csv`, bukan tiket sungguhan. Integrasi dengan sistem tiket (mis. Jira/ServiceNow) belum ada.
2. **Retrieval bergantung pada model embedding.** Vektor dari model berbeda tidak kompatibel; kualitas pencarian mengikuti kualitas model yang dipakai.
3. **Intent detection berbasis LLM tidak 100% deterministik.** Jawaban LLM bisa bervariasi antar pemanggilan. Lapisan keyword mengurangi risiko salah arah, tapi tidak menghilangkan.
4. **Pencarian FAISS exact (`IndexFlatIP`).** Cepat untuk knowledge base kecil (ratusan entri), tapi kurang skalabel untuk jutaan dokumen.
5. **LLM bisa salah mengekstrak makna query ambigu.** Default aman (diarahkan ke RAG) mengurangi dampaknya, tapi untuk kasus tertentu jawaban bisa kurang tepat.
6. **Tidak ada mekanisme umpan balik.** Sistem belum mencatat apakah jawaban yang diberikan benar-benar membantu karyawan.