# Turkish RAG

Türkçe metinler üzerinde anlamsal arama (semantic retrieval) ve doküman geri çağırma (document retrieval) için kütüphanedir. 
Çekirdek: TR-MTEB lideri **Mursit-Large-TR-Retrieval** + **ChromaDB**. Üretim üzerine inşa edilebilir; lokal LLM, agent veya analiz pipeline ile birleştirilebilir.

Tam pipeline (LLM dahil hazır kurulum) gerekiyorsa: [Turkish-LLM-RAG](https://github.com/RsGoksel/Turkish-LLM-RAG)

**Türkçe doküman tabanları** üzerinde sorgu-doküman eşleştirme (sözleşme arama, mevzuat arama, müşteri destek bilgi tabanı)
- **Semantik benzerlik** ölçümü (kelime eşleşmesinden bağımsız anlam eşleşmesi)
- **RAG pipeline'larının** retrieval katmanı (LLM bileşeniyle birleştirmek üzere)
- **Sınıflandırma / kümeleme** için cümle/paragraf embedding'leri
- **Çoğaltma tespiti (deduplication)** ve içerik kümelemesi

 (Baysan et al., EMNLP 2025) üzerinde lider olması sebebiyle Türkçe için resmi benchmark olan **TR-MTEB** ele alınmıştır.

| Model | TR-MTEB | Parametre | Bağlam |  |
|---|---|---|---|---|
| **Mursit-Large-TR-Retrieval** | **56.87** | 403M | 2048 token | |
| Mursit-Base-TR-Retrieval | 55.86 | 155M | 2048 token |  |
| BAAI/bge-m3 (çok dilli) | ~52* | 568M | 8192 token |  |
| multilingual-e5-large-instruct | ~51* | 560M | 514 token | |

\* Türkçe alt görev ortalaması 

Mursit, ModernBERT-large mimarisinde 112.7B token Türkçe-ağırlıklı korpus üzerinde sıfırdan ön eğitilmiş; Türkçenin sondan eklemeli morfolojisi için 59K kelimelik özel tokenizer kullanır. 

Tam karşılaştırma için bkz: [docs/01-leaderboard-ve-model-secimi.md](docs/01-leaderboard-ve-model-secimi.md)

---

## Kurulum

### 1. Repo'yu klonla

```powershell
git clone https://github.com/RsGoksel/Turkish-RAG
cd Turkish-RAG
```

### 2. Sanal ortam

Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / Mac:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. PyTorch

GPU (NVIDIA, önerilen):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

CPU-only (sunucu/laptop):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Paket

```bash
pip install -e ".[all]"
```

`all` extras: API (FastAPI/uvicorn) + dev (pytest).

## Model İndirme

Mursit-Large-TR-Retrieval (~1.5 GB):

```powershell
pip install hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"

hf download newmindai/Mursit-Large-TR-Retrieval --local-dir models/Mursit-Large-TR-Retrieval
```

Linux/Mac'te `$env:...` yerine `export HF_HUB_ENABLE_HF_TRANSFER=1`.

İndirme süresi: 18 Mbps'de ~11 dakika, 4.5 Mbps'de ~45 dakika.

Doğrulama:
```bash
python -c "from pathlib import Path; p=Path('models/Mursit-Large-TR-Retrieval/model.safetensors'); print(f'OK: {p.stat().st_size/1e9:.2f} GB' if p.exists() else 'BULUNAMADI')"
```

---

## Temel Kavramlar

**Chunk:** Doküman, embedder'ın bağlam penceresine (Mursit: 2048 token) sığacak parçalara bölünür. Tipik chunk 350-500 kelime, %15-20 örtüşme (overlap) ile.

**Embedding:** Her chunk, 1024 boyutlu yoğun vektöre dönüştürülür. Benzer anlamlı metinler vektör uzayında birbirine yakındır.

**Vektör DB:** Embedding'ler ChromaDB'ye yazılır; sorguda HNSW (Hierarchical Navigable Small World) algoritması ile yaklaşık en yakın komşu (ANN) araması yapılır.

**Similarity (benzerlik):** Cosine similarity. Normalize edilmiş vektörler için iki vektörün skaler çarpımına eşittir. `+1.0` = aynı yön, `0` = ortogonal (alakasız), `-1.0` = zıt.

**Distance:** ChromaDB cosine distance döndürür: `distance = 1 - similarity`. Yani `distance=0.2` → `similarity=0.8`.

---

## Hızlı Başlangıç

```python
from turkish_rag import Retriever, Config

retriever = Retriever(Config.default())

retriever.index_documents([
    {"id": "anayasa", "text": "Türkiye Cumhuriyeti Anayasası 1982 yılında halkoylaması ile kabul edildi."},
    {"id": "is_kanunu", "text": "4857 sayılı İş Kanunu işçi-işveren ilişkilerini düzenler."},
    {"id": "kvkk", "text": "6698 sayılı KVKK 7 Nisan 2016'da yürürlüğe girdi."},
])

for chunk in retriever.retrieve("KVKK ne zaman yürürlüğe girdi?", k=2):
    print(f"sim={chunk.similarity:+.3f}  doc={chunk.meta['doc_id']}")
    print(f"  {chunk.text[:100]}")
```

Beklenen çıktı:
```
sim=+0.686  doc=kvkk
  6698 sayılı KVKK 7 Nisan 2016'da yürürlüğe girdi.
sim=+0.250  doc=anayasa
  Türkiye Cumhuriyeti Anayasası 1982 yılında halkoylaması ile kabul edildi.
```

---

## Belge Yükleme

### Tek tek (kod)

```python
docs = [
    {"id": "d1", "text": "metin...", "meta": {"source": "wiki", "year": 2024}},
    {"id": "d2", "text": "başka metin...", "meta": {"source": "blog"}},
]
retriever.index_documents(docs)
```

`meta` opsiyonel, herhangi bir JSON-serileştirilebilir alan tutabilir; sonradan filtreleme için kullanılır.

### JSONL (toplu)

`data/docs.jsonl`:
```jsonl
{"id": "doc1", "text": "Birinci doküman metni...", "meta": {"source": "MEB"}}
{"id": "doc2", "text": "İkinci doküman...", "meta": {"source": "Resmi Gazete"}}
```

```bash
python -m turkish_rag.cli index data/docs.jsonl
```

### Klasördeki .txt / .md dosyaları

```python
from pathlib import Path
from turkish_rag import Retriever, Config

r = Retriever(Config.default())
docs = [
    {"id": p.stem, "text": p.read_text(encoding="utf-8"), "meta": {"path": str(p)}}
    for p in Path("data/corpus").rglob("*.txt")
]
r.index_documents(docs)
```

### PDF (örnek)

```python
import pypdf
docs = []
for p in Path("data/pdfs").glob("*.pdf"):
    reader = pypdf.PdfReader(str(p))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    docs.append({"id": p.stem, "text": text, "meta": {"path": str(p)}})
r.index_documents(docs)
```

### Chunking ayarları

```python
cfg = Config.default()
cfg.chunk.chunk_size = 350         # kelime
cfg.chunk.chunk_overlap = 70       # %20
cfg.chunk.split_by = "sentence"    # "word" | "sentence"
retriever = Retriever(cfg)
```

| Parametre | Önerilen | Not |
|---|---|---|
| `chunk_size` | 350-512 kelime | Mursit max=2048 token, ~400 kelime sweet-spot |
| `chunk_overlap` | %15-20 | Cümle sonlarının kesilmemesi için |
| `split_by` | "sentence" | Hukuki/akademik metinde daha iyi |
| `top_k` | 3-10 | Downstream kullanımına göre |

---

## Retrieval

### Temel sorgu

```python
results = retriever.retrieve("Sorgu metni", k=5)
for c in results:
    print(c.similarity, c.text, c.meta)
```

### Geri dönen `RetrievedChunk`

| Alan | Tip | Açıklama |
|---|---|---|
| `text` | `str` | Eşleşen chunk metni |
| `meta` | `dict` | `{doc_id, chunk_idx, ...sizin meta'larınız}` |
| `distance` | `float` | Cosine distance, ChromaDB'den ham değer |
| `similarity` | `float` | `1 - distance`, kullanım kolaylığı için |

### Eşik (threshold) filtreleme

Düşük benzerlikteki sonuçları at:

```python
results = [c for c in retriever.retrieve(q, k=10) if c.similarity >= 0.5]
```

Eşik değerleri için pratik kılavuz (Mursit):

| Similarity | Yorum |
|---|---|
| > 0.75 | Çok güçlü eşleşme, neredeyse paragraf |
| 0.55 - 0.75 | İyi eşleşme, alakalı |
| 0.30 - 0.55 | Zayıf eşleşme, konu yakın |
| < 0.30 | Alakasız büyük olasılıkla |

### Metadata filtreleme

```python
# ChromaDB native where filter
chunks = retriever.store.query(
    query_embedding=retriever.embedder.encode("sorgu")[0].tolist(),
    k=5,
    where={"source": "Resmi Gazete"},
)
```

---

## Benzerlik Metrikleri ve Değerlendirme

### Tek sorgu için benzerlik dağılımı

```python
results = retriever.retrieve("Sorgu", k=10)
for i, c in enumerate(results, 1):
    print(f"{i:2d}. sim={c.similarity:+.3f}  {c.text[:60]}")
```

Sağlıklı bir corpus'ta similarity skorları monoton azalmalıdır. İlk-1 ile ilk-2 arasında büyük fark (örn. 0.80 → 0.45) güçlü eşleşmenin işareti.

### Recall@k, MRR (test seti üzerinde)

```python
test_set = [
    {"query": "İş sözleşmesi feshi süresi", "gold_doc_id": "is_kanunu"},
    {"query": "KVKK ne zaman yürürlüğe girdi?", "gold_doc_id": "kvkk"},
    # ...
]

def recall_at_k(retriever, test_set, k=5):
    hits = sum(
        any(c.meta["doc_id"] == t["gold_doc_id"] for c in retriever.retrieve(t["query"], k=k))
        for t in test_set
    )
    return hits / len(test_set)

def mrr(retriever, test_set, k=10):
    total = 0.0
    for t in test_set:
        for rank, c in enumerate(retriever.retrieve(t["query"], k=k), 1):
            if c.meta["doc_id"] == t["gold_doc_id"]:
                total += 1.0 / rank
                break
    return total / len(test_set)

print(f"Recall@5: {recall_at_k(retriever, test_set, 5):.2%}")
print(f"MRR@10:   {mrr(retriever, test_set, 10):.3f}")
```

### Cosine similarity matrisi (ad-hoc analiz)

```python
import numpy as np
sentences = ["Türkçe metin 1", "Benzer Türkçe metin", "Alakasız konu"]
embs = retriever.embedder.encode(sentences)   # (3, 1024), normalize edilmiş
sim_matrix = embs @ embs.T                    # cosine similarity = dot product
print(np.round(sim_matrix, 3))
```

---

## CLI

```bash
# Demo veri ile çalıştır
python -m turkish_rag.cli demo

# JSONL dosyasından indeksle
python -m turkish_rag.cli index data/docs.jsonl

# Sorgu çalıştır
python -m turkish_rag.cli query "Sözleşme feshi nasıl yapılır?" -k 5
```

## HTTP API

```bash
uvicorn turkish_rag.api:app --host 0.0.0.0 --port 8000
```

Endpoint'ler:

| Method | Path | Body | Döner |
|---|---|---|---|
| GET | `/health` | - | `{status, indexed_chunks}` |
| POST | `/index` | `{documents:[{id,text,meta?}]}` | `{indexed_chunks, total}` |
| POST | `/query` | `{query, k}` | `{query, chunks:[{text, meta, similarity}]}` |

Örnek:
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"id":"d1","text":"İçerik..."}]}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Sorgu","k":3}'
```

---

## Mimari

```
turkish_rag/
├── config.py            Config (embedder + store + chunking + top_k)
├── embedder.py          Mursit wrapper (lazy load, device auto-detect)
├── chunking.py          word / sentence chunking
├── vector_store.py      ChromaDB wrapper (HNSW cosine)
├── retriever.py         Orchestrator: index_documents + retrieve
├── cli.py               python -m turkish_rag.cli
├── api.py               FastAPI uygulaması
└── demo_data.py         Türkçe hukuk corpus (6 doküman)
```

Bağımlılıklar: `torch`, `sentence-transformers`, `chromadb`, `numpy`, `huggingface_hub`. API extras: `fastapi`, `uvicorn`, `pydantic`.

---

## Donanım

| Senaryo | Donanım | Notes |
|---|---|---|
| GPU üretim | RTX 3060+ / 4060+ / A100 | Embedding: <50ms / 32 chunk |
| CPU üretim | 8+ çekirdek, 16+ GB RAM | Embedding: ~1-2s / 32 chunk |
| Edge / laptop | Herhangi modern CPU | Yavaş ama çalışır |

VRAM: Mursit-Large FP16 ~1.5 GB. CPU'da float32 ~3 GB RAM.

---

## İleri Kullanım

### Cross-encoder reranker ile hassasiyet artırma

`turkish_rag` salt bi-encoder (Mursit) ile çalışır. Recall'ı yüksek tutup precision'ı artırmak için top-K sonuçları bir reranker ile yeniden sırala:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("seroe/bge-reranker-v2-m3-turkish-triplet", max_length=8192)

# 1) Geniş retrieval
candidates = retriever.retrieve("Sorgu", k=20)
# 2) Yeniden sırala
pairs = [("Sorgu", c.text) for c in candidates]
scores = reranker.predict(pairs)
reranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)]
top5 = reranked[:5]
```

Reranker indirme: `hf download seroe/bge-reranker-v2-m3-turkish-triplet --local-dir models/bge-reranker`

### Hybrid retrieval (dense + BM25)

```python
# pip install rank-bm25
from rank_bm25 import BM25Okapi

# Indexleme aşamasında BM25 paralel
tokenized = [c.split() for c in all_chunks]
bm25 = BM25Okapi(tokenized)

# Sorgu
q_tokens = "Sorgu metni".split()
bm25_scores = bm25.get_scores(q_tokens)
dense_results = retriever.retrieve("Sorgu metni", k=20)
# İki skoru normalize edip ağırlıklı topla, sırala
```

---

## Test

```bash
pytest tests/test_chunking.py -v
```

Smoke testi (Mursit indirilmiş olmalı):
```python
from turkish_rag import Retriever, Config
from turkish_rag.demo_data import DEMO_DOCS

r = Retriever(Config.default())
r.index_documents(DEMO_DOCS)
assert r.retrieve("KVKK ne zaman?", k=1)[0].meta["doc_id"] == "kvkk_6698"
```

---

## Sık Hatalar

| Hata | Çözüm |
|---|---|
| `FileNotFoundError: Embedder model bulunamadi` | Mursit'i indir (yukarıda) |
| Benzerlik skorları çok düşük | `Config.embedder.normalize=True` olmalı (varsayılan) |
| Türkçe karakter bozuluyor (Windows) | `chcp 65001` veya `$env:PYTHONIOENCODING="utf-8"` |
| `ConnectionResetError` paket indirirken | `pip install ... --retries 5 --timeout 300` |
| CUDA wheel yok (Python 3.13) | `cu121` yerine `cu124` index URL'i kullan |

---

## Belgeler

- [Türkçe embedding modelleri leaderboard'u (TR-MTEB)](docs/01-leaderboard-ve-model-secimi.md)


  year   = {2025}
}
```

İlgili: [Turkish-LLM-RAG](https://github.com/RsGoksel/Turkish-LLM-RAG) — bu kütüphane üzerine LLM (Trendyol-LLM-8B-T1, Ollama, OpenAI) entegrasyonu eklenmiş tam pipeline.
