# Turkish RAG

Türkçe için sade, açık kaynak **retrieval** kütüphanesi. Sadece embedding + vektör arama; LLM yok.

LLM ile tam pipeline gerekiyorsa: [Turkish-LLM-RAG](https://github.com/RsGoksel/Turkish-LLM-RAG)

## Stack

| Bileşen | Model | Skor | Lisans |
|---|---|---|---|
| Embedder | `newmindai/Mursit-Large-TR-Retrieval` | TR-MTEB 56.87 (lider) | Apache 2.0 |
| Vektör DB | ChromaDB (HNSW cosine) | - | Apache 2.0 |

Tam karşılaştırma: [docs/01-leaderboard-ve-model-secimi.md](docs/01-leaderboard-ve-model-secimi.md)

## Kurulum

```powershell
git clone https://github.com/RsGoksel/Turkish-RAG
cd Turkish-RAG

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
```

Linux/Mac:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -e ".[all]"
```

## Model İndirme

```powershell
pip install hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"

hf download newmindai/Mursit-Large-TR-Retrieval --local-dir models/Mursit-Large-TR-Retrieval
```

Boyut: ~1.5 GB. 18 Mbps bağlantıda ~11 dakika.

## Hızlı Başlangıç

```python
from turkish_rag import Retriever, Config

r = Retriever(Config.default())

r.index_documents([
    {"id": "d1", "text": "4857 sayılı İş Kanunu işçi-işveren ilişkilerini düzenler."},
    {"id": "d2", "text": "6698 sayılı KVKK 7 Nisan 2016'da yürürlüğe girdi."},
])

for c in r.retrieve("İş kanunu kaç sayılıdır?", k=2):
    print(f"sim={c.similarity:+.3f}  {c.text}")
```

## CLI

```bash
python -m turkish_rag.cli demo
python -m turkish_rag.cli index data/docs.jsonl
python -m turkish_rag.cli query "Sözleşme feshi nasıl yapılır?"
```

JSONL formatı (her satır):
```json
{"id": "doc1", "text": "Metin...", "meta": {"source": "kaynak"}}
```

## HTTP API

```bash
uvicorn turkish_rag.api:app --host 0.0.0.0 --port 8000
```

Endpoint'ler: `POST /index`, `POST /query`, `GET /health`.

## Mimari

```
turkish_rag/
├── config.py           Config (embedder, vector_store, chunk, top_k)
├── embedder.py         Mursit wrapper (sentence-transformers)
├── chunking.py         word / sentence chunking
├── vector_store.py     ChromaDB wrapper
├── retriever.py        orchestrator (index + retrieve)
├── cli.py              komut satırı
├── api.py              FastAPI
└── demo_data.py        Türkçe hukuk corpus
```

## Test

```bash
pytest tests/test_chunking.py -v
```

## Çıktı Üretimi (LLM Eklemek)

Bu repo retrieve edilen chunk'ları döner. Bir LLM'e yedirip cevap üretmek istiyorsan, döndürülen `RetrievedChunk` listesini istediğin modele bağlam olarak ver:

```python
chunks = r.retrieve("Sorgu")
baglam = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
# OpenAI, Ollama, llama.cpp, transformers... istediğin LLM ile çağır.
```

Hazır LLM entegrasyonu (Trendyol, Cosmos, Ollama, OpenAI adapter'ları) için [Turkish-LLM-RAG](https://github.com/RsGoksel/Turkish-LLM-RAG).

## Lisans

Apache 2.0.

## Atıflar

```bibtex
@inproceedings{baysan2025trmteb,
  title={TR-MTEB: A Comprehensive Benchmark and Embedding Model Suite for Turkish Sentence Representations},
  author={Baysan, M. S. and Bebek, I. and Güngör, T.},
  booktitle={Findings of EMNLP 2025},
  year={2025}
}
```
