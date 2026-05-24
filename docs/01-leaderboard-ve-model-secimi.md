# Türkçe RAG: Leaderboard, Model Karşılaştırması ve Seçim

> Tarih: 2026-05-24
> Kaynak: TR-MTEB (EMNLP 2025), TurkishMMLU (EMNLP 2024), TR-MMLU, MMLU-Pro-TR, OpenLLMTurkishLeaderboard, TurkBench (SIGTURK 2026), Mursit teknik raporu

---

## 1. Türkçe için Geçerli Benchmarklar

| Benchmark | Yıl | Hedef | Görev Sayısı | Not |
|---|---|---|---|---|
| **TR-MTEB** | 2025 (EMNLP) | Embedding (cümle temsili) | 26 veri seti / 6 görev | Türkçe için ilk büyük ölçekli embedding benchmark'ı |
| **TurkishMMLU** | 2024 (EMNLP) | LLM bilgi/akıl yürütme | 10,032 soru / 9 ders | YKS müfredatı, native Türkçe |
| **TR-MMLU** | 2025 | LLM bilgi | 6,200 soru / 62 kategori | TUS/KPSS/AÖF dahil |
| **MMLU-Pro-TR** | 2024 | LLM zor akıl yürütme | 12,000 soru / 14 alan | 10 şıklı zor versiyon |
| **TurkBench** | 2026 (SIGTURK) | LLM çoklu yetenek | 8,151 örnek / 24 alt görev | Insan doğrulamalı, içerik moderasyonu dahil |
| **CETVEL** | 2025 | LLM üretken+ayırt edici | 23 görev | Sadece MCQ değil, generative dahil |

---

## 2. Türkçe Embedding Modelleri — TR-MTEB Liderlik Tablosu

| Sıra | Model (HF) | Param | TR-MTEB Skoru | Boyut (vec) | Max Token | Lisans | VRAM (FP16) |
|---|---|---|---|---|---|---|---|
| **1** | `newmindai/Mursit-Large-TR-Retrieval` | 403M | **56.87** | 1024 | 2048 | Apache 2.0 | ~1.5 GB |
| 2 | `newmindai/Mursit-Base-TR-Retrieval` | 155M | 55.86 | 768 | 2048 | Apache 2.0 | ~0.7 GB |
| 3 | `newmindai/Mursit-Embed-Qwen3-4B-TR` | 4B | 53.65 | 2560 | 1024 | Apache 2.0 | ~8.5 GB |
| 4 | `BAAI/bge-m3` | 568M | ~52* | 1024 | 8192 | MIT | ~2 GB |
| 5 | `intfloat/multilingual-e5-large-instruct` | 560M | ~51* | 1024 | 514 | MIT | ~2 GB |
| 6 | `trmteb/*` (10 model varyantı) | 100M-300M | 48-53 | 768/1024 | 512 | Açık | ~0.5-1.5 GB |
| 7 | `BAAI/bge-m3-unsupervised` | 568M | ~50* | 1024 | 8192 | MIT | ~2 GB |
| 8 | `sentence-transformers/LaBSE` | 471M | ~46* | 768 | 512 | Apache 2.0 | ~1.8 GB |

\* Multilingual modellerin Türkçe alt skorları (resmi TR-MTEB değil, MMTEB Turkish task ortalaması)

### Genel-Amaçlı Karşılaştırma (MTEB Global 2026)

| Model | MTEB Global | Açık Kaynak | Multilingual |
|---|---|---|---|
| Qwen3-Embedding-8B | 70.6 | Evet (Apache 2.0) | Evet |
| Google Gemini Embedding | 68.3 | Hayır (API) | Evet |
| gte-Qwen3-8B | 68.1 | Evet | Evet |
| NVIDIA NV-Embed-v2 | 67.5 | Evet | Evet |
| BGE-M3 | 63.2 | Evet (MIT) | Evet (100+ dil) |
| OpenAI text-embedding-3-large | 64.6 | Hayır | Evet |

> **Karar:** Türkçe-özel görevlerde **Mursit-Large-TR-Retrieval** lider. Çok dilli + 8K bağlam isteniyorsa **BAAI/bge-m3**. Maksimum kalite (donanım izin veriyorsa) **Qwen3-Embedding-8B**.

---

## 3. Türkçe Reranker Modelleri

| Model (HF) | Tip | Max Token | Base | Lisans | VRAM |
|---|---|---|---|---|---|
| `seroe/bge-reranker-v2-m3-turkish-triplet` | Cross-Encoder | 8192 | BGE-M3 (568M) | MIT | ~2 GB |
| `99eren99/ColBERT-ModernBERT-base-Turkish-uncased` | Late Interaction | 8192 | ModernBERT-base | Açık | ~1 GB |
| `seroe/jina-reranker-v2-base-multilingual-turkish-triplet_v1` | Cross-Encoder | 1024 | Jina-base (278M) | CC-BY-NC | ~1 GB |
| `BAAI/bge-reranker-v2-m3` (çok dilli baseline) | Cross-Encoder | 8192 | BGE-M3 | MIT | ~2 GB |

> **Karar:** Standart cross-encoder için **`seroe/bge-reranker-v2-m3-turkish-triplet`**, prodüksiyon ölçeği için **TurkColBERT** (MaxSim daha hızlı).

---

## 4. Türkçe LLM Modelleri — TurkishMMLU & MMLU-Pro-TR Tablosu

### 4.1 Açık Kaynak (Self-hosted)

| Model (HF) | Param | TurkishMMLU | MMLU-TR | MMLU-Pro-TR | Base | Lisans | VRAM FP16 / Q4 |
|---|---|---|---|---|---|---|---|
| `Trendyol/Trendyol-LLM-8B-T1` | 8B | ~62-65* | ~60* | n/a | Qwen3-8B | **Apache 2.0** | 17 GB / **5.5 GB** |
| `ogulcanaydogan/Turkish-LLM-14B-Instruct` | 14B | **61.33** | **59.77** | n/a | Qwen2.5-14B | Apache 2.0 | 29 GB / 10.5 GB |
| `WiroAI/wiroai-turkish-llm-9b` | 9B | n/a | **59.8** | n/a | Llama 3.1 9B | Llama 3.1 | 19 GB / 6.5 GB |
| `ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1` | 8B | n/a | 52.0 | n/a | Llama 3 8B | Llama 3 | 16 GB / 5.5 GB |
| `Metin/LLaMA-3-8B-Instruct-TR-DPO` | 8B | n/a | 49.71 | 27.00 | Llama 3 8B | Llama 3 | 16 GB / 5.5 GB |
| `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` | 8B | n/a | 51.75 | 23.90 | Llama 3 8B | Llama 3 | 16 GB / 5.5 GB |
| `VeriUS/VeriUS-LLM-8b-v0.2` | 8B | n/a | 48.81 | 23.23 | Llama 3 8B | Llama 3 | 16 GB / 5.5 GB |
| `Orbina/Orbita-v0.1` | 8B | n/a | 49.51 | 22.95 | Llama 3 8B | Llama 3 | 16 GB / 5.5 GB |
| `CohereForAI/aya-23-8B` | 8B | 45.0 | n/a | n/a | Aya/Command | **CC-BY-NC** | 16 GB / 5.5 GB |
| `CohereForAI/aya-23-35B` | 35B | 55.6 | n/a | n/a | Aya/Command | CC-BY-NC | 70 GB / 22 GB |
| `Trendyol/Trendyol-LLM-7b-chat-v1.8` | 7B | n/a | 41.91 | 18.15 | Llama 2 | Apache 2.0 | 14 GB / 4.5 GB |
| `KOCDIGITAL/Kocdigital-LLM-8b-v0.1` | 8B | n/a | 47.35 | 21.83 | Llama 3 8B | Llama 3 | 16 GB / 5.5 GB |
| `Hamza-xlarge` (KUIS-AI, sıfırdan eğitim) | 1.3B | düşük | düşük | n/a | GPT-2 | Açık | 3.5 GB / 1.5 GB |

\* Trendyol-LLM-8B-T1 için resmi TurkishMMLU yok ama Qwen3-8B base + Türkçe fine-tune + /think mode reasoning ile en üst kategoride.

### 4.2 Karşılaştırma (Kapalı Kaynak Referans)

| Model | TurkishMMLU |
|---|---|
| GPT-4o | 83.1 |
| Claude-3 Opus | 79.1 |
| GPT-4-turbo | 75.7 |
| Llama-3 70B-IT | 67.3 (en iyi açık) |
| Claude-3 Haiku | 65.4 |

---

## 5. Nihai Seçim ve Gerekçesi

**Seçim:**

| Bileşen | Model | Gerekçe |
|---|---|---|
| **Embedder** | `newmindai/Mursit-Large-TR-Retrieval` | TR-MTEB lideri (56.87), Apache 2.0, sadece 403M, 2K bağlam, Türkçe-özel sıfırdan eğitim |
| **Reranker** | `seroe/bge-reranker-v2-m3-turkish-triplet` | 8K bağlam, MIT, Türkçe triplet fine-tune |
| **LLM** | `Trendyol/Trendyol-LLM-8B-T1` | Apache 2.0, Qwen3-8B base (SOTA), 32K bağlam, `/think` modu (RAG için kritik), GGUF mevcut |

**Alternatif (akademik ağırlıklı):** Cosmos `Turkish-Llama-8b-DPO-v0.1` (akademik ekibe ait, Llama 3 lisansı kabul edilebiliyorsa).

**Alternatif (en yüksek MMLU skoru):** `ogulcanaydogan/Turkish-LLM-14B-Instruct` (61.33 TurkishMMLU) — fakat 14B = daha fazla VRAM.

---

## 6. Donanım Senaryoları

| Senaryo | Donanım | Konfigürasyon | Toplam VRAM |
|---|---|---|---|
| **Üretim — Tam doğruluk** | RTX 3090/4090 (24GB) | Mursit FP16 + Reranker FP16 + LLM BF16 | ~20 GB |
| **Geliştirme — Dengeli** | RTX 3060 / 4060Ti (12-16GB) | Mursit FP16 + Reranker FP16 + LLM GGUF Q4 | ~9-10 GB |
| **Laptop / Düşük VRAM** | 8GB GPU + 16GB RAM | Mursit FP16 (GPU) + LLM GGUF Q4 (CPU+GPU offload) | 2-3 GB GPU |
| **CPU-only** | 32GB RAM, no GPU | Mursit ONNX (CPU) + LLM GGUF Q4 (llama.cpp) | 0 VRAM, ~10 GB RAM |

---

## 7. Kaynaklar

- TR-MTEB (Baysan et al., 2025, EMNLP Findings): https://aclanthology.org/2025.findings-emnlp.471/
- TurkishMMLU (Yüksel et al., 2024, EMNLP Findings): https://aclanthology.org/2024.findings-emnlp.413/
- TR-MMLU (Bayram et al., 2024): https://arxiv.org/abs/2501.00593
- MMLU-Pro-TR: https://github.com/bezir/MMLU-pro-TR
- TurkBench (Toraman et al., 2026, SIGTURK)
- AI Turkish MMLU Leaderboard: https://huggingface.co/datasets/alibayram/yapay_zeka_turkce_mmlu_liderlik_tablosu
- OpenLLM Turkish Leaderboard: https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard
- Mursit modelleri: https://huggingface.co/newmindai
- Trendyol modelleri: https://huggingface.co/Trendyol
- Cosmos modelleri: https://huggingface.co/ytu-ce-cosmos
