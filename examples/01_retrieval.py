"""Türkçe retrieval örneği.

Calistir:
    python examples/01_retrieval.py
"""

from turkish_rag import Retriever, Config
from turkish_rag.demo_data import DEMO_DOCS


def main():
    r = Retriever(Config.default())

    if r.store.count() == 0:
        n = r.index_documents(DEMO_DOCS)
        print(f"[indexed] {n} chunk\n")

    sorular = [
        "Is sozlesmesi feshi icin bildirim suresi ne kadardir?",
        "KVKK ne zaman yururluge girdi?",
        "Anonim sirket sermayesi ne kadar olmali?",
    ]
    for q in sorular:
        print("=" * 80)
        print(f"SORGU: {q}")
        for i, c in enumerate(r.retrieve(q, k=3), 1):
            print(f"  [{i}] sim={c.similarity:+.3f}  src={c.meta.get('source')}")
            print(f"      {c.text[:200]}{'...' if len(c.text) > 200 else ''}")
        print()


if __name__ == "__main__":
    main()
