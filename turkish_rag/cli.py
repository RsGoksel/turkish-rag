"""Turkish-RAG CLI."""

from __future__ import annotations
import argparse
import json

from .config import Config
from .retriever import Retriever


def cmd_index(args):
    r = Retriever(Config.default())
    docs = _load_jsonl(args.input)
    n = r.index_documents(docs)
    print(f"Indekslendi: {len(docs)} dokuman, {n} chunk.")


def cmd_query(args):
    r = Retriever(Config.default())
    for i, c in enumerate(r.retrieve(args.query, k=args.k), 1):
        src = c.meta.get("source", c.meta.get("doc_id", "-"))
        print(f"[{i}] sim={c.similarity:+.3f}  src={src}")
        print(f"    {c.text[:200]}{'...' if len(c.text) > 200 else ''}")


def cmd_demo(args):
    from .demo_data import DEMO_DOCS
    r = Retriever(Config.default())
    if r.store.count() == 0:
        n = r.index_documents(DEMO_DOCS)
        print(f"Demo veri indekslendi: {n} chunk.\n")
    sorular = [
        "Is sozlesmesi feshi icin bildirim suresi ne kadardir?",
        "KVKK ne zaman yururluge girdi?",
        "Anonim sirket sermayesi ne kadar olmali?",
    ]
    for q in sorular:
        print("=" * 80)
        print(f"SORGU: {q}")
        for i, c in enumerate(r.retrieve(q, k=3), 1):
            src = c.meta.get("source", "-")
            print(f"  [{i}] sim={c.similarity:+.3f}  src={src}")
            print(f"      {c.text[:200]}{'...' if len(c.text) > 200 else ''}")
        print()


def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser(prog="turkish-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("index", help="JSONL dosyasindan indeksle")
    sp.add_argument("input", type=str, help="Her satiri {id, text, meta?} olan JSONL")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("query", help="Sorgu calistir")
    sp.add_argument("query", type=str)
    sp.add_argument("-k", type=int, default=5)
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("demo", help="Demo veri ile calistir")
    sp.set_defaults(func=cmd_demo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
