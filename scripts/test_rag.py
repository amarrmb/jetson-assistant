#!/usr/bin/env python3
"""Quick RAG test script."""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from jetson_speech.rag import RAGPipeline

rag = RAGPipeline("dota2")
print(f"Total chunks: {rag.count()}")

# Search for Kez
print("\n=== Search: Kez ===")
results = rag.retrieve("Kez hero", top_k=3)
for r in results:
    print(f"[{r['score']:.2f}] {r['content'][:100]}...")

# Search for newest heroes
print("\n=== Search: newest hero ===")
results = rag.retrieve("newest latest hero introduced", top_k=3)
for r in results:
    print(f"[{r['score']:.2f}] {r['content'][:100]}...")

# List all hero names
print("\n=== All heroes in RAG ===")
results = rag.store.collection.get(where={"type": "hero"}, include=["metadatas"])
heroes = sorted(set(m.get("hero_name", "?") for m in results["metadatas"]))
print(f"Found {len(heroes)} heroes")
print("Heroes:", ", ".join(heroes[:20]), "...")
