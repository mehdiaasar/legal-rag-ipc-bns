"""Logic tests for the demo's retrieval layer.

Uses a deterministic stub embedding so the suite runs offline, without the
sentence-transformers download or the real 30k-chunk index.
"""

import tempfile

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import rag


class WordOverlapEmbeddings(Embeddings):
    """Bag-of-words vector over a fixed vocabulary — no network, but real ranking."""

    VOCAB = ["theft", "trust", "murder", "punishment", "property", "dishonestly"]

    def _vec(self, text):
        words = text.lower().split()
        return [float(words.count(term)) for term in self.VOCAB]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_store(tmpdir):
    docs = [
        Document(
            page_content="Whoever commits theft shall be punished with imprisonment. theft theft",
            metadata={"statute": "BNS", "source": "bns.pdf", "page": 12},
        ),
        Document(
            page_content="Theft is defined where a person dishonestly takes movable property. theft",
            metadata={"statute": "IPC", "source": "ipc.pdf", "page": 44},
        ),
        Document(
            page_content="Criminal breach of trust entails dishonest misappropriation. trust trust",
            metadata={"statute": "IPC", "source": "ipc.pdf", "page": 90},
        ),
        Document(
            page_content="Punishment for criminal breach of trust under the Sanhita. trust punishment",
            metadata={"statute": "BNS", "source": "bns.pdf", "page": 30},
        ),
        Document(
            page_content="Murder carries the gravest punishment. murder",
            metadata={"statute": "IPC", "source": "ipc.pdf", "page": 61},
        ),
    ]
    return Chroma.from_documents(
        documents=docs,
        embedding=WordOverlapEmbeddings(),
        collection_name=rag.COLLECTION,
        persist_directory=tmpdir,
    )


def test_bns_leads_the_context(store):
    passages = rag.retrieve(store, "theft punishment", k=4, bns_floor=2)
    assert passages, "expected hits for a term present in both statutes"
    assert passages[0].statute == "BNS", (
        f"BNS priority violated: first hit is {passages[0].statute}"
    )
    statutes = [p.statute for p in passages]
    # every BNS hit must precede every IPC hit
    assert statutes == sorted(statutes, key=lambda s: s != "BNS"), statutes
    print(f"  ordering: {statutes}")


def test_bns_floor_pulls_in_bns_hits(store):
    # 'murder' appears only in the IPC doc, so a plain search returns IPC only;
    # the BNS-only second search must still contribute BNS passages.
    passages = rag.retrieve(store, "murder", k=5, bns_floor=2)
    assert any(p.statute == "BNS" for p in passages), (
        "BNS floor failed to inject BNS passages into an IPC-dominated result"
    )
    print(f"  statutes: {[p.statute for p in passages]}")


def test_no_duplicate_passages(store):
    passages = rag.retrieve(store, "trust punishment", k=6, bns_floor=4)
    keys = [(p.source, p.page, p.text[:120]) for p in passages]
    assert len(keys) == len(set(keys)), "merge of the two searches left duplicates"
    print(f"  {len(keys)} unique passages")


def test_context_carries_citations(store):
    passages = rag.retrieve(store, "theft", k=3, bns_floor=1)
    context = rag.build_context(passages)
    for p in passages:
        assert p.citation in context, f"missing citation for {p.citation}"
        assert p.text.strip() in context
    assert context.startswith("[1] ("), context[:40]
    print(f"  context {len(context)} chars, {len(passages)} passages")


def test_question_cleaning():
    assert rag.clean_question("  what is\n\n theft?  ") == "what is theft?"
    assert len(rag.clean_question("x" * 900)) == 400
    assert rag.clean_question(None) == ""
    assert rag.clean_question("") == ""
    print("  whitespace, length cap and empty input handled")


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        store = make_store(tmpdir)
        tests = [
            ("bns leads the context", lambda: test_bns_leads_the_context(store)),
            ("bns floor pulls in bns hits", lambda: test_bns_floor_pulls_in_bns_hits(store)),
            ("no duplicate passages", lambda: test_no_duplicate_passages(store)),
            ("context carries citations", lambda: test_context_carries_citations(store)),
            ("question cleaning", test_question_cleaning),
        ]
        for name, fn in tests:
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print("\nall passed" if not failures else f"\n{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
