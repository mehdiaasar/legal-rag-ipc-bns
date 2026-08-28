"""End-to-end test of the Streamlit demo, offline.

Drives the real app.py through Streamlit's AppTest harness against a stub index,
so the download-free suite still checks what a visitor actually sees: example
questions, BNS-first source ordering, page citations and the disclaimer.

    python test_app.py
"""

import pathlib
import tempfile
import textwrap

from streamlit.testing.v1 import AppTest

from test_rag import make_store

HERE = pathlib.Path(__file__).parent

WRAPPER = """
import sys, runpy
sys.path.insert(0, {here!r})
import rag
from test_rag import WordOverlapEmbeddings
from langchain_chroma import Chroma

rag.load_retriever = lambda index_dir=None: Chroma(
    collection_name=rag.COLLECTION,
    persist_directory={index!r},
    embedding_function=WordOverlapEmbeddings(),
)
runpy.run_path({app!r}, run_name="__main__")
"""


def run():
    with tempfile.TemporaryDirectory() as tmp:
        index = str(pathlib.Path(tmp) / "index")
        make_store(index)

        wrapper = pathlib.Path(tmp) / "wrapper.py"
        wrapper.write_text(
            textwrap.dedent(WRAPPER).format(
                here=str(HERE), index=index, app=str(HERE / "app.py")
            )
        )

        at = AppTest.from_file(str(wrapper), default_timeout=120)
        at.run()
        assert not at.exception, at.exception
        assert at.title[0].value.endswith("IPC + BNS")
        assert len(at.button) >= 3, "example questions missing"
        print("PASS  app renders with example questions")

        # Clicking an example fills the question box.
        at.button[0].click().run()
        assert not at.exception, at.exception
        assert at.text_input[0].value, "example click did not populate the input"
        print(f"PASS  example click -> {at.text_input[0].value!r}")

        # A typed question produces cited sources.
        at.text_input[0].set_value("What is the punishment for theft?").run()
        assert not at.exception, at.exception

        labels = [e.label for e in at.expander if e.label.startswith("[")]
        assert labels, "no retrieved passages shown"
        assert all("page" in l for l in labels), f"citations lack pages: {labels}"
        print(f"PASS  {len(labels)} cited passages shown")

        bns_last = [("BNS" not in l) for l in labels]
        assert bns_last == sorted(bns_last), f"BNS must precede IPC: {labels}"
        assert "BNS" in labels[0], f"BNS should lead: {labels[0]}"
        print(f"PASS  BNS-first ordering: {[l.split(' ')[1] for l in labels]}")

        # With no token configured the app must say so rather than fail.
        infos = [i.value for i in at.info]
        assert any("retrieval-only" in i for i in infos), infos
        print("PASS  degrades to retrieval-only without a token")

        captions = [c.value for c in at.caption]
        assert any("not legal advice" in c for c in captions), "disclaimer missing"
        print("PASS  disclaimer present")

    print("\nall passed")


if __name__ == "__main__":
    run()
