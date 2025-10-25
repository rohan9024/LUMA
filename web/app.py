# web/app.py (top of file)
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
    
    
import streamlit as st
from PIL import Image

from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor
from retrieval.engine import RetrievalEngine
from rag.generator import Generator
import time

def init_state():
    if "engine" not in st.session_state:
        idx = MultiTierIndex(CFG.models.d, CFG.index)
        pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
        ing = Ingestor(CFG, idx, pol)
        eng = RetrievalEngine(CFG, idx, ing)
        gen = Generator()
        st.session_state.update(idx=idx, pol=pol, ing=ing, engine=eng, gen=gen)

def run():
    st.title("LUMA-RAG: Lifelong Multimodal RAG that Remembers")
    init_state()
    ing, eng, gen = st.session_state.ing, st.session_state.engine, st.session_state.gen

    st.subheader("Ingest")
    with st.expander("Add images with captions"):
        files = st.file_uploader("Images", type=["jpg","png","jpeg"], accept_multiple_files=True)
        caps = st.text_area("Captions (one per line)")
        if st.button("Ingest images"):
            imgs = [Image.open(f).convert("RGB") for f in files]
            caps_list = [s.strip() for s in caps.split("\n") if s.strip()]
            metas = [{"source": f.name, "ts": time.time(), "modality":"image"} for f in files]
            ing.ingest_images_with_captions([f for f in files], caps_list, metas)
            st.success(f"Ingested {len(files)} images")
    with st.expander("Add text snippets"):
        txt = st.text_area("Text entries (one per line)")
        if st.button("Ingest text"):
            lines = [s.strip() for s in txt.split("\n") if s.strip()]
            metas = [{"source": f"text_{i}", "ts": time.time(), "modality":"text"} for i,_ in enumerate(lines)]
            ing.ingest_text(lines, metas)
            st.success(f"Ingested {len(lines)} texts")

    st.subheader("Ask")
    q = st.text_input("Your question")
    k = st.slider("k", 1, 20, 5)
    if st.button("Search"):
        qv = eng.embed_query(q, modality="text")
        D, I = eng.search(qv, k=k)
        st.write("Top hits:")
        evidences = []
        for d, i in zip(D, I):
            meta = st.session_state.idx.id2meta.get(int(i), {"source":"unknown"})
            evidences.append({"score": float(d), **meta})
            st.json({"score": float(d), "meta": meta})
        st.subheader("Answer")
        ans = gen.answer(q, evidences)
        st.code(ans, language="markdown")

if __name__ == "__main__":
    run()