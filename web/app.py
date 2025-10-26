# web/app.py
import os, time, sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Make repo root importable when running streamlit from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor
from retrieval.engine import RetrievalEngine
from rag.generator import Generator

def build_services():
    idx = MultiTierIndex(CFG.models.d, CFG.index)
    pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
    ing = Ingestor(CFG, idx, pol)
    eng = RetrievalEngine(CFG, idx, ing)
    gen = Generator()
    return idx, pol, ing, eng, gen

def init_state(force=False):
    if force or "engine" not in st.session_state:
        idx, pol, ing, eng, gen = build_services()
        st.session_state.update(idx=idx, pol=pol, ing=ing, engine=eng, gen=gen)

def run():
    st.set_page_config(page_title="LUMA-RAG", layout="wide")
    st.title("LUMA-RAG: Lifelong Multimodal RAG that Remembers")
    init_state()
    ing, eng = st.session_state.ing, st.session_state.engine

    # Controls / Reset
    colA, colB, colC = st.columns([1,1,3])
    with colA:
        if st.button("Reset index"):
            st.session_state.clear()
            st.rerun()
    with colB:
        captionless = st.checkbox("Captionless mode (ignore captions for alignment)", value=False)

    st.subheader("Ingest")
    with st.expander("Add images with captions"):
        files = st.file_uploader("Images", type=["jpg","jpeg","png"], accept_multiple_files=True, key="img_up")
        caps = st.text_area("Captions (one per line, same order as images)", key="caps_up")
        if st.button("Ingest images"):
            if not files:
                st.warning("Upload at least one image.")
            else:
                save_dir = Path("data/inbox")
                save_dir.mkdir(parents=True, exist_ok=True)
                # prepare captions
                caps_list = [s.strip() for s in (caps or "").split("\n") if s.strip()]
                if len(caps_list) != len(files):
                    # create dummy captions if captionless
                    if captionless:
                        caps_list = [""] * len(files)
                    else:
                        st.error(f"Need {len(files)} captions (got {len(caps_list)}).")
                        st.stop()

                image_paths, metas = [], []
                for f, cap in zip(files, caps_list):
                    p = save_dir / f.name
                    with open(p, "wb") as out:
                        out.write(f.getbuffer())
                    image_paths.append(str(p))
                    metas.append({
                        "source": str(p),
                        "caption": cap,
                        "text": cap,
                        "modality": "image",
                        "ts": time.time()
                    })
                ing.ingest_images_with_captions(image_paths, caps_list, metas, use_caption_alignment=not captionless)
                st.success(f"Ingested {len(image_paths)} images")

    with st.expander("Add text snippets"):
        txt = st.text_area("Text entries (one per line)")
        if st.button("Ingest text"):
            lines = [s.strip() for s in (txt or "").split("\n") if s.strip()]
            metas = [{"source": f"text_{i}", "text": line, "modality":"text", "ts": time.time()} for i, line in enumerate(lines)]
            ing.ingest_text(lines, metas)
            st.success(f"Ingested {len(lines)} texts")

    st.subheader("Ask")
    q = st.text_input("Your question", key="ask_text")
    k = st.slider("k", 1, 20, 5)
    if st.button("Search", type="primary"):
        if not q.strip():
            st.warning("Enter a question")
        else:
            qv = eng.embed_query(q, modality="text")
            D, I, tel = eng.search_with_telemetry(qv, k=k)
            st.write("Top hits:")
            evidences = []
            for d, i in zip(D, I):
                meta = st.session_state.idx.id2meta.get(int(i), {"source":"unknown"})
                evidences.append({"score": float(d), **meta})
                st.json({"score": float(d), "meta": meta})
                if meta.get("modality") == "image" and meta.get("source") and os.path.exists(meta["source"]):
                    st.image(meta["source"], width=250, caption=meta.get("caption", meta.get("text","")))
            st.markdown(f"Telemetry: margin_top2={tel['margin_top2']:.4f}, eps_align={tel['eps_align']:.4f}, zeta_pq={tel['zeta_pq']:.4f}, safe_top1={tel['safe_top1']}")
            st.subheader("Answer")
            ans = st.session_state.gen.answer(q, evidences)
            st.code(ans, language="markdown")

    with st.expander("Ask with an image"):
        qfile = st.file_uploader("Query image", type=["jpg","jpeg","png"], key="imgq")
        if qfile and st.button("Search by image"):
            im = Image.open(qfile).convert("RGB")
            qv = eng.embed_query(im, modality="image")
            D, I, tel = eng.search_with_telemetry(qv, k=k)
            st.write("Top hits (image query):")
            for d, i in zip(D, I):
                meta = st.session_state.idx.id2meta.get(int(i), {"source":"unknown"})
                st.write(f"{d:.3f} → {meta.get('source')}")
                if meta.get("modality") == "image" and os.path.exists(meta.get("source","")):
                    st.image(meta["source"], width=250, caption=meta.get("caption", ""))

    with st.expander("Ask with audio"):
        aq = st.file_uploader("Query audio (wav/mp3)", type=["wav","mp3"], key="audq")
        if aq and st.button("Search by audio"):
            if st.session_state.ing.aud is None:
                st.error("Audio model not available.")
            else:
                # save temp and embed
                tmp = Path("data/inbox") / f"_query_audio_{int(time.time())}.wav"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "wb") as out: out.write(aq.getbuffer())
                qv = eng.embed_query(str(tmp), modality="audio")
                D, I, tel = eng.search_with_telemetry(qv, k=k)
                st.write("Top hits (audio query):")
                for d, i in zip(D, I):
                    meta = st.session_state.idx.id2meta.get(int(i), {"source":"unknown"})
                    st.json({"score": float(d), "meta": meta})
                try: os.remove(tmp)
                except Exception: pass

if __name__ == "__main__":
    run()