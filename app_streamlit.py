import io
from datetime import datetime
from PIL import Image, ImageOps, ImageFile
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="MiniCPM-V-4 Vision App", layout="wide")
DEFAULT_PROMPT = "Describe in details this image"
MODEL_NAME = "openbmb/MiniCPM-V-4"

# Hard-force bigger uploader by directly targeting uploader container
st.markdown("""
<style>
/* Force a taller card around the uploader block */
div[data-testid="stVerticalBlock"] div:has(> div > section[data-testid="stFileUploader"]) {
  padding: 8px;
  border: 2px dashed #7b8cff;
  background: #f4f6ff;
  border-radius: 14px;
}
/* The actual dropzone area height */
section[data-testid="stFileUploader"] > div > div {
  min-height: 220px !important; /* bigger drop area */
  display: flex; align-items: center; justify-content: center;
}
/* Make the label text larger */
section[data-testid="stFileUploader"] label div {
  font-size: 1rem !important;
}
/* Result cards */
.result-card { border: 1px solid #e6e8f0; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; background: #fbfbff; }
.result-title { font-weight: 700; margin-bottom: 6px; color: #1f2430; word-break: break-word; }
.result-text { white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.95rem; line-height: 1.45rem; color: #222; margin: 0; }
.top-actions { position: sticky; top: 0; z-index: 20; background: white; padding: 8px 0 10px 0; border-bottom: 1px solid #eee; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

def pick_device_and_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32

@st.cache_resource(show_spinner=True)
def load_model(device_pref="auto"):
    if device_pref == "auto":
        device, dtype = pick_device_and_dtype()
    else:
        device = device_pref
        dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=dtype
        ).eval().to(device)
    except Exception:
        device, dtype = "cpu", torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=dtype
        ).eval().to(device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return model, tok, device, dtype

def to_pil(f):
    img = Image.open(f)
    if img.mode != "RGB": img = img.convert("RGB")
    return img

def make_thumbnail(img: Image.Image, max_side=220):
    return ImageOps.contain(img, (max_side, max_side))

def chat(model, tok, msgs):
    with torch.inference_mode():
        return model.chat(image=None, msgs=msgs, tokenizer=tok, do_sample=False, max_new_tokens=512)

st.title("MiniCPM-V-4 Vision App")

left, right = st.columns([0.55, 0.45], gap="large")

with left:
    st.subheader("1) Upload images")
    uploaded = st.file_uploader(
        "Drop images here or click to browse",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    # Show true thumbnails (resized)
    if uploaded:
        st.caption("Preview (thumbnails)")
        thumbs = []
        for f in uploaded:
            try:
                img = to_pil(f)
                thumbs.append((make_thumbnail(img), f.name))
            except Exception:
                pass
        if thumbs:
            cols = st.columns(min(4, len(thumbs)))
            for i, (timg, name) in enumerate(thumbs):
                with cols[i % len(cols)]:
                    st.image(timg, caption=name, use_container_width=True)

    st.subheader("2) Mode and prompt")
    mode = st.radio("Mode", ["Single", "Pair compare", "Sliding"], index=0, horizontal=True)
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, placeholder=DEFAULT_PROMPT, height=110)

    st.subheader("3) Device")
    device_pref = st.selectbox("Run on", ["auto", "cpu", "cuda"], index=0)

    run = st.button("Run", type="primary", use_container_width=True)

with right:
    st.subheader("Results")

if run:
    if not uploaded:
        st.warning("Please upload at least one image.")
        st.stop()
    if not prompt.strip():
        prompt = DEFAULT_PROMPT

    with st.spinner("Loading model..."):
        model, tok, device, dtype = load_model(device_pref)

    pil_list = []
    names = []
    for f in uploaded:
        try:
            pil_list.append(to_pil(f)); names.append(f.name)
        except Exception:
            st.error(f"Failed to read image: {getattr(f, 'name', 'file')}")
            st.stop()

    results = []
    with st.spinner("Generating..."):
        if mode == "Single":
            for img, name in zip(pil_list, names):
                ans = chat(model, tok, [{'role': 'user', 'content': [img, prompt]}])
                results.append((f"[Single] {name}", ans))
        elif mode == "Pair compare":
            if len(pil_list) < 2:
                st.error("Need at least 2 images for pair compare."); st.stop()
            for i in range(0, len(pil_list), 2):
                if i + 1 >= len(pil_list): break
                ans = chat(model, tok, [{'role': 'user', 'content': [pil_list[i], pil_list[i+1], prompt]}])
                results.append((f"[Pair] {names[i]} ↔ {names[i+1]}", ans))
        else:
            if len(pil_list) < 2:
                st.error("Need at least 2 images for sliding compare."); st.stop()
            for i in range(len(pil_list) - 1):
                ans = chat(model, tok, [{'role': 'user', 'content': [pil_list[i], pil_list[i+1], prompt]}])
                results.append((f"[Sliding] {names[i]} ↔ {names[i+1]}", ans))

    combined = "\n".join([f"{t}\n{x}\n" + "-" * 50 for t, x in results])

    with right:
        st.markdown('<div class="top-actions">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            bio = io.BytesIO(combined.encode("utf-8"))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("Download all", data=bio, file_name=f"outputs_{ts}.txt", mime="text/plain", use_container_width=True)
        with c2:
            with st.expander("Copy all (use the copy icon)", expanded=False):
                st.code(combined)
        st.markdown("</div>", unsafe_allow_html=True)

        for title, txt in results:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text">{txt}</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
