import os
import tempfile
from datetime import datetime
from PIL import Image
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openbmb/MiniCPM-V-4"
DEFAULT_PROMPT = "Describe in details this image"

# Global model cache
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)

def load_model():
    global model, tokenizer
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return model, tokenizer

def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

def run_infer(files, mode, prompt):
    if not files:
        return "Please upload at least one image.", gr.update(visible=False), None

    m, t = load_model()
    prompt = (prompt or "").strip() or DEFAULT_PROMPT

    # Load PIL images and names
    pil_images = []
    names = []
    for f in files:
        # gr.Files returns temp file paths
        if isinstance(f, str):
            img = Image.open(f)
            pil_images.append(to_rgb(img))
            names.append(os.path.basename(f))
        else:
            # Some versions may pass dicts with 'name'
            path = f.get("name") if isinstance(f, dict) else None
            if path:
                img = Image.open(path)
                pil_images.append(to_rgb(img))
                names.append(os.path.basename(path))

    if not pil_images:
        return "No readable images.", gr.update(visible=False), None

    # Build results
    results = []
    if mode == "Single":
        # Only the raw answer, no headers/separators
        for img in pil_images:
            msgs = [{'role': 'user', 'content': [img, prompt]}]
            ans = m.chat(image=None, msgs=msgs, tokenizer=t, do_sample=False, max_new_tokens=512)
            results.append(ans)
        combined = "\n\n".join(results).strip()
    elif mode == "Pair compare":
        if len(pil_images) < 2:
            return "Need at least 2 images for pair compare.", gr.update(visible=False), None
        blocks = []
        for i in range(0, len(pil_images), 2):
            if i + 1 >= len(pil_images):
                break
            msgs = [{'role': 'user', 'content': [pil_images[i], pil_images[i+1], prompt]}]
            ans = m.chat(image=None, msgs=msgs, tokenizer=t, do_sample=False, max_new_tokens=512)
            title = f"[Pair] {names[i]} ↔ {names[i+1]}"
            blocks.append(f"{title}\n{ans}\n" + "-" * 50)
        combined = "\n".join(blocks).strip()
    else:  # Sliding
        if len(pil_images) < 2:
            return "Need at least 2 images for sliding compare.", gr.update(visible=False), None
        blocks = []
        for i in range(len(pil_images) - 1):
            msgs = [{'role': 'user', 'content': [pil_images[i], pil_images[i+1], prompt]}]
            ans = m.chat(image=None, msgs=msgs, tokenizer=t, do_sample=False, max_new_tokens=512)
            title = f"[Sliding] {names[i]} ↔ {names[i+1]}"
            blocks.append(f"{title}\n{ans}\n" + "-" * 50)
        combined = "\n".join(blocks).strip()

    # Save to a real temp file path for gr.File
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = tempfile.mkdtemp(prefix="minicpm_")
    out_path = os.path.join(tmp_dir, f"outputs_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined)

    return combined, gr.update(visible=True, value=combined), out_path


def preview_gallery(files):
    # Gallery expects list of images or paths; we’ll return thumbnail PILs
    thumbs = []
    if not files:
        return thumbs
    for f in files:
        path = f if isinstance(f, str) else (f.get("name") if isinstance(f, dict) else None)
        if not path:
            continue
        try:
            img = Image.open(path)
            # Make thumbnails (max dimension 256)
            img.thumbnail((256, 256))
            thumbs.append(img)
        except Exception:
            pass
    return thumbs

with gr.Blocks(title="MiniCPM-V-4 Vision App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## MiniCPM-V-4 Vision App")

    with gr.Row():
        with gr.Column(scale=1):
            uploader = gr.Files(
                label="Drop images here",
                file_types=["image"],
                file_count="multiple",
                elem_id="dropzone"
            )
            gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height=260,
                object_fit="contain",
                show_label=True
            )
            mode = gr.Radio(["Single", "Pair compare", "Sliding"], value="Single", label="Mode")
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
            run = gr.Button("Run", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Output", lines=18, show_copy_button=True)
            download = gr.File(label="Download results")

    uploader.change(preview_gallery, inputs=uploader, outputs=gallery)
    run.click(run_infer, inputs=[uploader, mode, prompt], outputs=[output_text, output_text, download])

    # A bit of CSS to clearly enlarge drop area
    gr.HTML("""
    <style>
      #dropzone { border: 2px dashed #8fa3ff !important; padding: 20px; border-radius: 12px; }
    </style>
    """)

if __name__ == "__main__":
    demo.launch()
