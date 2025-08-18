
# MiniCPM‑V4 Image Analysis & Comparison Tool

A simple Python CLI script to analyze or compare images using a multimodal model  
([MiniCPM‑V](https://huggingface.co/openbmb/MiniCPM-V-4) by default).  
Outputs can optionally be saved as `.txt` files in the **same directory as the source image(s)**.

---

## 📦 Requirements

- **Python** 3.9 or newer  
- **Pip** (latest version recommended)
- A machine with **CUDA** GPU for best performance (falls back to CPU if needed)
- Recommended: [PyTorch](https://pytorch.org/get-started/locally/) compiled for your CUDA version

---

## 🔧 Installation

1. **Clone** or download this repository:
   ```bash
   git clone https://github.com/yourusername/minicpm-v4-analyzer.git
   cd minicpm-v4-analyzer
   ```

2. **Create & activate** a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install required packages**:

   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # adjust for your CUDA version
   pip install pillow tqdm transformers
   ```

   > 💡 **CPU only?** Use the `+cpu` index instead:
   >
   > ```bash
   > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   > ```

---

## 🚀 Usage

The script supports **three modes**:

* **Single analysis**: process each image individually
* **Pair comparison**: compare fixed pairs of images (`--compare`)
* **Sliding comparison**: compare each image with the next (`--sliding`)

Basic syntax:

```bash
python minicpmv4.py <path(s) to images or dirs> --prompt "<your prompt>" [options]
```

---

### **Examples**

**1️⃣ Analyze a single image**

```bash
python minicpmv4.py /path/to/image.jpg --prompt "Describe the content" --save
```

**2️⃣ Analyze all images in a folder**

```bash
python minicpmv4.py ./images --prompt "Summarize what's in this picture" --save
```

**3️⃣ Compare images in fixed pairs**

```bash
python minicpmv4.py ./compare_folder --prompt "Compare the differences" --compare --save
```

**4️⃣ Sliding comparison (each with the next)**

```bash
python minicpmv4.py ./sequence --prompt "Spot visual changes" --sliding --save
```

---

## ⚙ Options

| Option | Description | 
| --- | --- | 
| `--prompt` / `-p` | **(Required)** Prompt text for the model | 
| `--model-path` | Hugging Face model name or local path (default: `openbmb/MiniCPM-V-4`) | 
| `--device` | `cuda` (GPU) or `cpu` (default: `cuda`) | 
| `--compare` | Compare images in fixed pairs | 
| `--sliding` | Compare each image with the next | 
| `--save` | Save output as `.txt` (next to the image(s)) | 
| `--force` | Overwrite `.txt` if it already exists | 
| `--int4` | Use pre-quantized int4 model | 

---

## 📝 Output

* Saved `.txt` files are placed **in the same directory** as their corresponding image(s).
* A `batch_report_<timestamp>.txt` is generated for multi-image runs, listing all processed files.

---

## 📄 License

MIT License — free to use and modify.
