import argparse
import os
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from datetime import datetime

EXTS = (".png", ".jpg", ".jpeg", ".webp")

def load_images_from_path(path):
    images = []
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(EXTS):
                full_path = os.path.join(path, f)
                images.append((full_path, Image.open(full_path).convert('RGB')))
    elif os.path.isfile(path) and path.lower().endswith(EXTS):
        images.append((path, Image.open(path).convert('RGB')))
    return images

def save_output_if_needed(fname_base, answer, save_flag, force_flag):
    if not save_flag:
        return False
    out_name = f"{fname_base}.txt"
    if os.path.exists(out_name) and not force_flag:
        return False
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(answer)
    return True

def save_batch_report(processed_list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"batch_report_{timestamp}.txt"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(f"Batch processed: {len(processed_list)} items\n")
        for name in processed_list:
            f.write(f"- {name}\n")
    print(f"\n📄 Batch report saved as: {report_name}")

def main():
    parser = argparse.ArgumentParser(description="Analyze or compare images with a multimodal model.")
    parser.add_argument("paths", nargs="+", help="Path(s) to image(s) or directories.")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Prompt for the model.")
    parser.add_argument("--model-path", type=str, default="openbmb/MiniCPM-V-4", help="Model name or path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--compare", action="store_true", help="Compare images in fixed pairs.")
    parser.add_argument("--sliding", action="store_true", help="Sliding comparison: each image with the next.")
    parser.add_argument("--save", action="store_true", help="Save the output as .txt.")
    parser.add_argument("--force", action="store_true", help="Force overwrite if file exists.")
    args = parser.parse_args()

    if args.compare and args.sliding:
        print("⚠ Please choose only one comparison mode: --compare OR --sliding.")
        return

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
    ).eval().to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    all_images = []
    for path in args.paths:
        all_images.extend(load_images_from_path(path))

    if not all_images:
        print("No valid images found.")
        return

    processed_items = []

    # Compare mode
    if args.compare:
        for i in tqdm(range(0, len(all_images), 2), desc="Pair comparison"):
            if i+1 >= len(all_images):
                break
            fpath1, img1 = all_images[i]
            fpath2, img2 = all_images[i+1]

            dir1 = os.path.dirname(fpath1)
            name1 = os.path.splitext(os.path.basename(fpath1))[0]
            name2 = os.path.splitext(os.path.basename(fpath2))[0]
            out_base = os.path.join(dir1, f"{name1}_{name2}")

            if args.save and os.path.exists(f"{out_base}.txt") and not args.force:
                tqdm.write(f"⏩ Skipping {out_base} (already exists)")
                continue

            msgs = [{'role': 'user', 'content': [img1, img2, args.prompt]}]
            answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
            print(f"\nComparison {fpath1} ↔ {fpath2}:\n{answer}\n")

            if save_output_if_needed(out_base, answer, args.save, args.force):
                tqdm.write(f"💾 Saved: {out_base}.txt")
            processed_items.append(out_base)

    # Sliding mode
    elif args.sliding:
        for i in tqdm(range(len(all_images)-1), desc="Sliding comparison"):
            fpath1, img1 = all_images[i]
            fpath2, img2 = all_images[i+1]

            dir1 = os.path.dirname(fpath1)
            name1 = os.path.splitext(os.path.basename(fpath1))[0]
            name2 = os.path.splitext(os.path.basename(fpath2))[0]
            out_base = os.path.join(dir1, f"{name1}_{name2}")

            if args.save and os.path.exists(f"{out_base}.txt") and not args.force:
                tqdm.write(f"⏩ Skipping {out_base} (already exists)")
                continue

            msgs = [{'role': 'user', 'content': [img1, img2, args.prompt]}]
            answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
            print(f"\nComparison {fpath1} ↔ {fpath2}:\n{answer}\n")

            if save_output_if_needed(out_base, answer, args.save, args.force):
                tqdm.write(f"💾 Saved: {out_base}.txt")
            processed_items.append(out_base)

    # Single mode
    else:
        for fpath, img in tqdm(all_images, desc="Image analysis"):
            out_base = os.path.splitext(fpath)[0]

            if args.save and os.path.exists(f"{out_base}.txt") and not args.force:
                tqdm.write(f"⏩ Skipping {out_base} (already exists)")
                continue

            msgs = [{'role': 'user', 'content': [img, args.prompt]}]
            answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
            print(f"\nAnalysis {fpath}:\n{answer}\n")

            if save_output_if_needed(out_base, answer, args.save, args.force):
                tqdm.write(f"💾 Saved: {out_base}.txt")
            processed_items.append(out_base)

    if len(all_images) > 1:
        print("\n📊 Batch Report")
        print(f"Processed: {len(processed_items)} / {len(all_images)}")
        for name in processed_items:
            print(f" - {name}")
        save_batch_report(processed_items)

if __name__ == "__main__":
    main()
