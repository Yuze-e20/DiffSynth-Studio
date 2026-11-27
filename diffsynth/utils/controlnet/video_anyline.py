"""Anyline track for a whole video: the drop-in alternative to video_lineart.

    python -m diffsynth.utils.controlnet.video_anyline in.mp4 out.mp4

Same contract as video_lineart -- source fps, source frame size, white lines on
black, one output frame per input frame -- so a lineart.mp4 made by either one
can feed the same ControlNet branch. Anyline is TheMistoAI's MTEED detector
(TEED fine-tuned) plus a local-contrast line pass, shipped in controlnet_aux
0.0.10; only the 248 KB MTEED.pth has to be fetched separately.
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

DETECT_RESOLUTION = 1280    # Anyline's own default
MODEL_PATH = "models/Annotators/Anyline"


def video_to_anyline(input_path, output_path, model_path=MODEL_PATH,
                     detect_resolution=DETECT_RESOLUTION, device="cuda"):
    from controlnet_aux import AnylineDetector
    detector = AnylineDetector.from_pretrained(model_path, filename="MTEED.pth").to(device)
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()["fps"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=9, macro_block_size=1)
    n = 0
    for frame in reader:
        img = Image.fromarray(frame).convert("RGB")
        line = detector(img, detect_resolution=detect_resolution).convert("RGB")
        if line.size != img.size:
            line = line.resize(img.size, Image.BILINEAR)
        writer.append_data(np.asarray(line))
        n += 1
    reader.close()
    writer.close()          # without close() the mp4 trailer is missing and the file is unreadable
    return output_path, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--detect-resolution", type=int, default=DETECT_RESOLUTION)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    path, n = video_to_anyline(args.input, args.output, model_path=args.model_path,
                               detect_resolution=args.detect_resolution, device=args.device)
    print(f"anyline: {n} frames -> {path}", flush=True)


if __name__ == "__main__":
    main()
