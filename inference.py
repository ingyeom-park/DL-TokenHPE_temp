import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms

import utils
from model import TokenHPE


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run TokenHPE inference on one image or a folder.")
    parser.add_argument(
        "--model_path",
        default="./weights/TokenHPEv1-ViTB-224_224-lyr3.tar",
        help="Path to the trained TokenHPE weight file.",
    )
    parser.add_argument(
        "--image_path",
        default="",
        help="Optional single image path. If omitted, all images in --input_dir are processed.",
    )
    parser.add_argument(
        "--input_dir",
        default="./input",
        help="Folder to scan when --image_path is not provided.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output/vis",
        help="Directory where annotated images are saved.",
    )
    parser.add_argument(
        "--csv_path",
        default="./output/results.csv",
        help="CSV file where pitch/yaw/roll predictions are saved.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--disable_face_detection",
        action="store_true",
        help="Skip face detection and run on the full image.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(270),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weight file was not found: {model_path}")

    model = TokenHPE(
        num_ori_tokens=9,
        depth=3,
        heads=8,
        embedding="sine",
        dim=128,
        inference_view=False,
    ).to(device)

    saved_state_dict = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in saved_state_dict:
        model.load_state_dict(saved_state_dict["model_state_dict"])
    else:
        model.load_state_dict(saved_state_dict)

    model.eval()
    return model


def collect_image_paths(args):
    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image was not found: {image_path}")
        return [image_path]

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_face_detector(disable_face_detection):
    if disable_face_detection:
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    return detector


def detect_face_box(image_bgr, detector):
    if detector is None:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(w), int(h)


def expand_face_box(face_box, image_shape, margin_ratio=0.35):
    if face_box is None:
        height, width = image_shape[:2]
        return 0, 0, width, height

    x, y, w, h = face_box
    height, width = image_shape[:2]
    x_margin = int(w * margin_ratio)
    y_margin = int(h * margin_ratio)

    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(width, x + w + x_margin)
    y2 = min(height, y + h + y_margin)

    return x1, y1, x2 - x1, y2 - y1


def crop_image(image_bgr, box):
    x, y, w, h = box
    return image_bgr[y : y + h, x : x + w]


def predict_pose(model, image_bgr, crop_box, transform, device):
    crop_bgr = crop_image(image_bgr, crop_box)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb).convert("RGB")
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        rotation_matrix, _ = model(image_tensor)
        euler = utils.compute_euler_angles_from_rotation_matrices(rotation_matrix, use_gpu=device.type == "cuda")
        euler = euler * 180 / np.pi

    pitch = float(euler[0, 0].cpu().item())
    yaw = float(euler[0, 1].cpu().item())
    roll = float(euler[0, 2].cpu().item())
    return pitch, yaw, roll


def draw_prediction(image_bgr, crop_box, pitch, yaw, roll):
    output = image_bgr.copy()
    x, y, w, h = crop_box
    center_x = x + w / 2
    center_y = y + h / 2
    cube_size = max(60.0, min(w, h) * 0.6)

    utils.plot_pose_cube(output, yaw, pitch, roll, tdx=center_x, tdy=center_y, size=cube_size)

    text = f"Prediction: pitch:{pitch:.2f}, yaw:{yaw:.2f}, roll:{roll:.2f}"
    text_origin = (10, 28)
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(
        output,
        (6, 6),
        (16 + text_size[0], 12 + text_size[1] + baseline),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(output, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def ensure_parent_dir(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    cudnn.enabled = True

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_dir(args.csv_path)

    model = load_model(args.model_path, device)
    transform = build_transform()
    detector = load_face_detector(args.disable_face_detection)
    image_paths = collect_image_paths(args)

    if not image_paths:
        print("No images found. Put files into ./input or pass --image_path.")
        raise SystemExit(0)

    with open(args.csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["filename", "pitch", "yaw", "roll"])

        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            detected_box = detect_face_box(image_bgr, detector)
            crop_box = expand_face_box(detected_box, image_bgr.shape)
            pitch, yaw, roll = predict_pose(model, image_bgr, crop_box, transform, device)

            annotated_image = draw_prediction(image_bgr, crop_box, pitch, yaw, roll)
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), annotated_image)

            csv_writer.writerow([image_path.name, round(pitch, 2), round(yaw, 2), round(roll, 2)])
            print(f"{image_path.name} -> pitch:{pitch:.2f}, yaw:{yaw:.2f}, roll:{roll:.2f}")
            print(f"Saved: {output_path}")

    print(f"Results saved to {args.csv_path}")
