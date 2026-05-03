import argparse
from pathlib import Path
from io import BytesIO
import sys

import numpy as np
from PIL import Image


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    elif image.mode == "RGB":
        image = image.convert("RGBA")
    return image


def detect_onnxruntime_gpu() -> bool:
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        return any(p for p in providers if "CUDA" in p or "Tensorrt" in p)
    except Exception:
        return False


def remove_background_rembg(image: Image.Image) -> Image.Image | None:
    try:
        from rembg import remove
    except ImportError:
        return None

    try:
        if detect_onnxruntime_gpu():
            print("rembg detectó backend GPU (CUDA). Usando aceleración GPU.")
        else:
            print("rembg no detectó backend GPU; usando CPU o backend por defecto.")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        result = remove(buffer.getvalue())
        out = Image.open(BytesIO(result)).convert("RGBA")
        return out
    except Exception as exc:
        print("Error al usar rembg:", exc)
        print("Usando grabCut como respaldo. Para instalar rembg correctamente usa: pip install \"rembg[cpu]\"\n")
        return None


def remove_background_grabcut(image: Image.Image) -> Image.Image:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for grabCut fallback when rembg is not available. "
            "Install it with: pip install opencv-python"
        ) from exc

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (1, 1, max(1, w - 2), max(1, h - 2))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    alpha = (mask2 * 255).astype("uint8")
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def remove_background(image: Image.Image) -> Image.Image:
    result = remove_background_rembg(image)
    if result is not None:
        return result

    print("rembg no está instalado. Usando grabCut como respaldo.")
    return remove_background_grabcut(image)


def connected_components(binary_mask: np.ndarray) -> np.ndarray:
    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    def neighbors(y: int, x: int):
        if y > 0:
            yield y - 1, x
        if y < h - 1:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x < w - 1:
            yield y, x + 1

    for y in range(h):
        for x in range(w):
            if binary_mask[y, x] and labels[y, x] == 0:
                current_label += 1
                stack = [(y, x)]
                labels[y, x] = current_label
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in neighbors(cy, cx):
                        if binary_mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))

    return labels


def clean_alpha_components(image: Image.Image, min_area: int = 10) -> Image.Image:
    array = np.array(image)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("La imagen debe tener un canal alfa para limpiar componentes.")

    alpha = array[..., 3]
    mask = alpha > 0
    if not np.any(mask):
        return image

    labels = connected_components(mask.astype(np.uint8))
    keep_mask = np.zeros_like(mask, dtype=bool)

    for label in range(1, labels.max() + 1):
        component = labels == label
        if np.count_nonzero(component) >= min_area:
            keep_mask |= component

    cleaned = array.copy()
    cleaned[..., 3] = np.where(keep_mask, cleaned[..., 3], 0)
    return Image.fromarray(cleaned, mode="RGBA")


def split_objects(image: Image.Image, min_area: int = 100) -> list[Image.Image]:
    array = np.array(image)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("La imagen debe tener un canal alfa después de quitar el fondo.")

    alpha = array[..., 3]
    mask = alpha > 0
    if not np.any(mask):
        return []

    labels = connected_components(mask.astype(np.uint8))
    objects: list[Image.Image] = []

    for label in range(1, labels.max() + 1):
        coords = np.argwhere(labels == label)
        if coords.size == 0:
            continue

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        area = coords.shape[0]
        if area < min_area:
            continue

        crop = image.crop((x0, y0, x1, y1))
        obj = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        obj.paste(crop, (0, 0), crop)
        objects.append(obj)

    return objects


def save_images(images: list[Image.Image], output_dir: Path, base_name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for index, img in enumerate(images, start=1):
        path = output_dir / f"{base_name}_object_{index}.png"
        img.save(path)
        saved_paths.append(path)

    return saved_paths


def cleanup_small_images(paths: list[Path], min_width: int, min_height: int) -> list[Path]:
    removed: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            continue

        if width < min_width or height < min_height:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def build_output_name(input_path: Path) -> str:
    name = input_path.stem
    return name.replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quita el fondo de una imagen y guarda cada objeto detectado en archivos separados."
    )
    parser.add_argument("input_image", help="Ruta de la imagen de entrada.")
    parser.add_argument(
        "--output-dir",
        default="output_objects",
        help="Carpeta donde se guardarán las imágenes resultantes.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Área mínima en pixeles para considerar un objeto válido.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=40,
        help="Ancho mínimo en pixeles para conservar una imagen de objeto.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=40,
        help="Alto mínimo en pixeles para conservar una imagen de objeto.",
    )
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Guarda la imagen completa con fondo eliminado además de cada objeto separado.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_image)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: no se encontró la imagen de entrada: {input_path}")
        return 1

    image = load_image(input_path)
    result = remove_background(image)
    result = clean_alpha_components(result, min_area=max(10, args.min_area // 10))

    if args.save_full:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_path = output_dir / f"{build_output_name(input_path)}_no_background.png"
        result.save(full_path)
        print(f"Guardada imagen completa sin fondo: {full_path}")

    objects = split_objects(result, min_area=args.min_area)
    if not objects:
        print("No se detectaron objetos después de quitar el fondo.")
        return 0

    saved = save_images(objects, output_dir, build_output_name(input_path))
    removed = cleanup_small_images(saved, args.min_width, args.min_height)
    kept = [path for path in saved if path not in removed]

    if removed:
        print(f"Eliminadas {len(removed)} imágenes demasiado pequeñas (menor que {args.min_width}x{args.min_height}).")

    if not kept:
        print("No quedó ninguna imagen después de eliminar las miniaturas.")
        return 0

    print(f"Guardadas {len(kept)} imágenes de objeto en: {output_dir}")
    for path in kept:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
