"""Dataset discovery, validation and preparation (YOLO detection format).

The user's original dataset is only ever read. Everything we write goes into
the project folder.
"""
from __future__ import annotations

import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dng", ".mpo", ".heic"}
VAL_ALIASES = ("val", "valid", "validation")
SPLIT_SEED = 42
MAX_LISTED = 200  # cap list sizes in the report so huge datasets stay readable

ProgressFn = Callable[[int, int], None]  # (done, total)


class DatasetError(Exception):
    """Raised with a message that can be shown to the user as is."""


class Layout(str, Enum):
    SPLIT = "split"  # images/train, images/val, labels/train, labels/val
    FLAT = "flat"    # images/, labels/


@dataclass
class DatasetInfo:
    root: Path
    layout: Layout
    # split name -> (images dir, labels dir); FLAT uses the single key "all"
    splits: dict[str, tuple[Path, Path]]

    def describe(self) -> str:
        counts = {k: len(list_images(v[0])) for k, v in self.splits.items()}
        if self.layout is Layout.SPLIT:
            return f"Already split: {counts.get('train', 0)} train and {counts.get('val', 0)} val images"
        return f"Flat folder: {counts.get('all', 0)} images, split automatically"


@dataclass
class ValidationReport:
    images: dict[str, int] = field(default_factory=dict)        # split -> image count
    label_files: dict[str, int] = field(default_factory=dict)   # split -> label count
    class_counts: Counter = field(default_factory=Counter)      # class id -> box count
    background_images: int = 0                                  # empty label files
    missing_labels: list[Path] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)
    invalid_files: list[tuple[Path, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def total_images(self) -> int:
        return sum(self.images.values())


# ---------------------------------------------------------------- discovery

def list_images(folder: Path) -> list[Path]:
    """Images directly inside a folder, sorted for determinism."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _find_child(parent: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return None


def detect_layout(root: Path) -> DatasetInfo:
    """Detect a split or flat YOLO layout inside ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"The folder {root} does not exist.")
    images, labels = root / "images", root / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise DatasetError(
            "This folder needs an 'images' folder and a 'labels' folder next to each other. "
            "Either images/train + images/val with labels/train + labels/val, "
            "or a flat images/ and labels/ pair."
        )
    img_train = images / "train"
    img_val = _find_child(images, VAL_ALIASES)
    if img_train.is_dir() and img_val is not None:
        lbl_train = labels / "train"
        lbl_val = labels / img_val.name
        if not lbl_train.is_dir() or not lbl_val.is_dir():
            raise DatasetError(
                f"Found images/train and images/{img_val.name}, but labels/train or "
                f"labels/{img_val.name} is missing."
            )
        return DatasetInfo(root, Layout.SPLIT, {"train": (img_train, lbl_train), "val": (img_val, lbl_val)})
    if list_images(images):
        return DatasetInfo(root, Layout.FLAT, {"all": (images, labels)})
    raise DatasetError("No images found. Put .jpg or .png files in images/ (or images/train and images/val).")


def read_class_names(root: Path) -> tuple[list[str], str] | None:
    """Class names from classes.txt or a data.yaml (or any yaml with 'names'). Returns (names, source)."""
    root = Path(root)
    txt = root / "classes.txt"
    if not txt.is_file():
        txt = root / "labels" / "classes.txt"
    if txt.is_file():
        names = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        if names:
            return names, txt.name
    yamls = [root / "data.yaml"] + sorted(p for p in root.glob("*.y*ml") if p.name != "data.yaml")
    for y in yamls:
        if not y.is_file():
            continue
        try:
            data = yaml.safe_load(y.read_text(encoding="utf-8", errors="replace")) or {}
        except yaml.YAMLError:
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            names = [str(names[k]) for k in sorted(names, key=int)]
        if isinstance(names, list) and names:
            return [str(n) for n in names], y.name
    return None


def infer_class_count(info: DatasetInfo) -> int:
    """Highest class id found in any label file, plus one."""
    highest = -1
    for _, lbl_dir in info.splits.values():
        for txt in lbl_dir.glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if parts:
                    try:
                        highest = max(highest, int(float(parts[0])))
                    except ValueError:
                        pass
    return highest + 1


# ---------------------------------------------------------------- validation

def check_label_file(path: Path, num_classes: int) -> tuple[list[int], str | None]:
    """Return the class ids in a label file, or an error message for the first bad line."""
    ids: list[int] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [], "not a UTF-8 text file"
    for n, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            return ids, f"line {n}: expected 5 values (class x y w h), found {len(parts)}"
        try:
            values = [float(v) for v in parts]
        except ValueError:
            return ids, f"line {n}: contains a value that is not a number"
        cls = values[0]
        if not cls.is_integer():
            return ids, f"line {n}: class id {parts[0]} is not a whole number"
        if not 0 <= int(cls) < num_classes:
            return ids, f"line {n}: class id {int(cls)} is out of range (0 to {num_classes - 1})"
        if any(v < 0.0 or v > 1.0 for v in values[1:]):
            return ids, f"line {n}: box values must be between 0 and 1"
        ids.append(int(cls))
    return ids, None


def validate(info: DatasetInfo, num_classes: int) -> ValidationReport:
    """Check that every image has a label, class ids are in range and values are normalised."""
    rep = ValidationReport()
    if num_classes <= 0:
        rep.errors.append("Add at least one class name.")
    for split, (img_dir, lbl_dir) in info.splits.items():
        images = list_images(img_dir)
        rep.images[split] = len(images)
        label_files = {p.stem: p for p in lbl_dir.glob("*.txt") if p.name != "classes.txt"}
        rep.label_files[split] = len(label_files)
        stems = Counter(p.stem for p in images)
        for stem, count in stems.items():
            if count > 1:
                rep.warnings.append(f"{split}: {count} images share the name '{stem}' and would share one label file.")
        for img in images:
            lbl = label_files.get(img.stem)
            if lbl is None:
                rep.missing_labels.append(img)
                continue
            if num_classes <= 0:
                continue
            ids, problem = check_label_file(lbl, num_classes)
            if problem:
                rep.invalid_files.append((lbl, problem))
            elif not ids:
                rep.background_images += 1
            rep.class_counts.update(ids)
        image_stems = set(stems)
        rep.orphan_labels.extend(p for s, p in sorted(label_files.items()) if s not in image_stems)

    if rep.total_images == 0:
        rep.errors.append("No images found.")
    if info.layout is Layout.SPLIT and rep.images.get("val", 0) == 0:
        rep.errors.append("The val split has no images.")
    if info.layout is Layout.FLAT and 0 < rep.total_images < 2:
        rep.errors.append("At least 2 images are needed to make a train and a val split.")
    if rep.missing_labels:
        rep.errors.append(f"{len(rep.missing_labels)} image(s) have no matching label file.")
    if rep.invalid_files:
        rep.errors.append(f"{len(rep.invalid_files)} label file(s) are invalid.")
    if rep.orphan_labels:
        rep.warnings.append(f"{len(rep.orphan_labels)} label file(s) have no matching image and will be ignored.")
    if num_classes > 0:
        unused = [c for c in range(num_classes) if rep.class_counts.get(c, 0) == 0]
        if unused and rep.ok:
            rep.warnings.append(f"Class id(s) {', '.join(map(str, unused[:20]))} have no boxes.")
    return rep


def report_text(rep: ValidationReport, names: list[str]) -> str:
    """Plain-text (Markdown) report for the UI."""
    out: list[str] = []
    out.append("## Result")
    out.append("Ready to train." if rep.ok else "Training is blocked until the errors below are fixed.")
    for e in rep.errors:
        out.append(f"- Error: {e}")
    for w in rep.warnings:
        out.append(f"- Warning: {w}")
    out.append("\n## Images")
    for split, n in rep.images.items():
        out.append(f"- {split}: {n} images, {rep.label_files.get(split, 0)} label files")
    if rep.background_images:
        out.append(f"- {rep.background_images} background images (empty label file)")
    out.append("\n## Boxes per class")
    for cid, name in enumerate(names):
        out.append(f"- {cid} {name}: {rep.class_counts.get(cid, 0)}")
    if rep.missing_labels:
        out.append("\n## Missing labels")
        out += [f"- {p}" for p in rep.missing_labels[:MAX_LISTED]]
        if len(rep.missing_labels) > MAX_LISTED:
            out.append(f"- and {len(rep.missing_labels) - MAX_LISTED} more")
    if rep.invalid_files:
        out.append("\n## Invalid label files")
        out += [f"- {p.name}: {why}" for p, why in rep.invalid_files[:MAX_LISTED]]
        if len(rep.invalid_files) > MAX_LISTED:
            out.append(f"- and {len(rep.invalid_files) - MAX_LISTED} more")
    if rep.orphan_labels:
        out.append("\n## Labels without an image")
        out += [f"- {p.name}" for p in rep.orphan_labels[:MAX_LISTED]]
    return "\n".join(out)


# ---------------------------------------------------------------- preparation

def split_items(images: list[Path], val_ratio: float, seed: int = SPLIT_SEED) -> tuple[list[Path], list[Path]]:
    """Deterministic shuffle and split. Always keeps at least one image in each split when possible."""
    if not 0.0 < val_ratio < 1.0:
        raise DatasetError("The val share must be between 1 % and 99 %.")
    items = sorted(images)
    random.Random(seed).shuffle(items)
    n_val = round(len(items) * val_ratio)
    if len(items) >= 2:
        n_val = min(max(n_val, 1), len(items) - 1)
    return items[n_val:], items[:n_val]


def _copy_pairs(pairs: list[tuple[Path, Path | None]], img_out: Path, lbl_out: Path,
                progress: ProgressFn | None, done: int, total: int) -> int:
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        if lbl is not None and lbl.is_file():
            shutil.copy2(lbl, lbl_out / lbl.name)
        done += 1
        if progress and (done % 25 == 0 or done == total):
            progress(done, total)
    return done


def prepare_dataset(info: DatasetInfo, dest: Path, val_ratio: float = 0.2,
                    seed: int = SPLIT_SEED, progress: ProgressFn | None = None) -> dict[str, int]:
    """Copy the dataset into ``dest`` as images/{train,val} + labels/{train,val}.

    Flat datasets are split with a fixed seed. The source is never modified.
    """
    dest = Path(dest)
    src_root = info.root.resolve()
    if dest.resolve() == src_root or src_root in dest.resolve().parents:
        raise DatasetError("The project folder cannot be inside the source dataset.")
    if dest.exists():
        shutil.rmtree(dest)
    plan: dict[str, list[tuple[Path, Path | None]]] = {}
    if info.layout is Layout.FLAT:
        img_dir, lbl_dir = info.splits["all"]
        train, val = split_items(list_images(img_dir), val_ratio, seed)
        plan["train"] = [(p, lbl_dir / f"{p.stem}.txt") for p in train]
        plan["val"] = [(p, lbl_dir / f"{p.stem}.txt") for p in val]
    else:
        for split in ("train", "val"):
            img_dir, lbl_dir = info.splits[split]
            plan[split] = [(p, lbl_dir / f"{p.stem}.txt") for p in list_images(img_dir)]
    total = sum(len(v) for v in plan.values())
    done = 0
    for split, pairs in plan.items():
        done = _copy_pairs(pairs, dest / "images" / split, dest / "labels" / split, progress, done, total)
    return {k: len(v) for k, v in plan.items()}
