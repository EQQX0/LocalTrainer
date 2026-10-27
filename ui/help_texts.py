"""Every beginner-facing text in one place: setting tooltips, metric and button tips, validation fixes,
presets and the Help dialog pages. Plain English, metric units, no jargon without an explanation."""
from __future__ import annotations

import textwrap
from typing import Any

# ---------------------------------------------------------------- settings
# Each tip answers: what it does, when to change it, the recommended value, and the cost
# in speed, GPU memory and accuracy. Kept under 60 words so it fits in a tooltip.
SETTINGS: dict[str, dict[str, str]] = {
    "preset": {
        "label": "Preset",
        "tip": "Fills in good values for you. Easy suits most datasets. Quick test checks that everything works "
               "in a few minutes. Best accuracy trains a bigger model for longer on a 12 GB card. "
               "Change any value and the preset shows Custom.",
    },
    "family": {
        "label": "Family",
        "tip": "Which generation of YOLO to start from. yolo26 is the newest and usually the most accurate. "
               "Pick yolo11 or yolov8 only to match an older project. Recommended: yolo26. "
               "Speed and GPU memory are about the same for all three.",
    },
    "size": {
        "label": "Size",
        "tip": "How big the model is, from n (nano) to x (extra large). Bigger finds more objects but trains "
               "slower and needs more GPU memory. Recommended: s to start, m for best results on a 12 GB card. "
               "Use n for a quick test.",
    },
    "epochs": {
        "label": "Epochs",
        "tip": "How many times the model sees every training image. Raise it if mAP50 is still climbing at "
               "the end; lower it for a quick test. Recommended: 100. More epochs take longer but use no extra "
               "GPU memory. Training stops early when results stop improving.",
    },
    "imgsz": {
        "label": "Image size",
        "tip": "Images are resized to this many pixels on the long side before training. Raise it (for example "
               "to 960) when objects are small. Recommended: 640. Larger is slower, uses much more GPU memory "
               "and can find small objects better.",
    },
    "batch": {
        "label": "Batch",
        "tip": "How many images the GPU processes at once. Lower it if you see an out-of-memory error. "
               "Auto picks the largest batch that fits in about 60 % of GPU memory. Recommended: 16. "
               "Larger is faster per epoch and uses more GPU memory; accuracy barely changes.",
    },
    "patience": {
        "label": "Patience",
        "tip": "Stops training early when mAP has not improved for this many epochs, which saves time. "
               "Raise it for long runs; set 0 to always run every epoch. Recommended: 50. "
               "No effect on GPU memory or speed per epoch.",
    },
    "workers": {
        "label": "Workers",
        "tip": "Background processes that load images while the GPU trains. Set 0 if training hangs before the "
               "first epoch or the PC runs out of RAM. Recommended: 4. More workers can be faster but use more "
               "RAM, not GPU memory.",
    },
    "device": {
        "label": "Device",
        "tip": "Where training runs. Auto uses the NVIDIA GPU when it works and falls back to the CPU. "
               "Pick CPU only to test without a GPU. Recommended: Auto. The CPU is often 20 to 50 times slower; "
               "accuracy is the same.",
    },
    "val": {
        "label": "Val share",
        "tip": "Share of images kept aside to check the model, used only when your folder is not already split "
               "into train and val. The split is the same every time. Recommended: 20 %. More val images give a "
               "fairer score but fewer images to learn from.",
    },
    "auto_onnx": {
        "label": "ONNX",
        "tip": "When training ends, also save the best model as best.onnx, a format other programs can run. "
               "Turn it off if you only use LocalTrainer or Python. Recommended: on. Adds a few seconds at the "
               "end; training itself is unchanged.",
    },
}

# ---------------------------------------------------------------- collapsible sections
# id -> (title, setting keys inside). Order is the order in the sidebar.
SECTIONS: list[tuple[str, str, tuple[str, ...]]] = [
    ("training", "Advanced training settings", ("imgsz", "batch", "patience", "workers")),
    ("hardware", "Hardware settings", ("device",)),
    ("data", "Data settings", ("val",)),
    ("output", "Output settings", ("auto_onnx",)),
]

# ---------------------------------------------------------------- presets
# Only the values a preset controls. Device, workers, ONNX and val share are machine or
# workflow choices, so picking a preset leaves them alone and editing them keeps the preset.
PRESETS: dict[str, dict[str, Any]] = {
    "Easy (recommended)": {"family": "yolo26", "size": "s", "epochs": 100, "imgsz": 640, "batch": 16,
                           "batch_auto": False, "patience": 50},
    "Quick test": {"family": "yolo26", "size": "n", "epochs": 10, "imgsz": 640, "batch": 16,
                   "batch_auto": False, "patience": 50},
    "Best accuracy": {"family": "yolo26", "size": "m", "epochs": 200, "imgsz": 640, "batch": 8,
                      "batch_auto": False, "patience": 100},
}
CUSTOM = "Custom"
PRESET_TIPS = {
    "Easy (recommended)": "yolo26 s, 100 epochs, 640 px, batch 16. A good first run on most datasets.",
    "Quick test": "yolo26 n, 10 epochs. Checks that the dataset and GPU work in a few minutes.",
    "Best accuracy": "yolo26 m, 200 epochs, batch 8, patience 100. Sized for a 12 GB card.",
    CUSTOM: "Your own values. Pick a preset to fill them in again.",
}

# ---------------------------------------------------------------- monitor
METRICS: dict[str, str] = {
    "box_loss": "How far the predicted boxes are from your labelled boxes, on the training images. "
                "It should fall quickly and then flatten. Lower is better.",
    "map50": "The main score, from 0 to 1. A box counts as correct when it overlaps the labelled box by "
             "half or more. Above 0.5 is usable, above 0.8 is very good.",
    "map50_95": "A stricter version of mAP50 that also rewards tight boxes. It is always lower than mAP50. "
                "Use it to compare runs.",
    "precision": "Of all the boxes the model drew, the share that were right. Low precision means false "
                 "alarms.",
    "recall": "Of all the real objects, the share the model found. Low recall means missed objects.",
    "chart": "mAP50 (solid) and mAP50-95 (dashed) after each epoch. The lines should rise and level off. "
             "Hover, or focus and use the arrow keys, to read values.",
}

BUTTONS: dict[str, str] = {
    "help": "Open the built-in guide: quick start, every setting explained, results and fixes (F1)",
    "theme": "Switch between dark and light theme (Ctrl+Shift+L)",
    "update": "Look for a newer LocalTrainer release on GitHub (Ctrl+U)",
    "create": "Create the project folder, or open it if it exists",
    "pick": "Pick the folder that holds images/ and labels/ (Ctrl+O)",
    "validate": "Check every image and label, then copy the dataset into the project and write data.yaml (F5)",
    "start": "Train with the settings on the left (Ctrl+Enter)",
    "start_blocked": "Validate and prepare the dataset first",
    "stop": "Stop after the current batch and keep the last saved weights (Ctrl+.). Press again to force stop.",
    "results": "Open the selected run folder in Explorer (Ctrl+R)",
    "test": "Run the best weights on an image, a folder or a video (Ctrl+T)",
    "export": "Export the best weights to ONNX next to best.pt (Ctrl+E)",
    "show_all": "Expand or collapse every settings section",
    "reset": "Put every value in this section back to its default",
    "learn_more": "Open the quick start guide (F1)",
    # banner actions, keyed by their label
    "Help": "Open the matching Help page (F1)",
    "Copy command": "Copy the repair command to the clipboard",
    "Halve the batch": "Set the batch to half its current value, then start training again",
    "Open results folder": "Open the run folder in Explorer",
    "Open output": "Open the annotated images or video",
    "Show folder": "Open the folder that holds the exported file",
}

# ---------------------------------------------------------------- validation fixes
# (text found in a report line, how to fix it)
VALIDATION: list[tuple[str, str]] = [
    ("have no matching label file",
     "Every image needs a .txt file with the same name in labels/. Export the labels again from your "
     "labelling tool, or delete the images you did not label. An image with no objects needs an empty .txt file."),
    ("label file(s) are invalid",
     "Open a listed file. Each line must be: class x_center y_center width height, five numbers separated "
     "by spaces, all four box values between 0 and 1. Re-export in YOLO format if they are not."),
    ("out of range",
     "A label uses a class id that has no name. Add class names until every id has one, or fix the ids in "
     "the label files."),
    ("expected 5 values",
     "A line has the wrong number of values. This often means the labels are in segmentation or another "
     "format. Re-export as YOLO detection (boxes)."),
    ("between 0 and 1",
     "Box values must be fractions of the image size, not pixels. Re-export with normalised YOLO coordinates."),
    ("val split has no images",
     "Put some images in images/val with their labels in labels/val, or use a flat folder and let LocalTrainer "
     "split it."),
    ("No images found",
     "Put .jpg or .png files in images/ (or images/train and images/val) inside the folder you chose."),
    ("At least 2 images",
     "Add more images. A useful model needs at least a few dozen images per class."),
    ("Add at least one class name",
     "Type a name for each class under Classes, in class-id order, starting at 0."),
    ("share the name",
     "Two images differ only by extension, for example cat.jpg and cat.png. Rename one so each has its own label."),
    ("have no matching image",
     "These label files are skipped. Delete them or add the missing images."),
    ("have no boxes",
     "No image is labelled with these classes, so the model cannot learn them. Label some, or remove the "
     "unused class names at the end of the list."),
]


def fixes_for(lines: list[str]) -> list[tuple[str, str]]:
    """Return (problem, fix) for each report line that has a known fix."""
    out = []
    for line in lines:
        for needle, fix in VALIDATION:
            if needle.lower() in line.lower():
                out.append((line, fix))
                break
    return out


def tip(text: str, width: int = 56) -> str:
    """Wrap a tooltip into short lines. Qt shows plain-text tooltips on a single line otherwise."""
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


# ---------------------------------------------------------------- empty state
EMPTY_TITLE = "Name your model to begin"
EMPTY_BODY = ("Type a name such as helmets on the left and press Create. LocalTrainer makes a folder for it "
              "under projects/ and keeps every training run there, side by side.")
EMPTY_STEPS = [
    "Pick your labelled dataset folder. It is checked and copied; the original is never changed.",
    "Choose a preset, press Start training, and watch mAP50 climb epoch by epoch.",
    "Test the best weights on an image, a folder or a video, or export them to ONNX.",
]

# ---------------------------------------------------------------- help pages
_QUICKSTART = """
<h2>Quick start</h2>
<p>Train your first model in five steps. You need a folder of images labelled in YOLO format
(one .txt file per image).</p>
<ol>
<li><b>Name your model.</b> Type a name such as <i>helmets</i> under Project and press Create.</li>
<li><b>Choose the dataset folder.</b> It must hold <i>images/</i> and <i>labels/</i>, either flat or split
into <i>train</i> and <i>val</i>. Your files are only read, never changed.</li>
<li><b>Check the classes</b> and press <b>Validate and prepare</b> (F5). Fix anything the validation report lists.</li>
<li><b>Pick a preset.</b> <i>Easy (recommended)</i> suits most datasets. Leave the advanced settings closed
unless you have a reason to change them.</li>
<li><b>Press Start training</b> (Ctrl+Enter) and watch mAP50 rise. When it finishes, press <b>Test model</b>
to try it on new images.</li>
</ol>
<p>The trained model is saved as <i>best.pt</i> in
<i>projects/&lt;name&gt;/runs/&lt;run&gt;/weights/</i>.</p>
"""


def _settings_page() -> str:
    parts = ["<h2>Settings explained</h2>",
             "<p>The settings that matter most are always visible. The rest sit in sections you open with "
             "the + sign. Closed sections still apply their values. A dot on a section header means a value "
             "differs from the default; <i>Reset to defaults</i> puts it back.</p>",
             "<h3>Essentials</h3>"]
    for key in ("preset", "family", "size", "epochs"):
        parts.append(f"<p><b>{SETTINGS[key]['label']}.</b> {SETTINGS[key]['tip']}</p>")
    for _sid, title, keys in SECTIONS:
        parts.append(f"<h3>{title}</h3>")
        for key in keys:
            parts.append(f"<p><b>{SETTINGS[key]['label']}.</b> {SETTINGS[key]['tip']}</p>")
    return "\n".join(parts)


_RESULTS = """
<h2>Understanding results</h2>
<h3>The numbers</h3>
<p><b>mAP50</b> is the main score, from 0 to 1. It is high when the model finds the objects and draws boxes
that overlap yours by at least half. <b>mAP50-95</b> is stricter and also rewards tight boxes.
<b>Precision</b> is the share of drawn boxes that are right; <b>recall</b> is the share of real objects found.
<b>Box loss</b> measures how far boxes are from the labels while training; it should fall.</p>
<h3>Did it work?</h3>
<p>As a rough guide for mAP50: below 0.3 the model has not learned much yet; 0.5 to 0.7 is usable for many
tasks; above 0.8 is very good. Simple, clear objects score higher than small or crowded ones, so compare runs
on the same dataset rather than against other projects.</p>
<h3>Overfitting</h3>
<p>Overfitting means the model memorises the training images instead of learning the objects. The sign is
box loss still falling while mAP50 on the val split stops rising or drops. Patience stops training when this
happens and keeps the best epoch as <i>best.pt</i>. More varied images help more than more epochs.</p>
<h3>Next steps</h3>
<ul>
<li>Press <b>Test model</b> and look at the boxes on images the model has never seen.</li>
<li>If objects are missed (low recall), add more labelled examples of them, or raise the image size for small objects.</li>
<li>If there are false boxes (low precision), add images of the background without the object, with empty label files.</li>
<li>For a better score, try the <i>Best accuracy</i> preset or a bigger model size.</li>
<li>Use <i>best.pt</i> in Python, or <i>best.onnx</i> in other programs.</li>
</ul>
"""

_MODELS = """
<h2>Choosing a model</h2>
<p>Every family comes in five sizes. Bigger models are more accurate but slower and need more GPU memory.
The memory figures are rough, for training at 640 px with batch 16.</p>
<table cellspacing="0" cellpadding="6">
<tr><th align="left">Size</th><th align="left">Speed</th><th align="left">Accuracy</th><th align="left">GPU memory</th></tr>
<tr><td>n (nano)</td><td>Fastest</td><td>Lowest</td><td>about 3 GB</td></tr>
<tr><td>s (small)</td><td>Fast</td><td>Good</td><td>about 4 GB</td></tr>
<tr><td>m (medium)</td><td>Medium</td><td>Better</td><td>about 8 GB</td></tr>
<tr><td>l (large)</td><td>Slow</td><td>High</td><td>about 10 GB</td></tr>
<tr><td>x (extra large)</td><td>Slowest</td><td>Highest</td><td>14 GB or more</td></tr>
</table>
<h3>On a 12 GB card</h3>
<p>Start with <b>s</b>: it trains quickly and shows whether the dataset works. For the best result use
<b>m</b> with batch 8 to 16 (the <i>Best accuracy</i> preset). <b>l</b> fits with batch 8. <b>x</b> needs
batch 4 or Auto. If you see an out-of-memory error, halve the batch first, then lower the image size.</p>
<h3>Family</h3>
<p><b>yolo26</b> is the newest and the recommended choice. <b>yolo11</b> and <b>yolov8</b> are older and well
tested; pick them to match an existing project. Weights download once into <i>weights/</i> on first use.</p>
"""

_TROUBLESHOOTING = """
<h2>Troubleshooting</h2>
<h3><a name="oom"></a>Out of GPU memory</h3>
<p>The message says <i>CUDA out of memory</i>. Press <b>Halve the batch</b> in the message, or lower Batch in
Advanced training settings, then start again. If batch 4 still fails, lower Image size to 512 or pick a smaller
model size. Close games and other programs that use the GPU.</p>
<h3><a name="cuda"></a>CUDA not detected, training uses the CPU</h3>
<p>The status bar shows <i>CUDA: not available</i>. Check that an NVIDIA GPU is installed and its driver is
up to date (Windows Update or nvidia.com). Then copy the command from the yellow message and run it in the
LocalTrainer folder; it reinstalls PyTorch with GPU support. Restart LocalTrainer afterwards.</p>
<h3><a name="validation"></a>Validation errors</h3>
<p>The validation report lists each problem with a fix under <i>How to fix</i>. The most common causes:
images without a label file, labels exported in a format other than YOLO detection, box values in pixels
instead of 0 to 1, and class ids that have no name. Fix the files in your labelling tool, export again and
press Validate and prepare.</p>
<h3><a name="stuck"></a>Training seems stuck</h3>
<p>The first epoch can take a minute while the model downloads and labels are scanned. If nothing moves for
several minutes before the first epoch, set Workers to 0 in Advanced training settings and start again.
If every epoch is very slow, check the status bar: training may be running on the CPU.</p>
<h3><a name="training"></a>Training failed</h3>
<p>Read the last lines of the Log tab. Out-of-memory errors are covered above. A missing file usually means the
dataset changed after it was prepared: press Validate and prepare again.</p>
<h3><a name="update"></a>Update problems</h3>
<p>If the update check fails, check your internet connection and try again later; GitHub limits how often
it can be asked. If an update fails, LocalTrainer rolls back to the old version automatically and keeps your
projects and settings. You can always download the latest zip from the GitHub releases page and unpack it
over the old folder.</p>
"""

_GLOSSARY = """
<h2>Glossary</h2>
<p><b>Epoch.</b> One full pass over every training image.</p>
<p><b>Batch.</b> The number of images the GPU processes in one step.</p>
<p><b>Image size.</b> The size in pixels that images are resized to before training.</p>
<p><b>mAP.</b> Mean average precision, the standard score for object detection, from 0 to 1.
mAP50 counts a box as right when it overlaps the label by half; mAP50-95 averages stricter overlaps too.</p>
<p><b>Precision.</b> The share of predicted boxes that are correct.</p>
<p><b>Recall.</b> The share of real objects that the model found.</p>
<p><b>Loss.</b> How wrong the model is on the training images. Training tries to make it smaller.</p>
<p><b>Validation set.</b> Images kept aside and never trained on, used to score the model fairly.</p>
<p><b>best.pt and last.pt.</b> The saved model. <i>best.pt</i> is the epoch with the highest score;
<i>last.pt</i> is the most recent epoch, used to resume.</p>
<p><b>ONNX.</b> An open model format that many programs and devices can run without Python.</p>
<p><b>Class.</b> One kind of object to find, such as helmet or head. Class ids start at 0.</p>
"""

# (id, title, html)
HELP_SECTIONS: list[tuple[str, str, str]] = [
    ("quickstart", "Quick start", _QUICKSTART),
    ("settings", "Settings explained", _settings_page()),
    ("results", "Understanding results", _RESULTS),
    ("models", "Choosing a model", _MODELS),
    ("troubleshooting", "Troubleshooting", _TROUBLESHOOTING),
    ("glossary", "Glossary", _GLOSSARY),
]
