# Decisions

Choices I made while building LocalTrainer v1.0.0 unattended, with the reason for each.

## Setup and publishing

- **GitHub account.** `gh auth status` shows the keyring account `beboreport1-ops`, but `gh api user` returns the login `EQQX0`. I used `EQQX0`, the API's answer. `EQQX0/LocalTrainer` existed, was public and empty, so I cloned it into `Desktop\repos\LocalTrainer`. It had no files to merge.
- **Environment.** Python 3.13.15 (the only installed version), PyTorch 2.11.0+cu128 (its arch list includes `sm_120` for the RTX 5070), Ultralytics 8.4.190, PyQt6 6.11.
- **Release zip.** `tools/build_release.py` builds the zip from `git ls-files` rather than `git archive`, so the `.bat` files keep the CRLF endings that `.gitattributes` gives them on checkout. The zip leaves out `tests/`, `tools/`, `docs/`, `.github/` and these notes.
- **CI tests before release.** The release workflow runs the unit tests and refuses to publish if the tag does not match `app/version.py`.
- **README screenshot.** I rendered a real screenshot of the app (with simulated metrics) instead of a placeholder.

## Dataset

- **The dataset is always copied, even when it is already split.** Ultralytics writes `labels/train.cache` next to the labels it reads. Training straight from your folder would add files to it, and you asked for the original to stay untouched.
- **Missing label files are errors.** You asked for every image to have a matching label. An empty `.txt` file counts as a valid background image.
- **Accepted extras.** A `valid` or `validation` folder works as the val split (Roboflow exports use `valid`). Class names come from `classes.txt`, `labels/classes.txt`, `data.yaml`, or any other YAML in the dataset root with a `names` key.
- **Split.** Seed 42, files sorted before shuffling, so the same folder always splits the same way. Each split keeps at least one image.
- **Project names.** Letters, digits, `-` and `_`, up to 64 characters, starting with a letter or digit. Windows reserved names (CON, NUL, COM1 and so on) are rejected.

## Training

- **Separate process instead of a QThread.** Training runs in `python -m core.worker` under `QProcess`. The GUI never imports torch, the window cannot freeze, and a force stop can always kill the process.
- **Stop requests travel through a flag file, not stdin.** During testing, a thread blocked on reading stdin stalled the worker's `import torch` on Windows. The worker now reads its JSON payload, the parent closes stdin, and Stop creates a small flag file under `.cache/jobs/`.
- **Graceful stop.** The `on_train_batch_end` callback sets `trainer.stop = True`. Ultralytics then ends the epoch early, validates, saves `last.pt` and `best.pt` and runs the final evaluation. A second press kills the process; weights from completed epochs stay on disk.
- **Workers fallback.** If no batch arrives within 240 s of `on_train_start` and workers > 0, the session kills the run, deletes its empty folder and restarts with `workers = 0` as `run_<timestamp>_w0`.
- **Out of memory.** Ultralytics 8.4 already retries the first epoch with a smaller batch. If training still fails, the app shows a dialog and a banner suggesting a lower batch or image size, with a button that halves the batch.
- **Files stay inside the app folder.** The worker sets `YOLO_CONFIG_DIR` to `.ultralytics/` and runs with `weights/` as its working directory, so pretrained weights and the AMP check model download there.
- **Device.** "Auto" means GPU 0 when the startup check passes and the CPU otherwise. The check runs a tiny CUDA kernel and confirms the GPU's `sm_XY` is in `torch.cuda.get_arch_list()`.

## Interface

- **Art direction.** A graphite ground with a single chartreuse accent, the colour of an annotation box. Tone steps separate the surfaces; there are no shadows, gradients or hairline dividers. Corners are 2 px and spacing follows an 8 px grid. Bahnschrift (shipped with Windows) sets headings and numbers, Segoe UI Variable sets body text, and Cascadia Mono is used only in the log. All values live as tokens in `ui/theme.py`, which builds one QSS stylesheet per theme.
- **Hand-drawn chart.** I painted the mAP50 chart with `QPainter` instead of adding pyqtgraph or matplotlib. It follows the theme tokens, uses a solid line for mAP50 and a dashed line for mAP50-95 so colour is never the only cue, and you can read values by hovering or with the arrow keys.
- **Progressive disclosure.** The dataset, model and training sections appear only after you create or open a project.
- **Spin boxes have no arrow buttons.** You type values or use the arrow keys and mouse wheel. Arrow buttons are hard to style cleanly in QSS.
- **App preferences** (theme, last project, window size) live in `settings.json` at the app root. Updates never touch it.

## Updater

- **What gets replaced.** Every top-level item in the new zip, except protected names: `projects`, `settings.json`, `.venv`, `venv`, `env`, `weights`, `.ultralytics`, `.cache`, `update_backups`, `update.log`, `update_status.json`, `.git`. The helper moves replaced folders whole into `update_backups/<timestamp>/`, which removes stale files from old versions. It keeps the newest 3 backups.
- **Rollback.** A failed copy or a failed `pip install` restores the backup. The helper writes `update_status.json`, and the app shows the result on its next start.
- **Requirements changes.** The helper compares `requirements.txt` with line endings, blank lines and comments ignored, so a CRLF checkout does not trigger a needless pip run.
- **Helper process.** The app copies `utils/update_helper.py` (standard library only) to the temp folder and starts it with the venv's Python and `-I`, detached and without a window. It waits for the app's PID to exit, then relaunches through `run.bat`.
- **Verification.** The download must match the `.sha256` asset. The zip extractor rejects paths that escape the target folder, and the package's `app/version.py` must match the release tag.

## Verification notes

- `projects/coco8_e2e/` holds the runs from my end-to-end checks (1 epoch to `best.pt`, a graceful stop after epoch 2, prediction, ONNX export). It is gitignored; delete it whenever you like. `.cache/coco8/` holds the downloaded COCO8 sample.
- `main.py` quits on its own when `LOCALTRAINER_SMOKE=1` is set. The headless smoke test uses this.
