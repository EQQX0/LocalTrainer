# Decisions

Choices I made while building LocalTrainer v1.0.0 and v1.0.1 unattended, with the reason for each.

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

## 1.0.1: auto-export ONNX

- **The export runs in the training worker, after it reports "done".** The worker already has Ultralytics loaded, so the export skips a second process start and a second torch import. The window gets the "done" event first and knows the run succeeded before the export begins.
- **The export always runs on the CPU** (`device="cpu"`). ONNX export gains nothing from the GPU, and the CPU path avoids a CUDA memory spike right after training.
- **Stop during the export cancels only the export.** The app kills the worker at once, deletes the partial `best.onnx` and reports the run as finished. Waiting for a graceful end made no sense: the export has no checkpoint to save.
- **A force stop during training skips the export.** `best.pt` may be half written when the process dies, and exporting it could fail or produce a broken model.
- **A failed export shows a warning banner, not an error.** The run's weights are fine, so the title still reads "Training finished", with "ONNX export failed" appended and a hint to retry with Export ONNX.
- **New requirements.** `onnx` and `onnxslim` were already listed for the manual export. I added `onnxruntime>=1.20` so you can run the exported model, and so the updater test exercises a real `pip install` on a requirements change.
- **The option lives in each project's `settings.json`** with the other training settings. A 1.0.0 file has no key for it, and the loader gives missing keys their default, which is on.

## Verification notes

- `main.py` quits on its own when `LOCALTRAINER_SMOKE=1` is set. The headless smoke test uses this.
- The coco8 test projects and the downloaded COCO8 sample from the 1.0.0 and 1.0.1 checks are deleted.
- **Live update test, 1.0.0 to 1.0.1.** I unzipped the v1.0.0 release asset into a scratch folder and linked the repo's `.venv` into it with a junction. Then I seeded a `projects/helmets` project, a root `settings.json` and three dated old backups. A `sitecustomize.py` hook on `PYTHONPATH` pressed Check for updates and then Update now in the real dialog, and logged what each window showed. The helper runs with `-I`, so the hook never ran inside it. The update took about 20 s against live GitHub. pip ran because `requirements.txt` changed and found every package installed. The helper pruned the oldest seeded backup, kept the newest 3, and its new backup holds the 1.0.0 files. The relaunched app read `update_status.json`, deleted it, and showed "Updated to version 1.0.1".
- **How I checked "byte-identical".** The app saves its window geometry to `settings.json` when its window closes, and the old app closes as part of updating. A plain before and after comparison shows that write, so I hashed `settings.json` and every file under `projects/` twice: in the 1.0.0 process as it quit, and in the 1.0.1 process before `main.py` ran. All 7 files matched across the helper. The venv's package list, `pip freeze` output and `pyvenv.cfg` matched too.
- **Corrupt download test.** Using the installed copy's `stage_update`, I streamed the live v1.0.1 zip with one byte flipped and checked it against the real `.sha256` asset. It raised `ChecksumError` before extracting anything, and the 82 app files kept their hashes. A control run without the flip staged cleanly.
- **Temp folders stayed behind in 1.0.1.** The updater stages each download in `%TEMP%\LocalTrainer-update-<timestamp>` and no step deleted it. I deleted the folders my tests made, along with the scratch install. The next section adds the cleanup.

## Unreleased: temp folder cleanup

- **Three places delete staging folders.** The helper deletes its own folder after it writes `update_status.json` and before it restarts the app. The update dialog deletes the folder when a download fails or you cancel it. At startup the app sweeps `%TEMP%` for `LocalTrainer-update-*` folders on a background thread, so a slow disk cannot freeze the window.
- **The startup sweep skips folders changed in the last hour.** LocalTrainer has no single-instance lock. A second window could be downloading an update into one of those folders, or the helper could still be copying files out of it during a long `pip install`.
- **The helper now starts in the app folder.** My first real check failed: the helper logged "Could not delete" and left an empty folder. The venv's `python.exe` is a launcher. It starts the real interpreter as a child process and holds its own working directory, the work folder, until the child exits. Starting the helper with the app folder as its working directory fixed it. The helper runs with `-I`, which keeps the working directory off `sys.path`, so nothing else changes.
- **Deletion checks the folder name.** The app and the helper each refuse to delete a folder whose name lacks `LocalTrainer-update-`, so a wrong path cannot reach `projects/` or the app folder.
- **The update to the next version still leaves one folder.** The installed 1.0.1 code runs that update and has no cleanup. The new version deletes that folder on its first start an hour or more later.
- **`update.log` keeps the last 3 updates.** Each update appends a few lines, plus up to 8,000 characters of pip output when `requirements.txt` changed, and no step trimmed the file. The helper now cuts it at the start of each update, at the "Updating to" line that opens every update in 1.0.0 and later, so it never splits an update's entries. Three matches the number of backups kept in `update_backups/`. The helper writes the trimmed log to a temporary file and swaps it in, and it ignores any error there: a log file must not stop an update.
- **Real check.** I staged a package in a real `%TEMP%\LocalTrainer-update-<timestamp>` folder and started the copied helper the way `launch_helper` does: `-I`, no window, a new process group. The helper applied the update, kept `projects/` and `settings.json`, and deleted its folder. I then planted a two-day-old folder and a fresh one, and `cleanup_work_dirs()` deleted only the old one. A headless app start deleted a second planted old folder. No `LocalTrainer-update-*` folders remain in `%TEMP%`.
