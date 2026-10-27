# Changelog

## Unreleased

### Fixed

- Updates leave no `%TEMP%\LocalTrainer-update-<timestamp>` folders behind. The update helper deletes its folder when it finishes, and the update dialog deletes it when a download fails or you cancel it.
- At startup, LocalTrainer deletes update folders older than an hour, including the ones 1.0.0 and 1.0.1 left. It keeps newer folders, since another LocalTrainer window may still use them.
- `update.log` no longer grows with every update. The update helper keeps the entries of the last 3 updates and drops older ones.

## 1.0.1

### Added

- **Auto-export ONNX after training.** A new checkbox in the training settings, on by default. LocalTrainer exports `best.pt` to `best.onnx` in the same `weights/` folder after a run finishes, stops early or you stop it. The log and the message banner show the file's path and size.
- Cancel export: pressing Stop during the export cancels only the export and removes the partial `best.onnx`. The run and its weights stay.
- `onnxruntime` in `requirements.txt`, so you can load the exported model straight away. The in-app updater installs it for you.

### Changed

- A failed ONNX export, automatic or manual, now reports the error and keeps the run marked as finished. It never deletes weights.
- Manual Export ONNX also shows the file size.

Settings files from 1.0.0 lack the new option, so LocalTrainer turns it on for them.

## 1.0.0

First release.
