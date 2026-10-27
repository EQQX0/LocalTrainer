# Changelog

## 1.1.0

### Added

- **Simple view.** The sidebar shows only what you need to start: model name, dataset folder, family, size, epochs and Start training. Training, Hardware, Data and Output settings sit in sections that start closed. Click a header, or press Enter or Space on it, to open it. Show all and Hide all open or close every section at once. LocalTrainer remembers which sections you left open.
- **Presets.** Easy, Quick test and Best accuracy fill in the model and training settings for you. Change any of those values and the preset switches to Custom.
- **Reset per section.** A dot on a section header means a value inside differs from the default. Reset to defaults puts that section back.
- **A "?" next to every setting.** Hover it or tab to it and an explanation in plain words appears after a short delay. Tiles, the chart, the buttons and the validation report have explanations too.
- **Help (F1).** Press F1 or the Help button for a guide with six sections: Quick start, Settings explained, Understanding results, Choosing a model, Troubleshooting and a glossary. Type in the search box to filter it. The validation report and error messages link to the matching Help section.
- **First-start guide.** Before you open a project, the main area shows the three steps to a first model.

### Changed

- When the window is narrow or Windows scales the screen to 150 %, the run tools (run picker, Open results, Test, Export) move to a second line under Start and Stop.
- The chart needs less height, so the tabs stay visible at 1100x700.

Training, the settings files and your datasets work as in 1.0.3.

## 1.0.3

### Added

- **One-click installer.** Each release now has `LocalTrainer-Setup-<version>.exe`. Double-click it to install LocalTrainer to `%LOCALAPPDATA%\Programs\LocalTrainer`, run the setup, add the desktop shortcut and start the app. Running it on an existing install keeps your projects and settings.
- **Desktop shortcut.** `setup.bat` puts a LocalTrainer shortcut with the app icon on your desktop. It starts the app without a console window. Installs that already ran setup get the shortcut on their first start after the update, once. If you delete it, it stays deleted.

## 1.0.2

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
