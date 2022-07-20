from pathlib import Path

model_dir = Path.cwd() / 'models'
if not model_dir.exists():
    model_dir.mkdir()

datasets_dir = Path().cwd() / "datasets"
if not datasets_dir.exists():
    datasets_dir.mkdir()