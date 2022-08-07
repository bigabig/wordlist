import os
from omegaconf import OmegaConf
from pathlib import Path


# global config
__conf_file__ = os.getenv("CONFIG_YAML", Path(__file__).parent.resolve() / "configs/default.yaml")
print(__conf_file__)
conf = OmegaConf.load(__conf_file__)

model_dir = Path(__file__).parent.parent.resolve() / "models"
if not model_dir.exists():
    model_dir.mkdir()

datasets_dir = Path(__file__).parent.parent.resolve() / "datasets"
if not datasets_dir.exists():
    datasets_dir.mkdir()

spacy_dir = Path(__file__).parent.parent.resolve() / "spacy"
if not spacy_dir.exists():
    spacy_dir.mkdir()
