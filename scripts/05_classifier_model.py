"""Backward-compat wrapper.

The training script imports `classifier_model.py`.
This file is kept to match the project procedure step numbering.
"""

from classifier_model import FrozenBackboneClassifier, SourceLanguageClassifier, resolve_pretrained_source

__all__ = [
    "SourceLanguageClassifier",
    "FrozenBackboneClassifier",
    "resolve_pretrained_source",
]
