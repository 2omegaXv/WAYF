"""Classifier model for source language detection from Chinese translations.

This module is imported by training / evaluation scripts.

It supports two robust loading modes:
- Online-first: download (once) and cache locally (HF cache or a user-specified local directory)
- Offline: load strictly from disk (no network)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def resolve_pretrained_source(
    model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Resolve a Hugging Face `from_pretrained` source.

    - If `model_name_or_path` is already a local path, it is returned.
    - If `local_dir` is provided and `model_name_or_path` is a Hub repo id, the
      model files are snapshotted into `local_dir` (download once, reuse later).

    Returns:
        A string suitable for `AutoModel.from_pretrained(...)`.
    """

    candidate_path = Path(model_name_or_path)
    if candidate_path.exists():
        return str(candidate_path)

    if local_dir is None:
        return model_name_or_path

    from huggingface_hub import snapshot_download

    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    # Snapshot the whole repo into `local_dir` (no symlinks for portability).
    snapshot_download(
        repo_id=model_name_or_path,
        local_dir=str(local_dir_path),
        local_dir_use_symlinks=False,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
        token=token,
    )

    return str(local_dir_path)


class SourceLanguageClassifier(nn.Module):
    """Chinese RoBERTa-based classifier for detecting source language from translation.

    Uses [CLS] token representation with an MLP head.
    """

    def __init__(
        self,
        num_languages: int,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        dropout: float = 0.1,
        hidden_dim: int = 512,
        *,
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ):
        super().__init__()

        self.num_languages = num_languages
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.local_files_only = local_files_only
        self.revision = revision
        self.token = token

        resolved_source = resolve_pretrained_source(
            model_name,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )

        # Load pretrained Chinese RoBERTa
        self.backbone = AutoModel.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )
        self.hidden_size = self.backbone.config.hidden_size

        # Classification head (MLP)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_languages),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def get_tokenizer(self) -> AutoTokenizer:
        resolved_source = resolve_pretrained_source(
            self.model_name,
            cache_dir=self.cache_dir,
            local_dir=self.local_dir,
            local_files_only=self.local_files_only,
            revision=self.revision,
            token=self.token,
        )
        return AutoTokenizer.from_pretrained(
            resolved_source,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            revision=self.revision,
            token=self.token,
        )


class FrozenBackboneClassifier(nn.Module):
    """Baseline classifier with frozen backbone and only linear head trained."""

    def __init__(
        self,
        num_languages: int,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        *,
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ):
        super().__init__()

        self.num_languages = num_languages
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.local_files_only = local_files_only
        self.revision = revision
        self.token = token

        resolved_source = resolve_pretrained_source(
            model_name,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )

        self.backbone = AutoModel.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_languages)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def get_tokenizer(self) -> AutoTokenizer:
        resolved_source = resolve_pretrained_source(
            self.model_name,
            cache_dir=self.cache_dir,
            local_dir=self.local_dir,
            local_files_only=self.local_files_only,
            revision=self.revision,
            token=self.token,
        )
        return AutoTokenizer.from_pretrained(
            resolved_source,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            revision=self.revision,
            token=self.token,
        )
