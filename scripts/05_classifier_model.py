"""
Classifier model for source language detection from Chinese translations.
Implements step 2.1 from project procedure.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional


class SourceLanguageClassifier(nn.Module):
    """
    Chinese RoBERTa-based classifier for detecting source language from translation.
    Uses [CLS] token representation with MLP head.
    """
    
    def __init__(self, 
                 num_languages: int,
                 model_name: str = "hfl/chinese-roberta-wwm-ext",
                 dropout: float = 0.1,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.num_languages = num_languages
        
        # Load pretrained Chinese RoBERTa
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        # Classification head (MLP)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_languages)
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size]
        
        Returns:
            Dictionary with 'loss' and 'logits'
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(cls_output)  # [batch_size, num_languages]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def get_tokenizer(self, model_name: str = "hfl/chinese-roberta-wwm-ext"):
        """Get the tokenizer for this model."""
        return AutoTokenizer.from_pretrained(model_name)


class FrozenBackboneClassifier(nn.Module):
    """
    Baseline classifier with frozen backbone and only linear head trained.
    """
    
    def __init__(self,
                 num_languages: int,
                 model_name: str = "hfl/chinese-roberta-wwm-ext"):
        super().__init__()
        
        self.num_languages = num_languages
        
        # Load pretrained Chinese RoBERTa and freeze
        self.backbone = AutoModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.hidden_size = self.backbone.config.hidden_size
        
        # Simple linear head
        self.classifier = nn.Linear(self.hidden_size, num_languages)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with frozen backbone."""
        
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def get_tokenizer(self, model_name: str = "hfl/chinese-roberta-wwm-ext"):
        """Get the tokenizer for this model."""
        return AutoTokenizer.from_pretrained(model_name)
