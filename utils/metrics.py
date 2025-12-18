"""
Utility functions for computing evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Optional
import sacrebleu
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """
    Compute chrF score.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
    
    Returns:
        chrF score (0-100)
    """
    if not predictions or not references:
        return 0.0
    
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return chrf.score


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
    
    Returns:
        BLEU score (0-100)
    """
    if not predictions or not references:
        return 0.0
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_comet(predictions: List[str], 
                  references: List[str], 
                  sources: List[str],
                  model_name: str = "Unbabel/wmt22-comet-da") -> float:
    """
    Compute COMET score.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        sources: List of source texts
        model_name: COMET model name
    
    Returns:
        COMET score
    """
    try:
        from comet import download_model, load_from_checkpoint
        import torch
        
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        
        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]
        
        results = model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        return results['system_score']
    
    except ImportError:
        print("Warning: COMET not installed. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"Warning: COMET computation failed: {e}")
        return 0.0


def compute_classification_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_proba: Optional[np.ndarray] = None,
                                   num_classes: Optional[int] = None) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC)
        num_classes: Number of classes (for AUC)
    
    Returns:
        Dictionary with accuracy, macro F1, and AUC
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Macro F1
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    
    # AUC (one-vs-rest)
    if y_proba is not None and num_classes is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            metrics['auc'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: AUC computation failed: {e}")
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


def compute_per_language_metrics(predictions: List[str],
                                 references: List[str],
                                 languages: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per language.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        languages: List of language labels for each example
    
    Returns:
        Dictionary mapping language to metrics
    """
    # Group by language
    lang_groups = {}
    for pred, ref, lang in zip(predictions, references, languages):
        if lang not in lang_groups:
            lang_groups[lang] = {'predictions': [], 'references': []}
        lang_groups[lang]['predictions'].append(pred)
        lang_groups[lang]['references'].append(ref)
    
    # Compute metrics per language
    results = {}
    for lang, data in lang_groups.items():
        preds = data['predictions']
        refs = data['references']
        
        results[lang] = {
            'chrf': compute_chrf(preds, refs),
            'bleu': compute_bleu(preds, refs),
            'count': len(preds)
        }
    
    return results


def compute_length_ratio(predictions: List[str], references: List[str]) -> float:
    """
    Compute average length ratio between predictions and references.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
    
    Returns:
        Average length ratio
    """
    if not predictions or not references:
        return 0.0
    
    ratios = []
    for pred, ref in zip(predictions, references):
        if len(ref) > 0:
            ratio = len(pred) / len(ref)
            ratios.append(ratio)
    
    return np.mean(ratios) if ratios else 0.0


def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for a list of scores.
    
    Args:
        scores: List of scores
        confidence: Confidence level
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not scores:
        return (0.0, 0.0, 0.0)
    
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    
    # Using normal approximation
    z = 1.96 if confidence == 0.95 else 2.576  # for 95% or 99%
    margin = z * (std / np.sqrt(n))
    
    return (mean, mean - margin, mean + margin)
