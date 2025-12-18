"""
Utility module initialization.
"""

from .data_utils import (
    hash_text,
    normalize_text,
    clean_translation,
    check_cjk_ratio,
    is_cjk_char,
    remove_duplicates,
    split_by_length,
    normalize_chinese_punctuation,
    extract_json_from_response
)

from .metrics import (
    compute_chrf,
    compute_bleu,
    compute_comet,
    compute_classification_metrics,
    compute_per_language_metrics,
    compute_length_ratio,
    compute_confidence_interval
)

from .model_utils import (
    load_model_and_tokenizer,
    generate_batch,
    save_model_with_metadata,
    count_parameters,
    get_model_device,
    clear_gpu_memory
)

__all__ = [
    # data_utils
    'hash_text',
    'normalize_text',
    'clean_translation',
    'check_cjk_ratio',
    'is_cjk_char',
    'remove_duplicates',
    'split_by_length',
    'normalize_chinese_punctuation',
    'extract_json_from_response',
    
    # metrics
    'compute_chrf',
    'compute_bleu',
    'compute_comet',
    'compute_classification_metrics',
    'compute_per_language_metrics',
    'compute_length_ratio',
    'compute_confidence_interval',
    
    # model_utils
    'load_model_and_tokenizer',
    'generate_batch',
    'save_model_with_metadata',
    'count_parameters',
    'get_model_device',
    'clear_gpu_memory',
]
