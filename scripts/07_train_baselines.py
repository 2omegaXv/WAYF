"""
Training script for baseline classifiers.
Implements step 2.3 from project procedure.
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import pickle


def load_data(data_file: Path):
    """Load translation data."""
    texts = []
    labels = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item.get('zh_mt', ''))
            labels.append(item.get('src_lang', ''))
    
    return texts, labels


def compute_metrics(y_true, y_pred, y_proba=None, label_encoder=None):
    """Compute evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    auc = 0.0
    if y_proba is not None and label_encoder is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=label_encoder.classes_)
            auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except:
            pass
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auc': auc
    }


def train_majority_baseline(train_labels, test_labels):
    """Train majority class baseline."""
    # Find majority class
    from collections import Counter
    majority_class = Counter(train_labels).most_common(1)[0][0]
    
    # Predict majority class for all test samples
    predictions = [majority_class] * len(test_labels)
    
    metrics = compute_metrics(test_labels, predictions)
    return metrics, majority_class


def train_stratified_random_baseline(train_labels, test_labels):
    """Train stratified random baseline."""
    from collections import Counter
    
    # Get class distribution from training data
    train_dist = Counter(train_labels)
    total = sum(train_dist.values())
    class_probs = {cls: count / total for cls, count in train_dist.items()}
    
    # Random predictions following training distribution
    classes = list(class_probs.keys())
    probs = list(class_probs.values())
    predictions = np.random.choice(classes, size=len(test_labels), p=probs)
    
    metrics = compute_metrics(test_labels, predictions)
    return metrics, class_probs


def train_ngram_baseline(train_texts, train_labels, test_texts, test_labels, 
                         ngram_range=(3, 6), model_type='logistic'):
    """Train character n-gram baseline with Logistic Regression or Linear SVM."""
    
    # Vectorize with character n-grams
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=ngram_range,
        max_features=10000
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # Train model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'svm':
        model = LinearSVC(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, train_labels)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Probabilities (if available)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    
    metrics = compute_metrics(test_labels, predictions, y_proba, model)
    
    return metrics, {'vectorizer': vectorizer, 'model': model}


def train_fasttext_baseline(train_file: Path, test_file: Path, output_dir: Path):
    """Train fastText baseline (requires fasttext library)."""
    try:
        import fasttext
    except ImportError:
        print("fasttext not installed. Skipping fastText baseline.")
        return None, None
    
    # Prepare data in fastText format
    train_ft = output_dir / "train_fasttext.txt"
    test_ft = output_dir / "test_fasttext.txt"
    
    def write_fasttext_format(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                item = json.loads(line)
                text = item.get('zh_mt', '').replace('\n', ' ')
                label = item.get('src_lang', '')
                f_out.write(f"__label__{label} {text}\n")
    
    write_fasttext_format(train_file, train_ft)
    write_fasttext_format(test_file, test_ft)
    
    # Train fastText model
    model = fasttext.train_supervised(
        input=str(train_ft),
        lr=0.5,
        epoch=25,
        wordNgrams=3,
        dim=100
    )
    
    # Evaluate
    result = model.test(str(test_ft))
    
    metrics = {
        'accuracy': result[1],
        'macro_f1': 0.0,  # fastText doesn't provide F1 directly
        'auc': 0.0
    }
    
    return metrics, model


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifiers")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_texts, train_labels = load_data(Path(args.train_file))
    test_texts, test_labels = load_data(Path(args.test_file))
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    results = {}
    
    # 1. Majority baseline
    print("\n=== Training Majority Baseline ===")
    metrics, model = train_majority_baseline(train_labels, test_labels)
    results['majority'] = metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")
    
    # 2. Stratified random baseline
    print("\n=== Training Stratified Random Baseline ===")
    metrics, model = train_stratified_random_baseline(train_labels, test_labels)
    results['stratified_random'] = metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")
    
    # 3. Character n-gram + Logistic Regression
    print("\n=== Training Character N-gram + Logistic Regression ===")
    metrics, model = train_ngram_baseline(train_texts, train_labels, test_texts, test_labels,
                                         ngram_range=(3, 6), model_type='logistic')
    results['ngram_logistic'] = metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # Save model
    with open(output_dir / "ngram_logistic.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # 4. Character n-gram + Linear SVM
    print("\n=== Training Character N-gram + Linear SVM ===")
    metrics, model = train_ngram_baseline(train_texts, train_labels, test_texts, test_labels,
                                         ngram_range=(3, 6), model_type='svm')
    results['ngram_svm'] = metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # Save model
    with open(output_dir / "ngram_svm.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # 5. fastText (optional)
    print("\n=== Training fastText Baseline ===")
    metrics, model = train_fasttext_baseline(Path(args.train_file), Path(args.test_file), output_dir)
    if metrics:
        results['fasttext'] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        model.save_model(str(output_dir / "fasttext_model.bin"))
    
    # Save all results
    with open(output_dir / "baseline_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Summary ===")
    for name, metrics in results.items():
        print(f"{name}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}, AUC={metrics.get('auc', 0):.4f}")


if __name__ == "__main__":
    main()
