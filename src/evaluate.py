"""
Evaluation Module
=================
Evaluates model performance with standard metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of metric values
    """
    if metrics is None:
        metrics = ["mse", "mae", "r2", "ci", "pearson", "spearman"]
    
    results = {}
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if "mse" in metrics:
        results["mse"] = np.mean((y_true - y_pred) ** 2)
    
    if "mae" in metrics:
        results["mae"] = np.mean(np.abs(y_true - y_pred))
    
    if "rmse" in metrics:
        results["rmse"] = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    if "r2" in metrics:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        results["r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    if "ci" in metrics:
        results["ci"] = concordance_index(y_true, y_pred)
    
    if "pearson" in metrics:
        if HAS_SCIPY:
            corr, pvalue = stats.pearsonr(y_true, y_pred)
        else:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            pvalue = 0.0
        results["pearson"] = corr
        results["pearson_pvalue"] = pvalue
    
    if "spearman" in metrics:
        if HAS_SCIPY:
            corr, pvalue = stats.spearmanr(y_true, y_pred)
        else:
            corr = np.corrcoef(
                np.argsort(y_true), np.argsort(y_pred)
            )[0, 1]
            pvalue = 0.0
        results["spearman"] = corr
        results["spearman_pvalue"] = pvalue
    
    return results


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Concordance Index (CI).
    
    CI measures the ranking accuracy: for a random pair (i, j),
    if y_true[i] > y_true[j], is y_pred[i] > y_pred[j]?
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        CI score (0.5 is random, 1.0 is perfect)
    """
    n = len(y_true)
    if n < 2:
        return 0.5
    
    correct = 0
    total = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                    correct += 1
    
    return correct / total if total > 0 else 0.5


def evaluate_model(
    model,
    data_loader,
    device,
    criterion=None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Torch device
        criterion: Loss function (optional)
        
    Returns:
        Dictionary of metrics
    """
    import torch
    
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            drug_data = batch["drug_data"].to(device)
            protein = batch["protein"].to(device)
            target = batch["target"].to(device)
            
            prediction = model(drug_data, protein)
            
            if criterion is not None:
                loss = criterion(prediction, target)
                total_loss += loss.item()
            
            all_predictions.extend(prediction.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    metrics = calculate_metrics(targets, predictions)
    
    if criterion is not None:
        metrics["loss"] = total_loss / len(data_loader)
    
    return metrics


def generate_report(
    metrics: Dict[str, float],
    model_name: str = "Drug-Target Predictor"
) -> str:
    """
    Generate a text report of evaluation results.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        
    Returns:
        Formatted report string
    """
    report = f"\n{'='*60}\n"
    report += f"  {model_name} - Evaluation Report\n"
    report += f"{'='*60}\n\n"
    
    report += f"  {'Metric':<20} {'Value':>10}\n"
    report += f"  {'-'*30}\n"
    
    for metric, value in metrics.items():
        if "pvalue" in metric:
            continue
        report += f"  {metric.upper():<20} {value:>10.4f}\n"
    
    report += f"\n{'='*60}\n"
    
    return report


def compare_models(
    results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Compare results from multiple models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame comparing all models
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    
    return df


# Main entry point
if __name__ == "__main__":
    # Example usage
    y_true = np.random.rand(100) * 10
    y_pred = y_true + np.random.randn(100) * 0.5
    
    metrics = calculate_metrics(y_true, y_pred)
    print(generate_report(metrics))
