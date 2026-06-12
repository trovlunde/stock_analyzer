from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from stock_analysis.ai.helpers import classifiers


def get_classifier_param_grid():
    """Define refined parameter grids based on previous performance analysis."""
    return {
        "Linear SVM": {
            'C': [0.5, 1, 2],  # Centered around best performing C=1
            'kernel': ['linear'],
            'class_weight': ['balanced']
        },
        "Random Forest": {
            'n_estimators': [100, 150, 200],  # Increased from previous
            'max_depth': [4, 5, 6],  # Centered around best performing depth=5
            # Centered around best performing split=5
            'min_samples_split': [4, 5, 6],
            'class_weight': ['balanced'],
            'random_state': [42],
            'min_samples_leaf': [2, 3]  # New parameter to prevent overfitting
        },
        "Nearest Neighbors": {
            'n_neighbors': [4, 5, 6],  # Centered around best performing k=5
            'weights': ['distance'],  # Best performing weight type
            # Testing different distance metrics
            'metric': ['minkowski', 'manhattan']
        },
        "Voting Classifier": {
            'voting': ['soft'],
            # Testing different voting weights
            'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
        },
        "LightGBM": {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, -1],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 63],
        }
    }


def compare_classifiers(X, y, cv=5, class_weights=None, classifier_names=None):
    """
    Evaluate classifiers with class weights and stratified CV.
    """
    results = {}
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=1),
        'recall': make_scorer(recall_score, average='weighted', zero_division=1),
        'f1': make_scorer(f1_score, average='weighted', zero_division=1)
    }

    param_grid = get_classifier_param_grid()

    # Use time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv = tscv  # Replace standard CV with time series CV

    active_classifiers = (
        {n: c for n, c in classifiers.items() if n in classifier_names}
        if classifier_names is not None
        else classifiers
    )

    for name, clf in active_classifiers.items():
        if name in param_grid:
            print(f"\nTuning {name}...")

            # Add class_weight parameter if supported by classifier
            if hasattr(clf, 'class_weight') and class_weights is not None:
                param_grid[name]['class_weight'] = [class_weights]

            grid = GridSearchCV(
                clf,
                param_grid[name],
                scoring=scoring,
                refit='f1',
                cv=cv,  # Using stratified CV
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X, y)

            results[name] = {
                'best_params': grid.best_params_,
                'best_score': grid.best_score_,
                'cv_results': {
                    metric: grid.cv_results_[
                        f'mean_test_{metric}'][grid.best_index_]
                    for metric in scoring.keys()
                }
            }

    return results


def plot_classifier_comparison(results):
    """Plot comparison of classifier performances."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    n_metrics = len(metrics)
    n_classifiers = len(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.15
    x = np.arange(n_classifiers)

    for i, metric in enumerate(metrics):
        scores = [results[clf]['cv_results'][metric] for clf in results.keys()]
        ax.bar(x + i*width, scores, width, label=metric.capitalize())

    ax.set_ylabel('Score')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x + width * (n_metrics-1)/2)
    ax.set_xticklabels(results.keys(), rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def print_best_configs(results):
    """Print the best configuration and scores for each classifier."""
    print("\nBest Configurations and Scores:")
    print("-" * 80)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Best Parameters: {result['best_params']}")
        print("Scores:")
        for metric, score in result['cv_results'].items():
            print(f"  {metric.capitalize()}: {score:.4f}")
        print("-" * 80)
