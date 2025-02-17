from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier that can use either:
    1. Combined approach: Single classifiers trained on all classes
    2. Binary approach: Separate binary classifiers for positive and negative signals
    """

    def __init__(self, weights=None, mode='binary', threshold=0.6):
        """
        Initialize the ensemble classifier.

        Args:
            weights: Dictionary of weights for each classifier type
            mode: 'binary' or 'combined' classification approach
        """
        self.mode = mode

        # Initialize base classifiers
        base_classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'nearest_neighbors': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='minkowski',
                p=2,
            ),
            'linear_svm': SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }

        if mode == 'binary':
            # Initialize separate classifier sets for binary approach
            self.pos_classifiers = {name: clone(
                clf) for name, clf in base_classifiers.items()}
            self.neg_classifiers = {name: clone(
                clf) for name, clf in base_classifiers.items()}
            self.fitted_pos_classifiers = {}
            self.fitted_neg_classifiers = {}
        else:
            # Initialize single classifier set for combined approach
            self.classifiers = base_classifiers
            self.fitted_classifiers = {}

        # Set weights
        self.weights = weights if weights is not None else {
            'random_forest': 0.4,
            'nearest_neighbors': 0.25,
            'linear_svm': 0.35
        }

        self.threshold = threshold  # Threshold for strong signals

    def fit(self, X, y):
        """
        Fit the classifier(s) based on the selected mode.
        """
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))

        if self.mode == 'binary':
            # Create binary labels
            y_pos = (y == 2).astype(int)  # Convert 'positive' class to binary
            y_neg = (y == 0).astype(int)  # Convert 'negative' class to binary

            # Calculate weights for binary cases
            pos_classes = np.unique(y_pos)
            neg_classes = np.unique(y_neg)
            pos_weights = compute_class_weight(
                'balanced', classes=pos_classes, y=y_pos)
            neg_weights = compute_class_weight(
                'balanced', classes=neg_classes, y=y_neg)
            pos_weight_dict = dict(zip(pos_classes, pos_weights))
            neg_weight_dict = dict(zip(neg_classes, neg_weights))

            print("Training positive signal classifiers...")
            for name, clf in self.pos_classifiers.items():
                print(f"Training positive {name}...")
                if hasattr(clf, 'class_weight'):
                    clf.fit(X, y_pos)
                else:
                    clf.fit(X, y_pos)
                self.fitted_pos_classifiers[name] = clf

            print("\nTraining negative signal classifiers...")
            for name, clf in self.neg_classifiers.items():
                print(f"Training negative {name}...")
                if hasattr(clf, 'class_weight'):
                    clf.fit(X, y_neg)
                else:
                    clf.fit(X, y_neg)
                self.fitted_neg_classifiers[name] = clf

        else:  # Combined approach
            print("Training combined classifiers...")
            for name, clf in self.classifiers.items():
                print(f"Training {name}...")
                if hasattr(clf, 'class_weight'):
                    clf.fit(X, y)
                else:
                    clf.fit(X, y)
                self.fitted_classifiers[name] = clf

        return self

    def predict_proba(self, X):
        """
        Get probability predictions based on the selected mode.
        Returns probabilities for [negative, neutral, positive] classes.
        """
        if self.mode == 'binary':
            # Get probabilities for positive signals
            pos_predictions = {}
            for name, clf in self.fitted_pos_classifiers.items():
                pos_predictions[name] = clf.predict_proba(X)[:, 1]

            # Get probabilities for negative signals
            neg_predictions = {}
            for name, clf in self.fitted_neg_classifiers.items():
                neg_predictions[name] = clf.predict_proba(X)[:, 1]

            # Calculate weighted probabilities
            weighted_pos_prob = np.zeros(len(X))
            weighted_neg_prob = np.zeros(len(X))

            for name in self.weights.keys():
                weighted_pos_prob += self.weights[name] * pos_predictions[name]
                weighted_neg_prob += self.weights[name] * neg_predictions[name]

            # Combine into final probability array [negative, neutral, positive]
            final_probas = np.zeros((len(X), 3))
            final_probas[:, 0] = weighted_neg_prob  # Negative class
            final_probas[:, 2] = weighted_pos_prob  # Positive class

            # Calculate neutral probability
            neutral_prob = 1.0 - (weighted_pos_prob + weighted_neg_prob)
            neutral_prob = np.maximum(neutral_prob, 0)  # Ensure non-negative
            final_probas[:, 1] = neutral_prob  # Neutral class

        else:  # Combined approach
            predictions = {}
            for name, clf in self.fitted_classifiers.items():
                predictions[name] = clf.predict_proba(X)

            # Initialize weighted sum with zeros
            final_probas = np.zeros_like(
                predictions[list(predictions.keys())[0]])

            # Add weighted predictions
            for name, pred in predictions.items():
                final_probas += self.weights[name] * pred

        # Normalize probabilities to sum to 1
        row_sums = final_probas.sum(axis=1)
        final_probas = final_probas / row_sums[:, np.newaxis]

        return final_probas

    def predict(self, X):
        """
        Make final predictions using the probability predictions.
        Returns 0 for negative, 1 for neutral, 2 for positive.
        """
        probas = self.predict_proba(X)

        if self.mode == 'binary':
            # Initialize predictions as neutral (1)
            predictions = np.ones(len(X))

            # Override with positive/negative predictions only when signal is strong
            pos_probs = probas[:, 2]  # Positive class probabilities
            neg_probs = probas[:, 0]  # Negative class probabilities

            # Set positive predictions where positive probability is highest and above threshold
            pos_mask = (pos_probs > self.threshold) & (pos_probs > neg_probs)
            predictions[pos_mask] = 2

            # Set negative predictions where negative probability is highest and above threshold
            neg_mask = (neg_probs > self.threshold) & (neg_probs > pos_probs)
            predictions[neg_mask] = 0

            return predictions.astype(int)
        else:
            # For combined approach, simply take the highest probability class
            return np.argmax(probas, axis=1)


def test_ensemble(X_train, X_test, y_train, y_test):
    """
    Test the ensemble classifier and print detailed performance metrics.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create and train ensemble
    ensemble = EnsembleClassifier()
    ensemble.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return ensemble
