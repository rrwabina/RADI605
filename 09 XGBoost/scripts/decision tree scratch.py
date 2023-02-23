import matplotlib.pyplot as plt
import numpy as np

def find_split(X, y, n_classes):
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, None
    
    feature_ix, threshold = None, None
    sample_per_class_parent = [np.sum(y == c) for c in range(n_classes)] #[2, 2]
    best_gini = 1.0 - sum((n / n_samples) ** 2 for n in sample_per_class_parent)
    print(f'Best gini of dataset: {best_gini}')

    for feature in range(n_features):
        sample_sorted = sorted(X[:, feature])
        print(f'Sorted values: {sample_sorted}')
        sort_idx = np.argsort(X[:, feature])
        y_sorted = y[sort_idx]
                
        sample_per_class_left = [0] * n_classes 
        sample_per_class_right = sample_per_class_parent.copy()
        for i in range(1, n_samples):
            c = y_sorted[i - 1] 

            sample_per_class_left[c]  += 1
            sample_per_class_right[c] -= 1
            
            gini_left  = 1.0 - sum((sample_per_class_left[x] / i) ** 2 for x in range(n_classes))
            gini_right = 1.0 - sum((sample_per_class_right[x] / (n_samples - i)) ** 2 for x in range(n_classes))
            
            weighted_gini = ((i / n_samples) * gini_left) + ( (n_samples - i) /n_samples) * gini_right
            if sample_sorted[i] == sample_sorted[i - 1]:
                continue

            if weighted_gini < best_gini:
                print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Yes: weighted_gini: {weighted_gini} < best_gini: {best_gini}')
                best_gini = weighted_gini
                feature_ix = feature
                threshold = (sample_sorted[i] + sample_sorted[i - 1]) / 2
            print(f'============Feature {feature} | Value: \t{sample_sorted[i]} ============')
            print(f'Factor Left:    \t{i}/{n_samples}')
            print(f'Gini Left:      \t{np.round(gini_left, 2)}')
            print(f'Factor Right:   \t{n_samples - i}/{n_samples}')
            print(f'Gini Right:     \t{np.round(gini_right, 2)}')
            print()
            print(f'Equation:       \t({i}/{n_samples} * {np.round(gini_left, 2)}) + ({n_samples - i}/{n_samples} * {np.round(gini_right, 2)})')
            print(f'Weighted Gini:  \t{np.round(weighted_gini, 2)}')
            print()
    return feature_ix, threshold

data = np.array([[1, 4.8, 3.4, 1.9, 0.2, 1],
                 [2, 5, 3, 1.6, 1.2, 1],
                 [3, 5, 3.4, 1.6, 0.2, 1],
                 [4, 5.2, 3.5, 1.5, 0.2, 1],
                 [5, 4.8, 3.1, 1.6, 0.2, 1],
                 [6, 5.4, 3.4, 1.5, 0.4, 1],
                 [7, 6.4, 3.2, 4.7, 1.5, 0],
                 [8, 6.9, 3.1, 4.9, 1.5, 0],
                 [9, 5.5, 2.3, 4, 1.3, 0],
                 [10, 6.5, 2.8, 4.6, 1.5, 0],
                 [11, 5.7, 2.8, 4.5, 1.3, 0],
                 [12, 6.3, 3.3, 4.7, 1.6, 0]])

X = data[:, 4:5]
y = data[:, 5].astype('int')

feature_ix, threshold = find_split(X, y, 2)
print(f'Threshold: {threshold}')