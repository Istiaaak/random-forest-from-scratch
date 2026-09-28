"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    
    if labels is None or labels.shape == 1:
        return 0.0
    
    unique_values, counts = np.unique(labels, return_counts=True)
    prop = counts/len(labels)
    G = 1 - np.sum(prop**2)

    return G

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    col = features[:, feature_index]
    mask = col <= threshold
    left, right = features[mask], features[~mask]
    left_labels, right_labels = labels[mask], labels[~mask]

    return left, left_labels, right, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    p_len = len(parent_labels)
    l_len = len(left_labels)
    r_len = len(right_labels)

    G = impurity(parent_labels) - ((l_len/p_len)* impurity(left_labels) + (r_len/p_len)* impurity(right_labels))
    return G

# Step 4 - best_split (not yet solved)
# TODO: implement

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

