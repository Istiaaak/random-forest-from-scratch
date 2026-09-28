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

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    
    d = {
        "feature_index": None,
        "threshold": None,
        "score": 0.0
    }

    for fi in feature_indices:
        col = features[:, fi]
        values = np.unique(col)
        threshold = (values[:-1] + values[1:]) /2

        for t in threshold:
            lf, ll, rf, rl = split_dataset(features, labels, fi, t)
            if ll is not None and rl is not None:
                s = split_score(labels, ll, rl)

                if s > d["score"]:
                    d["score"] = s 
                    d["threshold"] = t 
                    d["feature_index"] = fi
    
    return d

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    if len(np.unique(labels)) == 1:
        return True
    elif depth >= max_depth:
        return True
    elif len(labels) < min_samples_split:
        return True
        
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    values, counts = np.unique(labels, return_counts=True)
    
    return int(values[np.argmax(counts)])

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

