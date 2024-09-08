import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def ROC_classify(positive, negative, plot=True):
    """
    Classify the data using the ROC curve.
    """

    # Combine the positive and negative data into a single array
    scores = np.concatenate([positive, negative])

    # Create an array of true labels: 1 for positive, 0 for negative
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Compute the "distance" to the point (0, 1) for each threshold
    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)

    # Find the threshold for which the distance is minimal
    best_threshold = thresholds[np.argmin(distances)]

    if plot:

        # Compute the ROC AUC (Area Under the Curve)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([min(tpr), 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return best_threshold