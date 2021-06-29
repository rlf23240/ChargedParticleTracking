from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from torch.nn import Module
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from cpt_data_preprocessing import HitGraph
from cpt_data_reader import HitGraphDataLoader, HitGraphDataset


def plot_confusion_matrix(model: Module, graph: HitGraph, save: Path = None):
    data_loader = HitGraphDataLoader(
        dataset=HitGraphDataset(
            graphs=[graph]
        )
    )

    # Mark model for evaluations.
    model.eval()

    # Use for loop to unpack data. It actually has only one graph.
    for batch_idx, (batch_input, batch_target) in enumerate(data_loader):
        # Evaluate model and remove batch dimension
        predictions = model(batch_input).squeeze().detach().numpy()

        """
        Confusion matrix for test data.
        """
        y_true = graph.truth
        y_pred = [1 if x > 0.5 else 0 for x in predictions]

        conf_matrix = confusion_matrix(y_true, y_pred, normalize='all')

        plt.figure(figsize=(8, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            xticklabels=['Fake Edge', 'Real Edge'],
            yticklabels=['Fake Edge', 'Real Edge']
        )
        plt.title("Confusion Matrix")

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def plot_auc_roc(model: Module, graph: HitGraph, save: Path = None):
    data_loader = HitGraphDataLoader(
        dataset=HitGraphDataset(
            graphs=[graph]
        )
    )

    # Mark model for evaluations.
    model.eval()

    # Use for loop to unpack data. It actually has only one graph.
    for batch_idx, (batch_input, batch_target) in enumerate(data_loader):
        # Evaluate model and remove batch dimension
        predictions = model(batch_input).squeeze().detach().numpy()

        """
        Confusion matrix for test data.
        """
        y_true = graph.truth
        y_pred = predictions

        auc = metrics.roc_auc_score(y_true, y_pred)

        fp, tp, _ = metrics.roc_curve(y_true, y_pred)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.plot(fp, tp, linewidth=3, label='AUC={0:.2f}'.format(auc))
        plt.legend()

        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.title("ROC Curve")

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def plot_loss_curve(history, save: Path = None):
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.set_title('Loss Function')

    ax.plot(train_loss, label='Train Loss')
    ax.plot(val_loss, label='Validation Loss')

    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')

    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True)
    )

    ax.legend()

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def plot_acc_curve(history, save: Path = None):
    val_acc = history['val_acc']

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.set_title('Accuracy')

    ax.plot(val_acc, label='Validation Accuracy')

    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')

    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True)
    )

    ax.legend()

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
