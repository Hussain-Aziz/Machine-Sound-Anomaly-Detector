from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix, RocCurveDisplay
from sklearn.manifold import TSNE
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_history(history, metric: str):
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history[f'val_{metric}'], label='validation')
    plt.title(f'{metric.title()} of Model')
    plt.xlabel('Epoch')
    plt.ylabel(metric.title())
    plt.legend()
    plt.grid()
    plt.show()    

def plot_roc(y_actual: np.ndarray, y_pred: np.ndarray, class_names: list):
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_actual[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} - AUC: {auc(fpr, tpr):.4f}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend()
    plt.grid()
    plt.show()
    

def plot_precision_recall(y_actual: np.ndarray, y_pred: np.ndarray, class_names: list):
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_actual[:, i], y_pred[:, i])
        plt.plot(recall, precision, label=f'{class_name}')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()

def to_ordinal(y):
    return np.argmax(y, axis = 1)

def tsne_plot(embeddings, labels, alpha = 1.0, title=""):
    tsne = TSNE(random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    tsne_results=pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    tsne_results['labels'] = labels
    sns.lmplot(x='tsne1', y='tsne2', data=tsne_results, hue='labels', fit_reg=False, scatter_kws={'alpha':alpha})
    plt.title(title)
    plt.show()
    
    
def display_confusion_matrix(y_test, y_pred, index = ['normal', 'abnormal']):
    conf_matrix = confusion_matrix(y_test, y_pred)
    display(pd.DataFrame(conf_matrix, index=index, columns=index))
    
    
def get_embeddings(encoder, X, X_normal_train, X_test):
    train_embeddings = encoder.predict(X_normal_train)
    train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
    test_embeddings = encoder.predict(X_test)
    test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)
    all_embeddings = encoder.predict(X)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)
    
    return train_embeddings, test_embeddings, all_embeddings
    
def plot_3_vars(df: pd.DataFrame, x: str, y: str, metric: str, title: str | None = None):
    '''
    Plots a scatter plot of a metric vs hyperparameter
    Args:
    df: A dataframe containing the 3 variables as columns
    x: The column to plot on the x-axis
    y: The column to plot on the y-axis
    metric: The column to plot as the color
    title: The title of the plot
    '''
    
    if title is None:
        title = metric
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=df, x=x, hue=metric, y=y, palette='Greens')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()

