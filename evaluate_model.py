import os
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO
import seaborn as sns
import numpy as np
import ultralytics
from matplotlib import pyplot as plt
from scipy.stats._fit import FitResult
import pandas as pd


def grid_evaluation(models_base_path):
    """ Extract mAP_scores, learning_rates and optimizers from the models in the models_base_path"""
    mAP_scores = []
    learning_rates = []
    optimizers = []
    for subfolder in os.listdir(models_base_path):
        if subfolder.startswith("YoloV8n_"):
            _, learning_rate, optimizer = subfolder.split("_")
            model_path = os.path.join(models_base_path, subfolder)
            model = YOLO(model_path + '/weights/best.pt')

            mAP50 = model.val(split='val', imgsz=640).box.map50

            mAP_scores.append(mAP50)
            learning_rates.append(float(learning_rate))
            optimizers.append(optimizer)

    return mAP_scores, learning_rates, optimizers


def plot_grid_evaluation(mAP_scores, learning_rates, optimizers):
    unique_optimizers = list(set(optimizers))
    unique_lrs = sorted(list(set(learning_rates)))
    
    # Create a 2D grouped bar chart
    barWidth = 0.25
    r1 = np.arange(len(unique_lrs))
    r2 = [x + barWidth for x in r1]
    
    for opt in unique_optimizers:
        scores = [mAP_scores[i] for i in range(len(mAP_scores)) if optimizers[i] == opt]
        sns.bar(r1 if opt == unique_optimizers[0] else r2, scores, width=barWidth, label=opt)
    
    # Add xticks on the middle of the group bars
    sns.xlabel('Learning Rates', fontweight='bold')
    sns.xticks([r + barWidth for r in range(len(unique_lrs))], [str(lr) for lr in unique_lrs])
    sns.ylabel('mAP Scores', fontweight='bold')
    sns.title('Comparison of Optimizers and Learning Rates to mAP Scores')
    sns.legend()
    
    sns.show()


def grid_heatmap_evaluation(mAP_scores, learning_rates, optimizers):
    """Visualize heatmap on mAP values of the models given by the lists"""
    matrix = np.array(mAP_scores).reshape((4, 3))
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, cmap="Reds", xticklabels=optimizers[:3], yticklabels=np.unique(learning_rates))
    plt.xlabel('Optimizer')
    plt.ylabel('Learning Rate')
    plt.title('mAP Scores Heatmap')
    plt.show()


def evaluate_yolo(models_path, mode):
    """ Extract the mAP50 and mAP75 from the model using ultralytics docs"""
    
    map50_results = []
    map50_95_results = []
    models = os.listdir(models_path)
    for path in models:
        if path.lower().startswith('yolo'):
            df = pd.read_csv(os.path.join(models_path, path, 'results.csv'))
            df.columns = df.columns.str.strip()

            map50 = df['metrics/mAP50(B)'].max()
            map50_9 = df['metrics/mAP50-95(B)'].max()

            map50_results.append(map50)
            map50_95_results.append(map50_9)
    print("mAP50: ", map50_results)
    print("mAP50-95: ", map50_95_results)
    return map50_results, map50_95_results


def modern_plot_map_scores(models, mAP_scores, type):
    # Set the theme for seaborn
    sns.set_theme(style="whitegrid")

    # Sort the scores in descending order for better visualization
    sorted_indices = sorted(range(len(mAP_scores)), key=lambda k: mAP_scores[k], reverse=True)
    models = [models[i] for i in sorted_indices]
    mAP_scores = [mAP_scores[i] for i in sorted_indices]

    plt.figure(figsize=(100, 100))
    # change the size of the x and y labels
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)

    # Create a barplot
    # change the size of the x and y labels
    ax = sns.barplot(x=models, y=mAP_scores, )
    # Add labels and title
    plt.xlabel('Models', fontsize=75, labelpad=80)
    plt.ylabel('mAP Score', fontsize=75, labelpad=80)
    plt.title(f'{type}-mAP Scores - Adam Yolov8 from Scratch Models', fontsize=75, pad=100)
    plt.ylim([0, max(mAP_scores) + min(mAP_scores)])  # Assuming mAP score is between 0 and 1

    # Display mAP values on each bar
    for index, value in enumerate(mAP_scores):
        ax.text(index, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=70, fontweight='bold')

    # Remove the borders for a cleaner look
    sns.despine(left=True, bottom=True)

    plt.show()
