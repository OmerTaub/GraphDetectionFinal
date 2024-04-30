import os
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO
import seaborn as sns
import numpy as np
import ultralytics
from matplotlib import pyplot as plt
from scipy.stats._fit import FitResult


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
    map75_results = []
    models = os.listdir(models_path)
    # iterate over all models
    for path in models:
        print(path)
        if path.startswith('Yolo'):
            model_path = models_path + '/' + path
            model = YOLO(model_path + '/weights/best.pt')
            metrics = model.val(split=mode, imgsz=640)
            # get the mAP50 and mAP75
            map50_results.append(metrics.box.map50)
            map75_results.append(metrics.box.map75)

    # print the results
    print("mAP50: ", map50_results)
    print("mAP75: ", map75_results)
    return map50_results, map75_results


def modern_plot_map_scores(models, mAP_scores, type):
    # Set the theme for seaborn
    sns.set_theme(style="whitegrid")

    # Sort the scores in descending order for better visualization
    sorted_indices = sorted(range(len(mAP_scores)), key=lambda k: mAP_scores[k], reverse=True)
    models = [models[i] for i in sorted_indices]
    mAP_scores = [mAP_scores[i] for i in sorted_indices]

    # Initialize the matplotlib figure
    plt.figure(figsize=(16, 8))

    # Create a bar plot
    ax = sns.barplot(x=models, y=mAP_scores, palette="Blues_d")

    # Add labels and title
    plt.xlabel('Models', fontsize=15, labelpad=15)
    plt.ylabel('mAP Score', fontsize=15, labelpad=15)
    plt.title(type + '-mAP Scores - Adam Yolov8 from Scratch Models', fontsize=18, pad=20)
    plt.ylim([0, max(mAP_scores) + min(mAP_scores)])  # Assuming mAP score is between 0 and 1

    # Display mAP values on each bar
    for index, value in enumerate(mAP_scores):
        ax.text(index, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=12, fontweight='bold')

    # Remove the borders for a cleaner look
    sns.despine(left=True, bottom=True)

    # Show the plot
    plt.tight_layout()
    plt.show()
