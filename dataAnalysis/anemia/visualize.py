import matplotlib.pyplot as plt
from xgboost import plot_tree
import joblib

name = "../modelXGBoost.pkl"
outputName = "exmapleFilePCAtree.png"


def visualizeTree(modelFile, outputFile):

    loaded_model = joblib.load(modelFile)
    trees = loaded_model.get_booster().get_dump()

    n_trees = len(trees)
    print(f"Liczba drzew w modelu: {n_trees}")

    plt.figure(figsize=(40, 20))
    plot_tree(loaded_model, num_trees=0)
    plt.savefig(outputFile, dpi=1200)
    plt.show()


visualizeTree(name, outputName)
