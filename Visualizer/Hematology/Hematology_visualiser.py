import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.ticker as mticker
import os


class AnemiaVisualizer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.label_columns = [col for col in self.df.columns if col.startswith("Label_")]
        self.numeric_cols = [col for col in self.df.columns if not col.startswith("Label_")]
        self.df["Label"] = self.df[self.label_columns].idxmax(axis=1).str.replace("Label_", "")
        self.df = self.df[self.df["Label"] != "Healthy"]

    def plot_density_heat_real_counts(self, column_name, save_path=None):
        values = self.df[column_name].dropna().values
        kde = gaussian_kde(values)
        x_min, x_max = values.min(), values.max()
        x = np.linspace(x_min, x_max, 1000)
        y = kde(x)

        total_count = len(values)
        scale_factor = total_count / np.trapz(y, x)
        y_scaled = y * scale_factor

        fig, ax = plt.subplots(figsize=(10, 2.5))
        X, Y = np.meshgrid(x, [0, 1])
        Z = np.tile(y_scaled, (2, 1))

        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='Blues')
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(column_name)
        ax.set_title(f"Gęstość rzeczywista: {column_name}")

        max_density = y_scaled.max()
        tick_step = 25000
        tick_locs = np.arange(0, max_density + tick_step, tick_step)

        cbar = plt.colorbar(im, orientation="horizontal", pad=0.5, ticks=tick_locs)
        cbar.set_label("Liczba przypadków (oszacowana)", fontsize=11)
        cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_all_heatmaps(self, save_dir=None):
        os.makedirs(save_dir, exist_ok=True) if save_dir else None
        for col in self.numeric_cols:
            path = os.path.join(save_dir, f"{col}_heatmap.png") if save_dir else None
            self.plot_density_heat_real_counts(col, save_path=path)


if __name__ == "__main__":
    path = "../../trainingData/anemia/synthetic_data_vae3.csv"
    viz = AnemiaVisualizer(path)
    viz.plot_all_heatmaps()
