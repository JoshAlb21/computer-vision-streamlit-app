import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_pairwise_vol_len(df: pd.DataFrame, cols_vol:list, cols_len:list, hue:str):

    # Columns to consider for pairwise scatter plots
    cols = ["volume_head", "volume_thorax", "volume_abdomen"]
    cols = ["length_head", "length_thorax", "length_abdomen"]
    plt.rcParams.update({'font.size': 16})

    # Volume
    sns.pairplot(df, hue=hue, vars=cols_vol, plot_kws={'alpha':0.7})
    vol_plot = plt.gcf()

    # Length
    sns.pairplot(df, hue=hue, vars=cols_len, plot_kws={'alpha':0.7})
    len_plot = plt.gcf()

    return vol_plot, len_plot
