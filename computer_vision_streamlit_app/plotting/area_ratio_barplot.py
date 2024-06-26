import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
import numpy as np

# Deprecated
def plot_single_segmented_ratio_barplot(ratios: dict, labels: list[str], return_image:bool=False):
    '''
    Plot a single bar plot where the bar is segmented into different parts.
    Each segment in the bar represents the ratio of a object_part, with the ratio
    written within each corresponding part.
    
    Parameters:
        ratios: dict
            A dictionary of the ratios of the areas of the segments.
            The keys are formatted as "{segment}_total".
            The values are formatted as float.
        labels: list
            A list of the segment names (e.g., ['dog', 'umbrella', 'bicycle']).
    '''

    # Initialize data for each segment
    segment_ratios = []

    # Collecting ratios for each object_part
    for label in labels:
        total_key = f'{label}_total'
        if total_key in ratios:
            segment_ratios.append(ratios[total_key])

    # Plotting
    plt.figure(figsize=(6, 6))
    bar_width = 0.35

    # Colors for each segment
    color_mapping = {'dog': 'green', 'umbrella': 'red', 'bicycle': 'blue'}

    # Stacking each segment in the bar
    bottom_height = 0
    for i, ratio in enumerate(segment_ratios):
        plt.bar(0, ratio, bar_width, bottom=bottom_height, color=color_mapping[labels[i]], label=labels[i])
        # Write the ratio inside each segment
        plt.text(0, bottom_height + ratio / 2, f'{ratio:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        bottom_height += ratio

    # Adding labels
    plt.ylabel('Ratio Value')
    plt.title('Segmented Area Ratios of Body Parts')
    plt.xticks([], [])  # No x-ticks needed as there's only one bar
    plt.legend(loc='upper right')
    plt.tight_layout()

    if return_image:
        return plt.gcf()
    else:
        plt.show()


def plot_grouped_ratio_barplot_with_labels(ratios:dict, labels:list[str]):
    '''
    Plot a grouped bar plot with the ratios of the areas of the segments.
    
    Parameters:
        ratios: dict
            A dictionary of the ratios of the areas of the segments.
            The keys are the labels of the segments.
            The values are the ratios of the areas of the segments.
            The keys are formatted as "{label1}_{label2}".
            The values are formatted as float.
        labels: list[str]
            A list of the labels of the segments.
    '''

    # Extract data for bar plot
    group_data = {}
    total_values = []
    
    for label1 in labels:
        group_data[label1] = []
        for label2 in labels:
            if label1 != label2:
                group_data[label1].append(ratios[f"{label1}_{label2}"])
        
        # Add total ratio at the end
        total_values.append(ratios[f"{label1}_total"])
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.25
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    bars1 = plt.bar(r1, [group_data[label][0] for label in labels], color='gray', width=bar_width, edgecolor='gray', label=labels[1])
    bars2 = plt.bar(r2, [group_data[label][1] for label in labels], color='gray', width=bar_width, edgecolor='gray', label=labels[2])
    bars3 = plt.bar(r3, total_values, color='blue', width=bar_width, edgecolor='gray', label='Total')
    
    # Add text on top of bars
    for i, bars in enumerate([bars1, bars2, bars3]):
        for j, bar in enumerate(bars):
            if i != 2:  # For the first two groups, add label names
                label = labels[(j + i + 1) % 3]
            else:  # For the third group, add "Total"
                label = 'Total'
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, label, ha='center', va='bottom', fontsize=10)
    
    # Add legend, title, and adjust layout
    plt.xlabel('Segments', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(labels))], labels)
    plt.ylabel('Ratio Value')
    plt.title('Segment Area Ratios')
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.show()

def plot_area_comparison_matrix(df_segments:pd.DataFrame):
    '''
    Plot a matrix of histograms comparing the distributions of the areas of the segments.

    TODO: not generalizable to more than 3 segments wiht differnt names
    '''
    # Set up the figure and axes again
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

    # List of segments and corresponding total columns
    segments = ['umbrella', 'bicycle', 'dog']
    total_cols = ['abdomen_total', 'thorax_total', 'head_total']

    # List of relationships between segments for off-diagonal plots
    relationships = [
        ['abdomen_thorax', 'abdomen_head'],
        ['thorax_abdomen', 'thorax_head'],
        ['head_abdomen', 'head_thorax']
    ]

    # Plot the distributions
    for i, segment in enumerate(segments):
        for j, total_col in enumerate(total_cols):
            if i == j:
                # Plot histogram on the diagonal
                sns.histplot(df_segments[total_col], ax=axes[i][j], bins=30, kde=True, color='skyblue')
                axes[i][j].set_title(f"{segment.capitalize()} Total")
            else:
                # Plot off-diagonal relationships
                sns.histplot(df_segments[relationships[i][j-1]], ax=axes[i][j], bins=30, kde=True, color='lightcoral')
                axes[i][j].set_title(relationships[i][j-1].capitalize().replace('_', ' '))

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_area_comparison_violin(df_segments:pd.DataFrame, col:str=None, category:str=None):

    if col is not None and category is not None:
        if col not in df_segments.columns:
            raise ValueError(f"Column '{col}' not in DataFrame.")
        
        df_segments = df_segments[df_segments[col] == category]

    area_cols = ['umbrella', 'bicycle', 'dog']

    # Plotting the violin plots for the area columns
    fig, axes = plt.subplots(nrows=1, ncols=len(area_cols), figsize=(18, 6))

    for ax, col in zip(axes, area_cols):
        sns.violinplot(y=df_segments[col], ax=ax, color='lightgreen')
        ax.set_title(col.capitalize() + ' Area')
        ax.set_ylabel('Value')

    # Adjust layout
    plt.title(f"Area Comparison Violin Plot for {category}")
    plt.tight_layout()
    plt.show()

def object_parts_scatter2D(df, object_parts, color_col):

    # Number of rows and columns for subplots
    nrows = 1
    ncols = 3 

    # Create figure and axes
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5))

    axs[0].scatter(df[object_parts[0]], df[object_parts[1]], c=df[color_col].astype('category').cat.codes)
    axs[0].set_xlabel(object_parts[0])  
    axs[0].set_ylabel(object_parts[1])
    axs[0].set_title(f"{object_parts[0]} vs {object_parts[1]}") 

    axs[1].scatter(df[object_parts[1]], df[object_parts[2]], c=df[color_col].astype('category').cat.codes)
    axs[1].set_xlabel(object_parts[1])  
    axs[1].set_ylabel(object_parts[2])
    axs[1].set_title(f"{object_parts[1]} vs {object_parts[2]}") 

    axs[2].scatter(df[object_parts[0]], df[object_parts[2]], c=df[color_col].astype('category').cat.codes)
    axs[2].set_xlabel(object_parts[0])
    axs[2].set_ylabel(object_parts[2])
    axs[2].set_title(f"{object_parts[0]} vs {object_parts[2]}") 

    # Adjust layout  
    fig.tight_layout()

    plt.show()