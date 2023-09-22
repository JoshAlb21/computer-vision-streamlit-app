import numpy as np
import matplotlib.pyplot as plt

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
