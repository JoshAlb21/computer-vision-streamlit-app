import pandas as pd

import computer_vision_streamlit_app as ta

csv_path = "/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/extract_csv/df_segments.csv"
dataset_csv_path = "/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/dataset_csv/merged.csv"
csv_enhanced_path = "/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/extract_csv/df_segments_dataset.csv"


#*******************
# Merge csv files
#*******************
# Load csv
df_segments = pd.read_csv(csv_path)
df_dataset = pd.read_csv(dataset_csv_path)
# Add data from df_dataset to df_segments
# format of "img_id": xx_"SpecimenCode/Filename"_xx_xx_xx
# Add new column "SpecimenCode/Filename" to df_segments by iterating over "img_id" and extracting the "SpecimenCode/Filename"
df_segments["SpecimenCode/Filename"] = df_segments["img_id"].apply(lambda x: x.split("_")[1])
# concat df_dataset and df_segments based on "SpecimenCode/Filename"
df_merged = pd.merge(df_segments, df_dataset, on="SpecimenCode/Filename")
# Save merged df
#df_merged.to_csv(csv_enhanced_path, index=False)

#*******************
# Prepare data
#*******************

# cut of rows with no data in "MalaiseDate" column
df_merged = df_merged[df_merged["MalaiseDate"].notnull()]

# convert Species column to "other" if occurence not in top k
k = 5
top_k = df_merged["Species"].value_counts()[:k].index.tolist()
df_merged.loc[~df_merged["Species"].isin(top_k), "Species"] = "other"

#*******************
# Plotting
#*******************
ta.plotting.area_ratio_barplot.plot_area_comparison_matrix(df_segments)

ta.plotting.area_ratio_barplot.plot_area_comparison_violin(df_merged)

#Create violin plot for each species
'''
species = df_merged["Species"].unique()
col = "Species"
for specie in species:
    print("Speci: ", specie)
    ta.plotting.area_ratio_barplot.plot_area_comparison_violin(df_merged, col, specie)
'''

object_parts = ["dog", "bicycle", "umbrella"]
ta.plotting.area_ratio_barplot.object_parts_scatter2D(df_merged, object_parts, "Species")