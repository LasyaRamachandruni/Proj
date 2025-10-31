import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the Excel files
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top1 fullmesh 4 trails.xlsx'    # full mesh
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top1 fullmesh 8 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top2 fullmesh 4 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top2 fullmesh 8 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top1 star 4 trails.xlsx'        # star
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top1 star 8 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top2 star 4 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top2 star 8 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top3 star 4 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top3 star 8 trails.xlsx'
# file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top3 star 16 trails 20 req.xlsx'
file_path = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/heuristics/top3 star 16 trails 40 req.xlsx'

# rl agent
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top1/full-mesh/top1-4trails.csv'   # full mesh
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top1/full-mesh/top1-8trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top2/full-mesh/top2-4trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top2/full-mesh/top2-8trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top1/star/top1-star-4trails.csv'   # star
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top1/star/top1-star-8trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top2/star/top2-star-4trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top2/star/top2-star-8trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top3/top3-4trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top3/top3-8trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top3/top3-16trails.csv'
# file_path2 = 'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/rl/top3/40 req/top3-16trails.csv'

# optimal scores line
# optimal_scores = [25, 25, 27, 25, 24, 30, 26, 28, 24, 24, 27, 30, 24, 26, 24, 27, 24, 25, 28, 27]                     # full mesh t1
# optimal_scores = [22, 22, 24, 23, 22, 27, 24, 25, 21, 24, 27, 29, 23, 24, 24, 24, 22, 24, 27, 25]
# optimal_scores = [225, 201, 218, 207, 205, 219, 196, 201, 225, 242, 218, 213, 222, 217, 215, 221, 200, 217, 208, 217] # t2
# optimal_scores = [204, 180, 194, 185, 176, 198, 171, 181, 197, 220, 185, 188, 192, 194, 180, 182, 174, 192, 192, 197]
# optimal_scores = [25, 25, 26, 25, 23, 28, 26, 28, 24, 24, 25, 28, 24, 26, 21, 26, 24, 25, 26, 23]                     # star t1
# optimal_scores = [20, 20, 21, 21, 21, 24, 22, 23, 19, 22, 24, 27, 21, 22, 21, 22, 21, 22, 25, 21]
# optimal_scores = [115, 98, 105, 112, 95, 101, 87, 90, 105, 136, 103, 107, 104, 119, 104, 119, 102, 101, 98, 100]      # t2
# optimal_scores = [90, 72, 85, 85, 61, 77, 64, 65, 77, 113, 73, 77, 69, 88, 67, 73, 72, 72, 76, 78]
# optimal_scores = [879, 821, 851, 815, 787, 845, 816, 905, 931, 977, 897, 906, 826, 766, 827, 845, 703, 849, 932, 902] # t3
# optimal_scores = [509, 555, 331, 398, 287, 428, 424, 401, 607, 603, 447, 416, 389, 320, 363, 397, 321, 334, 469, 553]
# optimal_scores =

df  = pd.read_excel(file_path)
# df2 = pd.read_csv(file_path2)

# print(df, df2)
# df3 = pd.concat([df, df2])
# print(df3)

print(df)


# Define colors for each box
colors = ['lightblue', 'lightgreen', 'salmon', 'orchid']

# Create the box plot
plt.figure(figsize=(8, 6))
ax = df.boxplot(patch_artist=True, showfliers=False, whis=0)  # This returns an Axes object

# Apply colors to each box
for i, patch in enumerate(ax.patches):
    patch.set_facecolor(colors[i % len(colors)])  # Cycle colors if needed

# create optimal score line
"""
opt_mean = np.mean(optimal_scores)
opt_med  = np.median(optimal_scores)
y_position = opt_med  # Adjust this based on where you want the line
plt.axhline(y=y_position, color='red', linestyle='--', linewidth=2, label='Optimal (median)')   # print line
# plt.text(x=4.55, y=y_position, s=f'{y_position}', color='red', fontsize=14, verticalalignment='center') # print opt value next to line    w/ rl
plt.text(x=3.55, y=y_position, s=f'{y_position}', color='red', fontsize=14, verticalalignment='center') # print opt value next to line     w/o rl
"""

# Increase font size of x-axis labels (box names)
plt.xticks(fontsize=14)  # Increase font size of category labels

# Customize the plot
#plt.title('Colored Box Plot for Multiple Columns')
#plt.xlabel('Columns')
plt.ylabel('Localization Narrowing Index', fontsize=14)

# Increase font size of the legend
plt.legend(fontsize=12)

# print data into console
"""
print(f"Optimal (Median): {opt_med}")
medians = df.median()
for column, median_value in medians.items():
    print(f"Column: {column}, Median: {median_value}, Distance: {median_value / opt_med}")
"""

# Show the plot
plt.show()
