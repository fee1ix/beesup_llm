import random
import colorsys
import numpy as np

def get_different_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5  # you can play with lightness 
        saturation = 0.9  # saturation set to 0.9 to ensure colors are fairly vivid
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(int(c * 255) for c in rgb))

    return ['rgb'+str(c) for c in colors]

def assign_colors_list(cluster_ids):
    unique_clusters = np.unique(cluster_ids)
    colors = get_different_colors(len(unique_clusters))
    random.shuffle(colors)
    cluster_colors = {cluster: color for cluster, color in zip(unique_clusters, colors)}
    return [cluster_colors[cluster] for cluster in cluster_ids]

from matplotlib.colors import hsv_to_rgb
def assign_colors_tree(node, hue_start=0, hue_end=1, depth=0):
    """
    Assign colors recursively to each node in the tree.
    
    Parameters:
    - node: The current node (from to_tree)
    - hue_start, hue_end: The range of hues to assign for this node and its children
    - depth: The current depth in the tree (for optional use)
    
    Returns:
    - The color of the current node
    """
    # Calculate the hue for the current node
    hue = (hue_start + hue_end) / 2
    saturation = 1.0
    value = 1.0
    
    # Convert HSV to RGB color
    node.color = hsv_to_rgb([hue, saturation, value])
    
    # If the node is not a leaf, recursively assign colors to its children
    if not node.is_leaf():
        # Split the hue range for the left and right children
        mid_hue = (hue_start + hue_end) / 2
        assign_colors_tree(node.left, hue_start, mid_hue, depth + 1)
        assign_colors_tree(node.right, mid_hue, hue_end, depth + 1)

import math
from IPython.display import display, HTML
def print_multicol(cols):
    col_width=math.floor(100/len(cols))
    html_content=""
    for col in cols:
        html_content+=f"""
<div style="float: left; width: {col_width}%;">
    <pre style="white-space: pre-wrap;">{col}</pre>
</div>
"""
    display(HTML(html_content))


