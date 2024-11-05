
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

def assign_colors(cluster_ids):
    unique_clusters = np.unique(cluster_ids)
    colors = get_different_colors(len(unique_clusters))
    random.shuffle(colors)
    cluster_colors = {cluster: color for cluster, color in zip(unique_clusters, colors)}
    return [cluster_colors[cluster] for cluster in cluster_ids]