import random

import colorsys
import numpy as np
import seaborn as sns

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def get_different_colors_hsv(n):
    hues = np.linspace(0, 1, n, endpoint=True)
    colors = [np.array([hue,0.9,0.9]) for hue in hues]
    return colors

def assign_colors_list_hsv(cids):
    cids = [cid if not np.isnan(cid) else -1.0 for cid in cids]
    unique_clusters = np.unique(cids)
    colors = get_different_colors_hsv(len(unique_clusters))
    random.shuffle(colors)
    cluster_colors = {cluster: color for cluster, color in zip(unique_clusters, colors)}

    default_colors = {
        np.nan:np.array([204,5,78]),
        -1: np.array([204,5,78])
        }
    cluster_colors.update(default_colors)

    return [cluster_colors[cluster] for cluster in cids]


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
    cluster_ids = [cid if not np.isnan(cid) else -1.0 for cid in cluster_ids]
    unique_clusters = np.unique(cluster_ids)
    colors = get_different_colors(len(unique_clusters))
    random.shuffle(colors)
    cluster_colors = {cluster: color for cluster, color in zip(unique_clusters, colors)}

    default_colors = {-1: 'rgb(114, 114, 114)'}
    cluster_colors.update(default_colors)

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
from IPython.display import display, clear_output, HTML

def print_prompt_messages(prompt_messages):
    for message in prompt_messages:
        print(message['role'])
        print(message['content'], end='\n\n')

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


def assign_pca_coords(embs_df, n_components=3):

    # prepare pca
    from sklearn.decomposition import PCA
    emb_space=np.vstack(embs_df['emb'].values)

    pca=PCA(n_components=n_components)
    pca.fit(emb_space)
    #logging.info(f'explained variance ratio: {pca.explained_variance_ratio_}')

    pca_coords=pca.transform(np.vstack(embs_df['emb']))

    dimensional_labels='xyzuvw'
    for i, letter in enumerate(dimensional_labels[:n_components]):
        embs_df[f'pca_{letter}']=pca_coords[:,i]
    
    return embs_df



import plotly.graph_objects as go
GO_LAYOUT=go.Layout(
        title="PCA reduced Embedding-Space",
        height=500,
        autosize=True,  # Enables automatic resizing
        scene=dict(
            bgcolor='black',       # Background color of the 3D scene
            xaxis=dict(backgroundcolor="grey", gridcolor="darkgrey"),
            yaxis=dict(backgroundcolor="grey", gridcolor="darkgrey"),
            zaxis=dict(backgroundcolor="grey", gridcolor="darkgrey")
        ),
        paper_bgcolor='black',   # Background color outside the 3D scene
        plot_bgcolor='black',     # Background color of the entire plotting area
        #showlegend=True,
    )


def plot_emb_clustering_scatter_3d(clustering_df, n_samples=None, random_state=22):
    
    if n_samples is None: n_samples=len(clustering_df)

    plot_df=clustering_df.copy()
    plot_df=plot_df.sample(n=n_samples, random_state=random_state)

    if not {'pca_x', 'pca_y', 'pca_z'}.issubset(plot_df.columns):
        plot_df=assign_pca_coords(plot_df, n_components=3)

    if not 'color' in plot_df.columns:
        plot_df['color']=assign_colors_list(plot_df['cid'].values)
    

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
        x=plot_df['pca_x'].values,
        y=plot_df['pca_y'].values,
        z=plot_df['pca_z'].values,
        mode='markers',  # 'markers' mode for points
        marker=dict(size=2, color=plot_df['color']),  # Customize marker appearance
        name='knowledge_points',  # Name for the legend
        showlegend=True,
        text=plot_df['label'].values if 'label' in plot_df.columns else None,
        hoverinfo='text'
        )
    )

    fig.update_layout(GO_LAYOUT)
    fig.show(renderer='vscode')