import seaborn as sns
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer

from beesup_llm.injection.taxomizing_utils import *

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

def add_internal_labels(tree: Node) -> None:

    for node in PreOrderIter(tree):
        if not node.is_leaf:
            children_is_leaf=[n.is_leaf for n in node.children]
            num_leafs=children_is_leaf.count(True)
            num_node_children=children_is_leaf.count(False)
            node.__setattr__('label',f"internal node ({len(node.children)}/ {len(node.leaves)})")
    return

def assign_colors_tree(tree: Node) -> None:

    num_internal_nodes=len([node for node in PreOrderIter(tree) if not node.is_leaf])
    colors=sns.color_palette("husl", num_internal_nodes)

    for node in InOrderIter(tree):
        if node.is_leaf: continue

        color=colors.pop()
        node.__setattr__('color',color)

        if all([n.is_leaf for n in node.children]):
            for n in node.children:
                n.__setattr__('color',color)
    return   

def get_plot_df_for_taxomizer_tree(tree: Node) -> pd.DataFrame:

    plot_data=[]

    for node in PreOrderIter(tree):

        plot_row=dict(
            id=node.name,
            parent_id=node.parent.name if node.parent is not None else None,
            dist=getattr(node,'dist',0),
            label=node.label,
            emb=node.emb,
            parent_emb=node.parent.emb if node.parent is not None else node.emb,
            parent_dist=getattr(node.parent,'dist',node.dist),
            husl_color=node.color,
        )

        plot_data.append(plot_row)

    plot_df=pd.DataFrame(plot_data)



    emb_matrix=np.vstack(plot_df['emb'].values)
    pca=PCA(n_components=2)
    pca.fit(emb_matrix)

    pca_emb_matrix=pca.transform(emb_matrix)

    #dist_scaler = MinMaxScaler(feature_range=(np.min(pca_emb_matrix), np.max(pca_emb_matrix)))
    dist_scaler = StandardScaler()


    scaler_dist_matrix = dist_scaler.fit_transform(np.vstack(plot_df['dist'].values))
    plot_df['pca_coords']=list(np.concatenate([pca_emb_matrix, scaler_dist_matrix], axis=1))


    parent_pca_emb_matrix=pca.transform(np.vstack(plot_df['parent_emb'].values))
    parent_scaler_dist_matrix=dist_scaler.transform(np.vstack(plot_df['parent_dist'].values))
    plot_df['parent_pca_coords']=list(np.concatenate([parent_pca_emb_matrix, parent_scaler_dist_matrix], axis=1))

    plot_df.drop(columns=['emb','parent_emb','dist','parent_dist'],inplace=True)
    return plot_df

def get_plot_for_taxomizer_tree(tree: Node) -> go.Figure:

    add_internal_labels(tree)
    assign_colors_tree(tree)

    plot_df=get_plot_df_for_taxomizer_tree(tree)

    # PLOT the tree
    fig = go.Figure()

    pca_coords=np.vstack(plot_df['pca_coords'].values)
    parent_pca_coords=np.vstack(plot_df['parent_pca_coords'].values)
    parent_traces = np.stack([pca_coords, parent_pca_coords], axis=1)  # Shape: (2445, 2, 3)

    # Nodes
    node_points=go.Scatter3d(
            x=pca_coords[:,0],
            y=pca_coords[:,1],
            z=pca_coords[:,2],
            mode='markers',  # 'markers' mode for points
            marker=dict(size=2, color=plot_df['husl_color']),  # Customize marker appearance
            name='nodes',  # Name for the legend
            #showlegend=True,
            #text=plot_df['label'].values,
            text=plot_df.apply(lambda x: f"{x.id} {x.label}", axis=1).values,
            hoverinfo='text'
            )
    fig.add_trace(node_points)

    for i in range(parent_traces.shape[0]):
        parent_trace=go.Scatter3d(
                x=parent_traces[i][:,0],
                y=parent_traces[i][:,1],
                z=parent_traces[i][:,2],
                mode='lines',  # 'markers' mode for points
                line=dict(width=1, color=2*[plot_df['husl_color'][i]]),  # Customize marker appearance
                name='parent_trace',  # Name for the legend
                text=None,
                hoverinfo='none',
                #legendgroup=
                #legendgroup='parent_traces',
                showlegend=False,
                #text=nodes_df['label'].values,
                #hoverinfo='text'
                )

        fig.add_trace(parent_trace)

    fig.update_layout(GO_LAYOUT)
    fig.show(renderer='vscode')

    return fig
