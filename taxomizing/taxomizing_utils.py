
import numpy as np

def assign_data_tree(node, df):

    # Base case: if the node is a leaf
    if node.is_leaf():
        # Get the corresponding embedding from knowledge_df by the leaf node ID
        node.emb = df.iloc[node.id]['emb']
        node.label = df.iloc[node.id]['label']
        node.chunk = df.iloc[node.id]['chunk']
        return node.emb
    
    else:
        # Recursively assign embeddings to the left and right children
        left_emb = assign_data_tree(node.left, df)
        right_emb = assign_data_tree(node.right, df)
        
        # Compute the average embedding for the parent node
        node.emb = np.mean([left_emb, right_emb], axis=0)
        node.label= f"ID: {node.id}<br>DIST: {node.dist:.2f}<br>N: {node.count}"
        node.chunk = None
        return node.emb
    
def get_node_data(tree):

    def listify_tree(node):
        """
        Recursively collect attributes from the tree into a list.
        
        Each node's attributes include: node_id, left_child_id, right_child_id, color, and embedding.
        """
        node_row = {
            'node_id': node.id,
            'dist': node.dist,
            'n_children': node.count,
            'left_child_id': node.left.id if node.left else None,
            'right_child_id': node.right.id if node.right else None,
            'is_leaf': node.is_leaf(),
            'color': node.color,
            'emb': node.emb,
            'label': node.label,
            'chunk': node.chunk,
        }
        node_data.append(node_row)
        
        if not node.is_leaf():
            listify_tree(node.left)
            listify_tree(node.right)

    node_data=[]
    listify_tree(tree)

    return node_data


def nodes_df_from_hdbscan(clusterer,chunks_df):

    chunks_df['cluster_id']=clusterer.labels_

    # Convert the condensed tree to a DataFrame
    nodes_df = clusterer.condensed_tree_.to_pandas()
    nodes_df = nodes_df.rename(columns={
        'parent':'parent_id',
        'child':'child_id',
        'child_size':'n_children',
    })
    nodes_df['is_leaf']=nodes_df.apply(lambda x: x['n_children']==1, axis=1)

    nodes_df=nodes_df.sort_values(by=['n_children','child_id'], ascending=[True,True])

    nodes_df['label']=nodes_df.apply(lambda x: f"CLUSTER WITH {x.n_children} MEMBERS", axis=1)

    nodes_df.loc[nodes_df['is_leaf'], 'label'] = chunks_df['label'].values
    nodes_df.loc[nodes_df['is_leaf'], 'emb'] = chunks_df['emb'].values

    return nodes_df

from anytree import Node

def tree_from_nodes_df(nodes_df):
    # Step 1: Create a dictionary to hold nodes (each 'child_id' and 'parent_id' is a node)
    nodes = {}

    # Step 2: Create nodes for each unique 'child_id' only, starting from the bottom
    for _, row in nodes_df.iterrows():
        child_id = row['child_id']
        if child_id not in nodes:
            # Create the node with attributes and store it in the nodes dictionary
            nodes[child_id] = Node(
                int(child_id),
                parent_id=row['parent_id'],
                lambda_val=row['lambda_val'], 
                n_children=row['n_children'],
                label=row['label'],
                emb=row['emb'],
                )

    # Step 3: Assign parents according to the 'parent_id'
    for _, row in nodes_df.iterrows():
        parent_id = row['parent_id']
        child_id = row['child_id']
        
        # If the parent exists, assign it as the parent of the child node
        if parent_id:
            if parent_id not in nodes:
                # Create the parent node if it does not exist
                nodes[parent_id] = Node(parent_id)
            nodes[child_id].parent = nodes[parent_id]

    # Step 4: Identify disconnected components and handle them as independent trees or under a new root
    root = Node("ROOT")  # Create an artificial root if needed to unify the trees

    for node in nodes.values():
        if node.parent is None:  # Identify root nodes of disconnected components
            node.parent = root  # Attach them to the artificial root, or leave as is if separate trees are desired
    
    return root