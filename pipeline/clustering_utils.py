
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


