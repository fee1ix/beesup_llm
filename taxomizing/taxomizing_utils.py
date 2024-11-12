import pandas as pd
import numpy as np

from ..toolkit.setup_utils import *

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

    chunks_df['cid']=clusterer.labels_

    # Convert the condensed tree to a DataFrame
    nodes_df = clusterer.condensed_tree_.to_pandas()
    nodes_df = nodes_df.rename(columns={
        'parent':'parent_id',
        'child':'id',
        'child_size':'size',
    })
    nodes_df['is_leaf']=nodes_df.apply(lambda x: x['size']==1, axis=1)

    nodes_df=nodes_df.sort_values(by=['size','id'], ascending=[True,True])

    nodes_df['label']=nodes_df.apply(lambda x: f"CLUSTER WITH {x['size']} MEMBERS", axis=1)

    nodes_df.loc[nodes_df['is_leaf'], 'label'] = chunks_df['label'].values
    nodes_df.loc[nodes_df['is_leaf'], 'emb'] = chunks_df['emb'].values
    nodes_df.loc[nodes_df['is_leaf'], 'cid'] = chunks_df['cid'].values

    return nodes_df

def add_root_row(nodes_df):

    parent_ids=set(nodes_df['parent_id'].values)
    node_ids=set(nodes_df['id'].values)
    root_ids=parent_ids-node_ids

    if len(root_ids)!=1: raise ValueError('more than one root found')

    root_row=dict(
        parent_id=None,
        id=root_ids.pop(),
        label='ROOT',
        is_leaf=False,
    )

    nodes_df=pd.concat([nodes_df,pd.DataFrame([root_row])], ignore_index=True)
    return nodes_df

def add_parent_cids(nodes_df):
    
    for _,row in nodes_df[nodes_df.cid.isna()].iterrows():

        selected_df=nodes_df[nodes_df.parent_id==row.id]

        if selected_df.empty: continue
        if selected_df.cid.nunique()!=1: continue

        cid=selected_df['cid'].iloc[0]
        if pd.isna(cid): continue

        nodes_df.loc[row.name,'cid']=cid
    
    return nodes_df

def add_parent_embs(nodes_df):
    while nodes_df['emb'].isna().any():  # Continue until no NaNs remain in 'emb'
        selector = nodes_df['emb'].notna()

        for parent_id in nodes_df[selector]['parent_id'].unique():
            # Calculate mean embedding for this parent node based on its children embeddings
            children_embeddings = nodes_df[nodes_df['parent_id'] == parent_id]['emb']

            if not children_embeddings.isna().any():  # Ensure all children have embeddings
                parent_emb = np.mean(children_embeddings.values, axis=0)

                if parent_id in nodes_df['id'].values:
                    parent_idx = nodes_df[nodes_df['id']==parent_id].index[0]
                
                    # Assign mean embedding for each row with child_id == parent_id
                    nodes_df.at[parent_idx, 'emb'] = parent_emb#] * sum(nodes_df['child_id'] == parent_id)
    
    return nodes_df


from anytree import Node

def to_anytree(nodes_df):
    # Step 1: Create a dictionary to hold nodes (each 'child_id' and 'parent_id' is a node)
    nodes = {}

    # Step 2: Create nodes for each unique 'child_id' only, starting from the bottom
    for _, row in nodes_df.iterrows():

        id = row['id']
        if id not in nodes:
            # Create the node with attributes and store it in the nodes dictionary
            nodes[id] = Node(int(id))
            for k,v in row.items():
                if k in ['is_leaf','is_root','size','depth','height']: continue
                nodes[id].__setattr__(k,v)

    # Step 3: Assign parents according to the 'parent_id'
    for _, row in nodes_df.iterrows():
        parent_id = row['parent_id']
        id = row['id']
        
        # If the parent exists, assign it as the parent of the child node
        if parent_id:
            if parent_id not in nodes:
                continue
                # Create the parent node if it does not exist
                #nodes[parent_id] = Node(parent_id)
            nodes[id].parent = nodes[parent_id]
    
    # Step 4: Identify disconnected components and handle them as independent trees or under a new root
    for node in nodes.values():
        if node.parent is None:  # Identify root nodes of disconnected components
            return node

def to_nodes_df(tree):

    node_data=[]

    for node in [tree] + list(tree.descendants):

        node_row=dict(id=node.name)
        node_row.update(filter_attributes(node,include_types=None))
        del node_row['name']
        node_row.update(dict(
            is_leaf=node.is_leaf,
            is_root=node.is_root,
            size=node.size,
            height=node.height,
            depth=node.depth,
        ))

        node_data.append(node_row)

    nodes_df=pd.DataFrame(node_data)
    nodes_df=nodes_df.sort_values(by=['size','id'], ascending=[True,True])
    
    return nodes_df



