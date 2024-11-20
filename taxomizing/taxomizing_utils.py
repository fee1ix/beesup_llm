import pandas as pd
import numpy as np

from ..toolkit.setup_utils import *



def is_normalized_embs(embs):
    norm_embs = np.linalg.norm(embs, axis=1)
    return np.allclose(norm_embs, 1, atol=1e-6) # check if all embeddings are normalized


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
    nodes_df.loc[nodes_df['is_leaf'], 'subject'] = chunks_df['subject'].values
    nodes_df.loc[nodes_df['is_leaf'], 'predicate'] = chunks_df['predicate'].values
    nodes_df.loc[nodes_df['is_leaf'], 'object'] = chunks_df['object'].values
    
    nodes_df.loc[nodes_df['is_leaf'], 'emb'] = chunks_df['emb'].values
    nodes_df.loc[nodes_df['is_leaf'], 'cid'] = chunks_df['cid'].values



    return nodes_df


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

def propagate_emb(tree):

    def _recursive(node):

        if not node.children:  # Base case: if it's a leaf node, return its embedding
            return getattr(node,'emb') if hasattr(node, 'emb') else None

        # Collect embeddings from all descendants
        embs = []
        for child in node.children:
            child_emb = _recursive(child)
            if child_emb is not None:
                embs.append(child_emb)
        
        # Calculate the mean embedding if there are any valid embeddings
        mean_emb=None
        if embs:
            mean_emb = np.mean(embs, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)  # Normalize the mean embedding
            setattr(node,'emb',mean_emb)

        return mean_emb
    
    _recursive(tree)

    return tree


def propagate_mean(tree, key, return_df=False):

    if isinstance(tree,pd.DataFrame):
        tree=to_anytree(tree)
        return_df=True

    def _recursive(node, key):

        if not node.children:  # Base case: if it's a leaf node, return its embedding
            return getattr(node,key) if hasattr(node, key) else None

        # Collect embeddings from all descendants
        embs = []
        for child in node.children:
            child_embedding = _recursive(child, key)
            if child_embedding is not None:
                embs.append(child_embedding)
        
        # Calculate the mean embedding if there are any valid embeddings
        if embs:
            setattr(node,key,np.mean(embs, axis=0))
        else:
            setattr(node,key,None) # Or np.zeros(...), depending on your preference
    
        return getattr(node,key)
    
    _recursive(tree, key)
    
    if return_df:
        tree=to_nodes_df(tree)
    
    return tree

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


from anytree import Node, RenderTree, PreOrderIter

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

def to_nodes_df(tree, iterator=None):

    if iterator is None:
        nodes=[tree] + list(tree.descendants)
    else:
        nodes=iterator(tree)

    node_data=[]

    for node in nodes:

        node_row=dict(id=node.name)
        node_row.update(filter_attributes(node,include_types=None))
        del node_row['name']
        node_row.update(dict(
            is_leaf=node.is_leaf,
            is_root=node.is_root,
            size=node.size,
            #height=node.height,
            depth=node.depth,
        ))

        node_data.append(node_row)

    nodes_df=pd.DataFrame(node_data)

    if iterator is None:
        nodes_df=nodes_df.sort_values(by=['size','id'], ascending=[True,True])
    
    return nodes_df




def is_member(node):
    """Check if the node or any of its descendants has a 'cid' not equal to -1 or np.nan."""
    if node.cid not in [-1, np.nan]:
        return True
    return any(is_member(descendant) for descendant in node.children)

def get_member_tree(tree):

    new_nodes={}
    for node in [tree] + list(tree.descendants):
        if not is_member(node): continue
        new_nodes[node.id] = Node(node.id)


        # Set the parent if it exists in the filtered nodes
        if node.parent and node.parent.id in new_nodes:
            new_nodes[node.id].parent = new_nodes[node.parent.id]


        for k,v in node.__dict__.items():
            if k in ['is_leaf','is_root','size','depth','height']: continue
            if k.startswith('_'): continue
            new_nodes[node.id].__setattr__(k,v)
    
    for node in new_nodes.values():
        decendant_cids=set([int(d.cid) for d in node.descendants if d.cid>=0])
        new_nodes[node.id].__setattr__('decendant_cids',decendant_cids)
        new_nodes[node.id].__setattr__('label',f"Cluster with {node.size} members")


    # Step 4: Identify disconnected components and handle them as independent trees or under a new root
    for node in new_nodes.values():
        if node.parent is None:  # Identify root nodes of disconnected components
            return node
