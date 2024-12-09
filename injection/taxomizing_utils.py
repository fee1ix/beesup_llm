import pandas as pd
import numpy as np
from ..toolkit.setup_utils import *


# ANYTREE UTILITY FUNCTIONS
from anytree import Node, RenderTree, PreOrderIter, PostOrderIter

from anytree import Node, NodeMixin

class InOrderIter:
    """
    InOrder iterator for anytree.
    Traverses the tree in InOrder fashion:
      - Visits the left/middle children first.
      - Visits the current node.
      - Visits the remaining children.
    """
    def __init__(self, node):
        self.node = node

    def __iter__(self):
        return self._inorder(self.node)

    def _inorder(self, node):
        if not node:
            return
        # Split children into two halves
        mid = len(node.children) // 2
        
        # Traverse the first half of the children
        for child in node.children[:mid]:
            yield from self._inorder(child)
        
        # Visit the current node
        yield node
        
        # Traverse the second half of the children
        for child in node.children[mid:]:
            yield from self._inorder(child)

ANYTREE_ATTRIBUTES=['is_leaf','is_root','size','depth','height']

def linkage_to_btree(linkage_matrix, chunks_df):
    nodes={}

    # create leaf nodes
    for i, chunk_row in chunks_df.iterrows():
        node=Node(i)

        for k,v in chunk_row.items():
            if k in ANYTREE_ATTRIBUTES: continue
            node.__setattr__(k,v)
        
        node.__setattr__('dist',0.0)
        node.__setattr__('is_chunk',True)
        nodes[i]=node

    #create internal nodes
    n_leaves=len(chunks_df)
    for i, (left_child_id,right_child_id,dist,size) in enumerate(linkage_matrix):
        node_id=n_leaves+i
        node=Node(node_id)
        node.__setattr__('dist',dist)
        node.__setattr__('is_chunk',False)

        nodes[node_id]=node
        nodes[int(left_child_id)].parent=nodes[node_id]
        nodes[int(right_child_id)].parent=nodes[node_id]

    root=nodes[len(linkage_matrix) + n_leaves - 1]


    logging.info(get_tree_info_dict(root))

    return root

def add_ddist(tree):

    for node in PreOrderIter(tree):
        
        if not node.parent: continue
        if not hasattr(node,'dist'): continue
        if not hasattr(node.parent,'dist'): continue

        node.__setattr__('ddist',node.parent.dist-node.dist)
    
    return 

def propagate_emb_from_leaves(tree):

    for node in PostOrderIter(tree):
        if node.is_leaf: continue
        setattr(node,'emb',np.mean([child.emb for child in node.children],axis=0))

    return 

# TREE METRICS
def get_branching_factor(tree): #average number of children per internal node
    internal_nodes = [n for n in tree.descendants if n.children]
    if not internal_nodes:
        return 0  # Avoid division by zero if there are no internal nodes
    total_children = sum(len(n.children) for n in internal_nodes)
    return total_children / len(internal_nodes)

def get_maximum_degree(tree): #The maximum number of children among all nodes
    return max(len(n.children) for n in tree.descendants) if tree.descendants else len(tree.children)

def get_average_degree(tree): #The average number of children for all nodes
    all_nodes = list(tree.descendants) + [tree]
    total_children = sum(len(n.children) for n in all_nodes)
    return total_children / len(all_nodes)

def get_balance(tree): #balance factor of a node is the difference in the height of its left and right subtrees
    if not tree.children:
        return 0  # A leaf node has a balance factor of 0
    child_heights = [child.height for child in tree.children]
    return max(child_heights) - min(child_heights) if len(child_heights) > 1 else max(child_heights)

def get_average_balance(tree): #is a measure of how evenly distributed the nodes are across levels.
    all_nodes = list(tree.descendants) + [tree]
    total_balance_factor = sum(get_balance(n) for n in all_nodes)
    return total_balance_factor / len(all_nodes)

def get_path_length(tree, depth=0): #sum of depths for all nodes in the tree.
    return depth + sum(get_path_length(child, depth + 1) for child in tree.children)

def get_tree_diameter(tree): #longest path between any two nodes in the tree.

    def _recursive(tree):
        if not tree.children:
            return 0, 0  # height, diameter
        heights = []
        diameters = []
        for child in tree.children:
            h, d = _recursive(child)
            heights.append(h)
            diameters.append(d)
        if len(heights) > 1:
            max_heights = sorted(heights)[-2:]
            max_diameter = sum(max_heights) + 2
        else:
            max_diameter = max(heights) + 1 if heights else 0
        return max(heights) + 1, max(max_diameter, max(diameters))
    
    return _recursive(tree)[1]

def get_sum_depths(tree, depth=0):
    return depth + sum(get_sum_depths(child, depth + 1) for child in tree.children)

def get_average_depth(tree): #Average distance from the root to all nodes, indicating how deep the nodes are distributed in the tree
    all_nodes = list(tree.descendants) + [tree]
    total_depth = get_sum_depths(tree)
    return total_depth / len(all_nodes)  


from collections import defaultdict

def tree_width(tree): #maximum number of nodes at any level
    # Dictionary to store the number of nodes at each level
    level_counts = defaultdict(int)
    
    # Helper function to traverse the tree and populate level counts
    def traverse(node, level=0):
        level_counts[level] += 1
        for child in node.children:
            traverse(child, level + 1)
    
    traverse(tree)
    #return level_counts
    return max(level_counts.values())

def get_tree_density(tree): #A measure of how "full" the tree is compared to a perfectly balanced tree of the same height

    branching_factor = get_branching_factor(tree)
    total_nodes = tree.size
    height = tree.height
    
    # Calculate maximum possible nodes for the given branching factor and height
    if branching_factor > 1:
        max_possible_nodes = (branching_factor ** (height + 1) - 1) // (branching_factor - 1)
    else:
        max_possible_nodes = height + 1  # Linear tree
    
    # Compute density
    return total_nodes / max_possible_nodes

def get_tree_info_dict(tree, round_digits=2):
    nodes=list(PreOrderIter(tree))

    tree_info_dict=dict(
        num_nodes=tree.size,
        height=tree.height,
        num_root_children=len(tree.children),
        num_leaves=[n.is_leaf for n in nodes].count(True),
        branching=get_branching_factor(tree),
        avg_degree=get_average_degree(tree),
        avg_balance=get_average_balance(tree),
        dia=get_tree_diameter(tree),
        avg_depth=get_average_depth(tree),
        width=tree_width(tree),
        #density=get_tree_density(tree),
    )

    if round_digits is not None:
        tree_info_dict={k: round(v, round_digits) for k,v in tree_info_dict.items()}

    return tree_info_dict

def log_tree_info(tree, prefix=None):
    tree_info_dict=get_tree_info_dict(tree)

    outputs='\n'
    for k,v in tree_info_dict.items():

        if isinstance(v, int):
            output=f'{k}: {v}'
        if isinstance(v, float):
            output=f'{k}: {v:.2f}'
        
        outputs+=output+' | '
    
    if prefix:
        outputs=prefix+outputs

    logging.info(outputs)
 



# FLATTENING
from kneed import KneeLocator
import matplotlib.pyplot as plt

# dist flattening --> vertical!!
def get_dist_kneepoint(tree, include_leaves=True, plot=False):

    if include_leaves:
        sorted_dists=sorted([n.dist for n in PreOrderIter(tree)])
    else:
        sorted_dists=sorted([n.dist for n in PreOrderIter(tree) if not n.is_leaf])


    knee_index = KneeLocator(list(range(len(sorted_dists))), sorted_dists, curve="convex", direction="increasing").knee
    knee_dist = sorted_dists[knee_index]

    logging.info(f'knee_dist: {knee_dist:.4f}, knee_index: {knee_index}/ {len(sorted_dists)}')

    if plot:
        plt.figure(figsize=(18, 5))
        plt.plot(sorted_dists)

        plt.axhline(y=knee_dist, color='red', linestyle='-', linewidth=0.5)
        plt.axvline(x=knee_index, color='red', linestyle='-', linewidth=0.5)
        plt.ylabel('dist')

        plt.grid(True)
        plt.show()

    return knee_dist, knee_index

def get_dist_std(tree, std_factor=1.0, include_leaves=True, plot=False):

    if include_leaves:
        sorted_dists=np.array(sorted([n.dist for n in PreOrderIter(tree)]))
    else:
        sorted_dists=np.array(sorted([n.dist for n in PreOrderIter(tree) if not n.is_leaf]))

    threshold_dist=std_factor*sorted_dists.std()
    threshold_index=np.searchsorted(sorted_dists, threshold_dist, side="right")

    logging.info(f'threshold_dist: {threshold_dist:.4f}, threshold_index: {threshold_index}/ {len(sorted_dists)}')

    if plot:
        plt.figure(figsize=(18, 5))
        plt.plot(sorted_dists)

        plt.axhline(y=threshold_dist, color='red', linestyle='-', linewidth=0.5)
        plt.axvline(x=threshold_index, color='red', linestyle='-', linewidth=0.5)
        plt.ylabel('dist')

        plt.grid(True)
        plt.show()
    
    return threshold_dist, threshold_index

def get_dist_sorted_nodes(tree):
    #return sorted(PreOrderIter(btree), key=lambda node: getattr(node, 'dist', np.inf))
    return sorted((n for n in PreOrderIter(tree) if not n.is_leaf), key=lambda node: getattr(node, 'dist', np.inf))

def do_dist_flattening(tree, threshold_dist=None):

    tree=copy.deepcopy(tree)
    dist_sorted_nodes= get_dist_sorted_nodes(tree)
    deleted_nodes=[]

    while True:
        node=dist_sorted_nodes.pop(0)

        if node.dist >= threshold_dist: break

        for child in node.children:
            child.parent=node.parent

        node.parent=None
        deleted_nodes.append(node.name)
    
    logging.info(get_tree_info_dict(tree))
    return tree


# ddist flattening --> horizontal!!
def get_ddist_kneepoint(tree, include_leaves=True, plot=False):

    if include_leaves:
        sorted_ddists=sorted([n.ddist for n in tree.descendants])
    else:
        sorted_ddists=sorted([n.ddist for n in tree.descendants if not n.is_leaf])

    knee_index = KneeLocator(list(range(len(sorted_ddists))), sorted_ddists, curve="convex", direction="increasing").knee
    knee_ddist = sorted_ddists[knee_index]

    logging.info(f'knee_ddist: {knee_ddist:.4f}, knee_index: {knee_index}/ {len(sorted_ddists)}')

    if plot:
        plt.figure(figsize=(18, 5))
        plt.plot(sorted_ddists)
        #plt.scatter(knee_index, knee_dist, color="red", s=100, label="Kneepoint", marker='x')

        plt.axhline(y=knee_ddist, color='red', linestyle='-', linewidth=0.5)
        plt.axvline(x=knee_index, color='red', linestyle='-', linewidth=0.5)
        plt.ylabel('ddist')

        #plt.ylim(0, 0.01)  # Adjust the y-axis range
        plt.grid(True)
        plt.show()

    return knee_ddist, knee_index

def get_ddist_std(tree, std_factor=1.0, include_leaves=True, plot=False):

    if include_leaves:
        sorted_ddists=np.array(sorted([n.ddist for n in tree.descendants]))
    else:
        sorted_ddists=np.array(sorted([n.ddist for n in tree.descendants if not n.is_leaf]))

    threshold_ddist=std_factor*sorted_ddists.std()
    threshold_index=np.searchsorted(sorted_ddists, threshold_ddist, side="right")

    logging.info(f'threshold_ddist: {threshold_ddist:.4f}, threshold_index: {threshold_index}/ {len(sorted_ddists)}')

    if plot:
        plt.figure(figsize=(18, 5))
        plt.plot(sorted_ddists)

        plt.axhline(y=threshold_ddist, color='red', linestyle='-', linewidth=0.5)
        plt.axvline(x=threshold_index, color='red', linestyle='-', linewidth=0.5)
        plt.ylabel('ddist')

        plt.grid(True)
        plt.show()
    
    return threshold_ddist, threshold_index

def get_ddist_sorted_nodes(tree):
    #return sorted(PreOrderIter(btree), key=lambda node: getattr(node, 'dist', np.inf))
    return sorted((n for n in PreOrderIter(tree) if not n.is_chunk), key=lambda node: getattr(node, 'ddist', np.inf))

def do_ddist_flattening(tree, threshold_ddist=None):
    tree=copy.deepcopy(tree)
    add_ddist(tree)
    ddist_sorted_nodes= get_ddist_sorted_nodes(tree)

    deleted_nodes=[]
    affected_nodes=[]

    i=0
    while True:
        node=ddist_sorted_nodes.pop(0)

        if node.ddist >= threshold_ddist: break
  
        children=[child for child in node.children]

        if any(c.name in affected_nodes for c in children):
            ddist_sorted_nodes= get_ddist_sorted_nodes(tree)
            #print(f"i={i}\taffected_children: {len(affected_nodes)}\tdeleted_nodes: {len(deleted_nodes)}"+25*" ")
            affected_nodes=[]
            continue

        for child in children:
            child.ddist=node.parent.dist-child.dist
            child.parent=node.parent
            affected_nodes.append(child.name)

        node.parent=None
        deleted_nodes.append(node.name)

        i+=1
        # if verbose:
        #     print(f"i={i} {len(deleted_nodes)}/{num_delete_nodes}\t"+25*" ",end='\r')
    
    logging.info(get_tree_info_dict(tree))
    return tree



def recover_leaf_parents(tree):

    used_ids=[n.name for n in PreOrderIter(tree)]
    free_ids=set(range(max(used_ids)+1))-set(used_ids)

    new_ids=[]
    for node in PreOrderIter(tree):

        if node.is_leaf: continue
        children_is_leaf=[n.is_leaf for n in node.children]

        num_leaf_children=children_is_leaf.count(True)
        num_node_children=children_is_leaf.count(False)

        if num_node_children==0: continue
        if num_leaf_children==0: continue

        old_child_order=[n for n in node.children if not n.is_leaf]
        new_parent_node_id=free_ids.pop()
        new_parent_node=Node(new_parent_node_id)

        #new_dist=np.mean([child.dist for child in node.children if not child.is_leaf])
        new_parent_node.__setattr__('dist',node.dist)

        new_ids.append(new_parent_node_id)

        for child in node.children:
            if child.is_leaf:
                child.parent=new_parent_node
                
        new_parent_node.parent=node
        new_child_order=[new_parent_node]+old_child_order
        node.children=tuple(new_child_order)

    logging.info(get_tree_info_dict(tree))
    return tree