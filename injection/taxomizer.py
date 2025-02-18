from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.injection.taxomizer_utils import *

from beesup_llm.dataset import *
from beesup_llm.model_pipelines import *

import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


class Taxomizer(BaseDirectory):
    type='taxomizer'

    def __init__(self, ref=None, dataset_ref=None, llm_ref=None, df=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(

            dist_flattening_config=dict(
                include_leaves=False,
                use_kneepoint=True,
                use_std=False,
                std_factor=1.0,
            ),
            ddist_flattening_config=dict(
                include_leaves=False,
                use_kneepoint=True,
                use_std=False,
                std_factor=1.0,
            ),
            linkage_args=dict(
                method='ward', #single #complete #average #weighted #centroid #median #ward
                optimal_ordering=False
            ),

            llm_config=dict(
                generation_config=dict(
                    max_new_tokens=4096,
                    max_time=1200,
                ),
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['df','nodes_df','tree'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)    
        self.handle_flattening_config()

        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()
        
        if dataset_ref:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            self.update_config(dict(dataset_config=self.dataset.get_config()), overwrite_if_conflict=False)

        # load stored data if exists
        if self.is_spawned(): self.load()

        if isinstance(df, pd.DataFrame):
            self.df=df

    def load(self):

        if os.path.exists(f"{self._path}/flattened_tree.pkl"):
            with open(f"{self._path}/flattened_tree.pkl", "rb") as f:
                self.flattened_tree=pickle.load(f)
            self.logger.debug(f"Loaded flattened_tree from {self._path}/flattened_tree.pkl")

        if os.path.exists(f"{self._path}/embedding_tree.pkl"):
            with open(f"{self._path}/embedding_tree.pkl", "rb") as f:
                self.embedding_tree=pickle.load(f)
            self.logger.debug(f"Loaded embedding_tree from {self._path}/embedding_tree.pkl")
        
        if os.path.exists(f"{self._path}/header_tree.pkl"):
            with open(f"{self._path}/header_tree.pkl", "rb") as f:
                self.header_tree=pickle.load(f)
            self.logger.debug(f"Loaded header_tree from {self._path}/header_tree.pkl")
        
        if os.path.exists(f"{self._path}/df.pkl"):
            self.df=pd.read_pickle(f"{self._path}/df.pkl")
            self.logger.debug(f"Loaded df from {self._path}/df.pkl")

    def process(self, df=None, verbose=False):

        if not self.is_spawned(): raise ValueError("Cannot process without spawning")
        if isinstance(df, pd.DataFrame): self.df=df
        if not hasattr(self, 'df'): self.logger.warning("No df provided")
        
        df=self.df

        self.flattened_tree=self.get_flattened_tree(df)
        self.embedding_tree=self.get_embedding_tree(self.flattened_tree, df, verbose=verbose)
        self.header_tree=self.generate_headers(self.embedding_tree, df, verbose=verbose)
        
    def handle_flattening_config(self):
        for flattening_config in ['dist_flattening_config', 'ddist_flattening_config']:
            if getattr(self,flattening_config)['use_kneepoint'] and getattr(self,flattening_config)['use_std']:
                raise ValueError(f"Cannot use both kneepoint and std for {flattening_config}")

            # if getattr(self,flattening_config)['use_kneepoint']:
            #     setattr(self,flattening_config, 'std_factor', None)
        
    def load_linkage_matrix(self, df):

        distance_matrix = cosine_distances(np.vstack(df['emb'].values))
        distance_matrix = distance_matrix.astype(np.float64)
        distance_matrix = squareform(distance_matrix, checks=False)

        self.linkage_matrix = linkage(distance_matrix, **self.linkage_args)

        return 
    
    def get_linkage_matrix(self, df=None):

        linkage_matrix=getattr(self, 'linkage_matrix', None)
        if linkage_matrix is None:
            self.load_linkage_matrix(df)
            linkage_matrix=self.linkage_matrix
            del self.linkage_matrix
            return linkage_matrix
        
        else:
            return self.linkage_matrix
        
    def get_flattened_tree(self, df=None):

        linkage_matrix=self.get_linkage_matrix(df)
        tree=linkage_to_btree(linkage_matrix, df)

        ### Flattening the tree: several options
        self.flattening_info=dict()
        self.flattening_info['tree_before_flattening']=get_tree_info_dict(tree)

        # DIST FLATTENING
        include_leaves=self.dist_flattening_config['include_leaves']
        if self.dist_flattening_config['use_kneepoint']:
            threshold_dist, index = get_dist_kneepoint(tree,  include_leaves=include_leaves)
        elif self.dist_flattening_config['use_std']:
            threshold_dist = get_dist_std(tree, std_factor=self.dist_flattening_config['std_factor'], include_leaves=include_leaves)
        
        tree=do_dist_flattening(tree, threshold_dist=threshold_dist)

        self.flattening_info['threshold_dist']=threshold_dist
        self.flattening_info['tree_after_dist_flattening']=get_tree_info_dict(tree)

        # DDIST FLATTENING
        add_ddist(tree)
        include_leaves=self.ddist_flattening_config['include_leaves']
        if self.ddist_flattening_config['use_kneepoint']:
            threshold_ddist, index = get_ddist_kneepoint(tree, include_leaves=include_leaves)
        elif self.ddist_flattening_config['use_std']:
            threshold_ddist = get_ddist_std(tree, std_factor=self.ddist_flattening_config['std_factor'], include_leaves=include_leaves)

        tree=do_ddist_flattening(tree, threshold_ddist=threshold_ddist)
        self.flattening_info['threshold_ddist']=threshold_ddist
        self.flattening_info['tree_after_ddist_flattening']=get_tree_info_dict(tree)

        tree=recover_leaf_parents(tree)
        self.flattening_info['tree_after_recover_leaf_parents']=get_tree_info_dict(tree)


        for node in PreOrderIter(tree):
            if not hasattr(node,'is_chunk'):
                if node.is_leaf: node.__setattr__('is_chunk',True)
                else: node.__setattr__('is_chunk',False)
                print(node.name, end=', ')

        if self.is_spawned():
            set_config(self.get_config(),path=self._path)
            with open(f"{self._path}/flattened_tree.pkl", "wb") as f: pickle.dump(tree, f)

        return tree
    
    def get_embedding_tree(self, flattened_tree, df, verbose=False):

        propagate_emb_from_leaves(flattened_tree)
        tree = add_order_idc(flattened_tree, df, verbose=verbose)

        # CHECK if all chunks are included in the tree
        df['node_id']=None
        for node in PreOrderIter(tree):
            if node.is_leaf: continue
            if all([d.is_leaf for d in node.children]):
                df.loc[node.include_chunk_idc,'node_id']=node.name
        self.logger.info(f"all chunks included in the tree: {len(df[df['node_id'].isna()])==0}")

        # TEST IF EXCLUDE and INCLUDE CHUNKS ARE DISJOINT
        for node in PreOrderIter(tree):
            if node.is_leaf: continue
            
            include_chunk_idc=set(node.include_chunk_idc)
            exclude_chunk_idc=set(node.exclude_chunk_idc)

            intersection=include_chunk_idc.intersection(exclude_chunk_idc)
            if len(intersection)>0:
                self.logger.warning(f"Node {node.name} has overlapping include and exclude chunks: {intersection}")

        if self.is_spawned():
            with open(f"{self._path}/embedding_tree.pkl", "wb") as f:
                pickle.dump(tree, f)
            df.to_pickle(f"{self._path}/df.pkl")
        
        return tree
    
    def generate_headers(self, embedding_tree=None, df=None, verbose=False):

        if (not embedding_tree) and self.is_spawned():
            with open(f"{self._path}/embedding_tree.pkl", "rb") as f:
                embedding_tree=pickle.load(f)
        
        if (not isinstance(df,pd.DataFrame)) and self.is_spawned():
            df=pd.read_pickle(f"{self._path}/df.pkl")

        tree = reset_headers(embedding_tree)

        self.llm_pipe.prepare_inference()

        if verbose: print(f"[{tree.name}] {tree.header}")

        for pre, fill, node in RenderTree(tree):
            if node.is_leaf: continue
            if node.is_root: continue
            prompt=get_header_prompt(node, tree, df)
            header=self.llm_pipe(prompt,use_chatformat=True, stop_strings=['\n'], max_new_tokens=100)[0]['generated_text']
            header=clean_header(header)
            node.__setattr__('header',header)

            if verbose: print(f"{pre} [{node.name}] {node.header}")

        if self.is_spawned():
            with open(f"{self._path}/header_tree.pkl", "wb") as f:
                pickle.dump(tree, f)
        
        return tree

    def get_table_of_contents(self):
        return get_table_of_contents(self.header_tree)

    


        
        



    


    



    

        
    




        



