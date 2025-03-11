import os
import pickle
import pandas as pd

from beesup_llm import get_labhandler, _isinstance
from beesup_llm.injection.taxomizing_utils import *
from beesup_llm.llm import LLMPipeline


from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

class Taxomizer(object):

    logger=logging.getLogger(__name__)

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe.generation_config['max_time'] = 1200
        llm_pipe.generation_config['stop_strings'] = ['\n']
        llm_pipe.generation_config['max_new_tokens'] = 100
        return llm_pipe
    
    def __init__(
            self,
            ref=None,
            label:str=None,
            chunks_df=None,
            llm_pipe=None,
            labh=get_labhandler(),
            **kwargs):

        self.label = label
        self.chunk_txt_key = kwargs.get('txt_key', 'spo')
        self.chunk_emb_key = kwargs.get('emb_key', 'emb')

        self.linkage_args=kwargs.get('linkage_args', dict(
            method=kwargs.get('method','ward'), #single #complete #average #weighted #centroid #median #ward
            optimal_ordering=False
        ))
        
        # DIST FLATTENING CONFIG (tree depth)
        self.dist_flattening_config=dict(
            include_leaves=False,
            use_kneepoint=True,
            use_std=False,
            std_factor=1.0,
        )
        self.dist_flattening_config.update(kwargs.get('dist_flattening_config',{}))
        if self.dist_flattening_config['use_kneepoint'] and self.dist_flattening_config['use_std']:
            raise ValueError(f"Cannot use both kneepoint and std for dist_flattening_config")

        # DDIST FLATTENING CONFIG (tree width)
        self.ddist_flattening_config=dict(
            include_leaves=False,
            use_kneepoint=True,
            use_std=False,
            std_factor=1.0,
        )
        self.ddist_flattening_config.update(kwargs.get('ddist_flattening_config',{}))
        if self.ddist_flattening_config['use_kneepoint'] and self.ddist_flattening_config['use_std']:
            raise ValueError(f"Cannot use both kneepoint and std for ddist_flattening_config")

        # Storage Handling
        if labh is not None:
            self.labh=labh(locals())
            chunks_df=self.labh.handle_object(locals(),'chunks_df', save_file=True, overwrite=False)
            llm_pipe=self.labh.handle_object(locals(),'llm_pipe')
        else:
            self._path = os.getcwd()
        

        if not hasattr(self,'tree_info'):
            self.tree_info=dict()

        
        if isinstance(chunks_df, pd.DataFrame):
            self.chunks_df = chunks_df.copy(); del chunks_df

        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe=self.fit_llm_pipe(llm_pipe, **kwargs)
            self.llm_pipe=llm_pipe
        
        self.load_data()

    def load_linkage_matrix(self, chunks_df:pd.DataFrame, **kwargs) -> None:

        distance_matrix = cosine_distances(np.vstack(chunks_df[self.chunk_emb_key].values))
        distance_matrix = distance_matrix.astype(np.float64)
        distance_matrix = squareform(distance_matrix, checks=False)
        linkage_matrix=linkage(distance_matrix, **self.linkage_args)

        with open(f"{self._path}/linkage_matrix.pkl", "wb") as f:
            pickle.dump(linkage_matrix, f)
        
        self.linkage_matrix=linkage_matrix
        return
    
    def load_bin_tree(self, linkage_matrix: np.ndarray, chunks_df: pd.DataFrame, **kwargs) -> None:
        bin_tree=linkage_to_btree(linkage_matrix, chunks_df)
        self.tree_info['bin_tree']=get_tree_info_dict(bin_tree)

        with open(f"{self._path}/bin_tree.pkl", "wb") as f:
            pickle.dump(bin_tree, f)
        
        self.bin_tree=bin_tree
        return

    def load_dist_tree(self, bin_tree: Node, **kwargs) -> None:

        # DIST FLATTENING
        include_leaves=self.dist_flattening_config['include_leaves']
        if self.dist_flattening_config['use_kneepoint']:
            dist_threshold, index = get_dist_kneepoint(bin_tree,  include_leaves=include_leaves, **kwargs)
        elif self.dist_flattening_config['use_std']:
            dist_threshold = get_dist_std(bin_tree, std_factor=self.dist_flattening_config['std_factor'], include_leaves=include_leaves, **kwargs)
        
        dist_tree=do_dist_flattening(bin_tree, threshold_dist=dist_threshold)
        self.tree_info['dist_threshold']=dist_threshold

        self.tree_info['dist_tree']=get_tree_info_dict(dist_tree)

        with open(f"{self._path}/dist_tree.pkl", "wb") as f:
            pickle.dump(dist_tree, f)
    
        self.dist_tree=dist_tree
        return 

    def load_ddist_tree(self, dist_tree: Node, **kwargs) -> None:
        add_ddist(dist_tree)
        include_leaves=self.ddist_flattening_config['include_leaves']
        if self.ddist_flattening_config['use_kneepoint']:
            ddist_threshold, index = get_ddist_kneepoint(dist_tree, include_leaves=include_leaves, **kwargs)
        elif self.ddist_flattening_config['use_std']:
            ddist_threshold = get_ddist_std(dist_tree, std_factor=self.ddist_flattening_config['std_factor'], include_leaves=include_leaves, **kwargs)

        ddist_tree=do_ddist_flattening(dist_tree, threshold_ddist=ddist_threshold)
        self.tree_info['ddist_threshold']=ddist_threshold
        self.tree_info['ddist_tree_raw']=get_tree_info_dict(ddist_tree)

        ddist_tree=recover_leaf_parents(ddist_tree)
        self.tree_info['ddist_tree_rec']=get_tree_info_dict(ddist_tree)

        for node in PreOrderIter(ddist_tree):
            if not hasattr(node,'is_chunk'):
                if node.is_leaf: node.__setattr__('is_chunk',True)
                else: node.__setattr__('is_chunk',False)
                print(node.name, end=', ')

        with open(f"{self._path}/ddist_tree.pkl", "wb") as f:
            pickle.dump(ddist_tree, f)
        
        self.ddist_tree=ddist_tree
        return
    
    def load_emb_tree(self, ddist_tree: Node, chunks_df: pd.DataFrame=None, verbose=False, **kwargs) -> None:

        propagate_emb_from_leaves(ddist_tree)
        emb_tree = add_order_idc(ddist_tree, chunks_df, verbose=verbose)

        # CHECK if all chunks are included in the tree
        chunks_df['node_id']=None
        for node in PreOrderIter(emb_tree):
            if node.is_leaf: continue
            if all([d.is_leaf for d in node.children]):
                chunks_df.loc[node.include_chunk_idc,'node_id']=node.name
        self.logger.info(f"all chunks included in the tree: {len(chunks_df[chunks_df['node_id'].isna()])==0}")

        # TEST IF EXCLUDE and INCLUDE CHUNKS ARE DISJOINT
        for node in PreOrderIter(emb_tree):
            if node.is_leaf: continue
            
            include_chunk_idc=set(node.include_chunk_idc)
            exclude_chunk_idc=set(node.exclude_chunk_idc)

            intersection=include_chunk_idc.intersection(exclude_chunk_idc)
            if len(intersection)>0:
                self.logger.warning(f"Node {node.name} has overlapping include and exclude chunks: {intersection}")

        with open(f"{self._path}/emb_tree.pkl", "wb") as f:
            pickle.dump(emb_tree, f)
        chunks_df.to_pickle(f"{self._path}/chunks_df.pkl")

        self.emb_tree=emb_tree
        self.chunks_df=chunks_df
        return 

    def load_llm_tree(self, emb_tree:Node, chunks_df:pd.DataFrame, verbose=False, **kwargs):

        llm_tree=copy.deepcopy(reset_headers(emb_tree))
        self.llm_pipe.prepare_inference()

        if verbose: print(f"[{llm_tree.name}] {llm_tree.header}")

        for pre, fill, node in RenderTree(llm_tree):
            if node.is_leaf: continue
            if node.is_root: continue
            prompt=get_header_prompt(node, llm_tree, chunks_df)
            header=self.llm_pipe(prompt,use_chatformat=True)
            header=clean_header(header)
            node.__setattr__('header',header)

            if verbose: print(f"{pre} [{node.name}] {node.header}")

        with open(f"{self._path}/llm_tree.pkl", "wb") as f:
            pickle.dump(llm_tree, f)
        
        self.llm_tree=llm_tree
        return

    def load_data(self):

        if os.path.exists(f"{self._path}/linkage_matrix.pkl"):
            with open(f"{self._path}/linkage_matrix.pkl", "rb") as f:
                self.linkage_matrix=pickle.load(f)

        if os.path.exists(f"{self._path}/bin_tree.pkl"):
            with open(f"{self._path}/bin_tree.pkl", "rb") as f:
                self.bin_tree=pickle.load(f)
        
        if os.path.exists(f"{self._path}/dist_tree.pkl"):
            with open(f"{self._path}/dist_tree.pkl", "rb") as f:
                self.dist_tree=pickle.load(f)
        
        if os.path.exists(f"{self._path}/ddist_tree.pkl"):
            with open(f"{self._path}/ddist_tree.pkl", "rb") as f:
                self.ddist_tree=pickle.load(f)

        if os.path.exists(f"{self._path}/emb_tree.pkl"):
            with open(f"{self._path}/emb_tree.pkl", "rb") as f:
                self.emb_tree=pickle.load(f)
            self.logger.debug(f"Loaded emb_tree from {self._path}/emb_tree.pkl")

        if os.path.exists(f"{self._path}/llm_tree.pkl"):
            with open(f"{self._path}/llm_tree.pkl", "rb") as f:
                self.llm_tree=pickle.load(f)
            self.logger.debug(f"Loaded llm_tree from {self._path}/llm_tree.pkl")
        
        if os.path.exists(f"{self._path}/chunks_df.pkl"):
            self.chunks_df=pd.read_pickle(f"{self._path}/chunks_df.pkl")
            self.logger.debug(f"Loaded chunks_df from {self._path}/chunks_df.pkl")

    def process_until_ddist_tree(self, verbose=False, **kwargs) -> None:

        self.load_linkage_matrix(self.chunks_df, **kwargs)
        self.load_bin_tree(self.linkage_matrix, self.chunks_df)
        self.load_dist_tree(copy.deepcopy(self.bin_tree),**kwargs)
        self.load_ddist_tree(copy.deepcopy(self.dist_tree), **kwargs)

        if hasattr(self, 'save_config'): self.save_config() # using labhandler to save config (tree_info)

    def process(self, verbose=False, **kwargs) -> None:

        if not hasattr(self,'emb_tree'):
            self.load_linkage_matrix(self.chunks_df, **kwargs)
            self.load_bin_tree(self.linkage_matrix, self.chunks_df)
            self.load_dist_tree(copy.deepcopy(self.bin_tree),**kwargs)
            self.load_ddist_tree(copy.deepcopy(self.dist_tree), **kwargs)
            self.load_emb_tree(copy.deepcopy(self.ddist_tree), self.chunks_df, verbose=verbose)

            if hasattr(self, 'save_config'): self.save_config() # using labhandler to save config (tree_info)
        
        if not hasattr(self,'llm_tree'):
            self.load_llm_tree(self.emb_tree, self.chunks_df, verbose=verbose)

            if hasattr(self, 'save_config'): self.save_config() # using labhandler to save config (tree_info)

    def get_table_of_contents(self):
        return get_table_of_contents(self.llm_tree)

    @property
    def toc(self):
        return self.get_table_of_contents()

    


        
        



    


    



    

        
    




        



