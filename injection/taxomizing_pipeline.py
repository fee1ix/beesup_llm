from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.injection.taxomizing_utils import *

from beesup_llm.dataset import *
from beesup_llm.model_pipelines import *


import pickle
import pandas as pd


from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


class TaxomizingPipeline(BaseDirectory):
    type='taxomizing_pipeline'

    def __init__(self, ref=None, dataset_ref=None, llm_ref=None, chunks_df=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(

            flattening_config=dict(
            ),

            linkage_args=dict(
                method='ward', #single #complete #average #weighted #centroid #median #ward
                optimal_ordering=False
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['chunks_df','nodes_df','tree'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()
        
        if dataset_ref:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            self.update_config(dict(dataset_config=self.dataset.get_config()), overwrite_if_conflict=False)

        if chunks_df:
            self.chunks_df=chunks_df
    
    def load_linkage_matrix(self, chunks_df):

        distance_matrix = cosine_distances(np.vstack(chunks_df['emb'].values))
        distance_matrix = distance_matrix.astype(np.float64)
        distance_matrix = squareform(distance_matrix, checks=False)

        self.linkage_matrix = linkage(distance_matrix, **self.linkage_args)

        return 
    
    def get_linkage_matrix(self, chunks_df=None):

        linkage_matrix=getattr(self, 'linkage_matrix', None)
        if linkage_matrix is None:
            self.load_linkage_matrix(chunks_df)
            linkage_matrix=self.linkage_matrix
            del self.linkage_matrix
            return linkage_matrix
        
        else:
            return self.linkage_matrix
        
    def get_flattened_tree(self, chunks_df=None):

        linkage_matrix=self.get_linkage_matrix(chunks_df)
        tree=linkage_to_btree(linkage_matrix, chunks_df)

        ### Flattening the tree: several options

        threshold_dist, index = get_dist_kneepoint(tree,  include_leaves=False)
        tree=do_dist_flattening(tree, threshold_dist=threshold_dist)

        add_ddist(tree)
        threshold_ddist, index = get_ddist_kneepoint(tree, include_leaves=False, plot=False)
        tree=do_ddist_flattening(tree, threshold_ddist=threshold_ddist)

        tree=recover_leaf_parents(tree)

        return tree   



    

        
    




        



