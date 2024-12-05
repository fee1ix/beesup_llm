from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.injection.taxomizing_utils import *

from beesup_llm.model import *
from beesup_llm.dataset import *

import pickle
import pandas as pd


from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform





class TaxomizingPipeline(BaseDirectory):
    type='taxomizing_pipeline'

    def __init__(self, ref=None, dataset_ref=None, model_ref=None, **kwargs):
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

        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )


        model_ref = model_ref or getattr(self, 'model_config', None)
        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)
            self.update_config(dict(model_config=self.modelwrap.get_config()), overwrite_if_conflict=False)
            
        dataset_ref = dataset_ref or getattr(self, 'dataset_config', None)
        if dataset_ref is not None:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            self.update_config(dict(dataset_config=self.dataset.get_config()), overwrite_if_conflict=False)
    

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
        
    




        



