from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.visualization import * 

from beesup_llm.pipeline import BasePipeline


from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import pickle

class ScipyClusteringPipeline(BasePipeline):
    type='clustering'

    def __init__(self, ref=None, dataset_ref=None, **kwargs):
        super().__init__(ref, dataset_ref, **kwargs)
        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['linkage_matrix'])

        self._default_config=dict(
            linkage_args=dict(
                method='average',
                metric='cosine',
                optimal_ordering=False
            )
        )
        self.update_attributes(self._default_config, overwrite=False)


    def get_linkage_matrix(self, df=None):
        

        if df is None: df = self.df

        # if os.path.exists(f'{self.path}/linkage_matrix.pkl'):
        #     with open(f'{self.path}/linkage_matrix.pkl', 'rb') as f:
        #         self.linkage_matrix = pickle.load(f)
        
        linkage_matrix = linkage(np.vstack(df['emb'].values), **self.linkage_args)
        self.logger.info(f"shape {linkage_matrix.shape}")

        self.get_linkage_matrix = linkage_matrix
        return linkage_matrix
