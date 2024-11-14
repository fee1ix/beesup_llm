from beesup_llm import *
from ..toolkit.setup_utils import *
#from ..toolkit.visualization import * 

from beesup_llm.model import *
from beesup_llm.dataset import *

from beesup_llm.taxomizing.taxomizing_utils import *

import pandas as pd

import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import pickle


class TaxomizingPipeline(BaseDirectory):
    type='taxomizing'

    def __init__(self, ref=None, dataset_ref=None, emb_model_ref=None, gen_model_ref=None, **kwargs):
        super().__init__(ref, **kwargs)
        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['chunks_df','nodes_df','tree'])

        self._default_config=dict(
            linkage_args=dict(
                method='average',
                metric='cosine',
                optimal_ordering=False
            )
        )
        self.update_attributes(self._default_config, overwrite=False)

        if dataset_ref is not None:
            if isinstance(dataset_ref, pd.DataFrame):
                self.chunks_df=dataset_ref

            else:
                self.dataset=BaseDataset(dataset_ref)
                self.chunks_df=self.dataset.dataset_df
                #self.dataset_config=dataset.get_config()
                #self.df=dataset.dataset_df


        if emb_model_ref is not None:
            self.emb_model=EmbModelWrap.from_ref(emb_model_ref)
        
        if gen_model_ref is not None:
            self.gen_model=GenModelWrap.from_ref(gen_model_ref)

    def get_linkage_matrix(self, df=None):
        
        if df is None: df = self.df

        # if os.path.exists(f'{self.path}/linkage_matrix.pkl'):
        #     with open(f'{self.path}/linkage_matrix.pkl', 'rb') as f:
        #         self.linkage_matrix = pickle.load(f)
        
        linkage_matrix = linkage(np.vstack(df['emb'].values), **self.linkage_args)
        self.logger.info(f"shape {linkage_matrix.shape}")

        self.linkage_matrix = linkage_matrix
        return linkage_matrix
    
    def get_cluster_tree(self, linkage_matrix=None):
        if linkage_matrix is None: linkage_matrix = self.linkage_matrix

        cluster_tree = to_tree(linkage_matrix, rd=False)

        self.cluster_tree = cluster_tree
        return cluster_tree
    
    def get_nodes_df(self, linkage_matrix=None, cluster_tree=None, df=None):
        if linkage_matrix is None: linkage_matrix = self.linkage_matrix
        if cluster_tree is None: cluster_tree = self.cluster_tree
        if df is None: df = self.df

        assign_data_tree(cluster_tree, df)

        from ..toolkit.visualization import assign_colors_tree
        assign_colors_tree(cluster_tree)

        node_data=get_node_data(cluster_tree)
        nodes_df=pd.DataFrame(node_data)
        self.nodes_df=nodes_df

        return nodes_df

class HDBScanTaxomizingPipeline(TaxomizingPipeline):
    def __init__(self, ref=None, dataset_ref=None, emb_model_ref=None, gen_model_ref=None, **kwargs):
        super().__init__(ref, dataset_ref, emb_model_ref, gen_model_ref, **kwargs)


        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict(
            cluster_config=dict(
                min_cluster_size=60, #5
                max_cluster_size=0, #0
                min_samples=None, #None
                p=2, #2
                alpha=1.0, #1.0
                leaf_size=40, #40
                cluster_selection_epsilon=0.1, #0.0
                cluster_selection_method="eom", #eom, leaf
                allow_single_cluster=False, #False
                metric='precomputed',
                approx_min_span_tree=True, #True
                gen_min_span_tree=False, #False
                cluster_selection_epsilon_max=float('inf'), #float('inf')
            )
        )
        self.update_attributes(self._default_config, overwrite=False)

    def get_cluster_config(self, **kwargs):

        cluster_config=self.cluster_config

        if 'cluster_config' in kwargs:
            kwargs=kwargs['cluster_config']

        cluster_config.update(filter_kwargs(kwargs,allowed_keys=cluster_config.keys()))

        return cluster_config
    
    def load_clusterer(self, **kwargs):
        self.clusterer=hdbscan.HDBSCAN(**self.get_cluster_config(**kwargs))

    
    def fit(self, data, **kwargs):

        cluster_config=self.get_cluster_config(**kwargs)
        #self.clusterer=hdbscan.HDBSCAN(**cluster_config)
        #(embs==np.vstack(chunks_df['emb'].values)).all()
        
        if cluster_config['metric']=='precomputed':
            data=cosine_distances(data)
            data = data.astype(np.float64)

        self.clusterer.fit(data)

        self.logger.info(f"number of clusters: {len(set(self.clusterer.labels_))}")
        self.logger.info(f"number of clustered: {np.sum(self.clusterer.labels_!=-1)}")
        self.logger.info(f"number of unclustered: {np.sum(self.clusterer.labels_==-1)}")

    
    def fit_chunks_df(self, chunks_df, **kwargs):

        self.fit(np.vstack(chunks_df['emb'].values), **kwargs)
        chunks_df['cid']=self.clusterer.labels_
        
        nodes_df=nodes_df_from_hdbscan(self.clusterer,chunks_df)
        nodes_df=add_root_row(nodes_df)
        nodes_df=propagate_mean(nodes_df,key='emb')
        

        tree=to_anytree(nodes_df)
        self.logger.info(f"tree.height: {tree.height}")


        member_tree=get_member_tree(tree)
        self.logger.info(f"member_tree.size: {member_tree.size}")
        member_df=to_nodes_df(member_tree)

        self.chunks_df=chunks_df
        self.nodes_df=nodes_df
        self.tree=tree
        self.member_tree=member_tree
        self.member_df=member_df

        return












    


    
    def get_nodes_df(self,**kwargs):

        cluster_config=self.cluster_config
        cluster_config.update(kwargs)

        clusterer = hdbscan.HDBSCAN(**cluster_config)

        distance_matrix = cosine_distances(np.vstack(self.chunks_df['emb'].values))
        distance_matrix = distance_matrix.astype(np.float64)

        clusterer.fit(distance_matrix)

        self.logger.info(f"number of clusters: {len(set(clusterer.labels_))}")
        self.logger.info(f"number of clustered: {np.sum(clusterer.labels_!=-1)}")
        self.logger.info(f"number of unclustered: {np.sum(clusterer.labels_==-1)}")

        nodes_df=nodes_df_from_hdbscan(clusterer,self.chunks_df)
        nodes_df=add_parent_embs(nodes_df)

        self.nodes_df=nodes_df

        return nodes_df




















