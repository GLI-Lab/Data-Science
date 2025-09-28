import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, Data, Batch


class MUTAGFromCSV(Dataset):
    def __init__(self, root='./data/MUTAG'):
        super().__init__()
        self.graphs_df = pd.read_csv(os.path.join(root, 'graphs.csv'))
        self.nodes_df = pd.read_csv(os.path.join(root, 'nodes.csv'))
        self.edges_df = pd.read_csv(os.path.join(root, 'edges.csv'))
        
        self.data_list = self._process_data()
        self._create_data_batch()
    
    def _process_data(self):
        data_list = []
        
        for graph_id in self.graphs_df['graph_id'].unique():
            graph_info = self.graphs_df[self.graphs_df['graph_id'] == graph_id].iloc[0]
            
            graph_nodes = self.nodes_df[self.nodes_df['graph_id'] == graph_id].copy()
            graph_nodes = graph_nodes.sort_values('node_id')
            
            node_feature_cols = [col for col in graph_nodes.columns if col.startswith('feature_')]
            node_features = graph_nodes[node_feature_cols].values
            x = torch.tensor(node_features, dtype=torch.float)
            
            graph_edges = self.edges_df[self.edges_df['graph_id'] == graph_id]
            
            edge_index = torch.tensor(
                np.stack([
                    graph_edges['source_node'].values,
                    graph_edges['target_node'].values
                ]), dtype=torch.long
            )
            
            edge_feature_cols = [col for col in graph_edges.columns if col.startswith('feature_')]
            edge_features = graph_edges[edge_feature_cols].values
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            y = torch.tensor([graph_info['label']], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)
        
        return data_list

    def _create_data_batch(self):
        """
        Create batch for dataset statistics overview (optional)
        Not required for PyG Dataset functionality, but useful for data overview
        """
        self.data = Batch.from_data_list(self.data_list)
        self.y = self.data.y
        self.x = self.data.x
        self.edge_index = self.data.edge_index
        self.edge_attr = self.data.edge_attr
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    dataset = MUTAGFromCSV()

    print(f"Total {len(dataset)} Graphs: {dataset.data}")
    print(f"Total Label Distribution: {np.bincount(dataset.y)}")
    print(f"Graph (1/{len(dataset)}): {dataset[0]}")
    # print(f"- Node Features {list(dataset[0].x.shape)}:\n{dataset[0].x}")
    # print(f"- Edge Features {list(dataset[0].edge_attr.shape)}:\n{dataset[0].edge_attr}")
