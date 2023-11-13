import pandas as pd
import glob
import torch
import os
from torch_geometric.data import Data


def combine(file_path):
    # List of all files that you want to combine
    files = glob.glob(file_path + '/*.tsv')

    # Initialize an empty DataFrame to store the final combined data
    combined_data = pd.DataFrame(columns=['head_entity', 'relation', 'tail_entity', 'probability'])

    # Loop through each file and append its content to the combined_data DataFrame
    for file in files:
        # Read the file into a DataFrame
        df = pd.read_csv(file, sep='\t', header=None, names=['head_entity', 'relation', 'tail_entity', 'probability'])
        # Append the DataFrame to combined_data
        combined_data = combined_data.append(df, ignore_index=True)

    # Remove duplicate rows
    combined_data.drop_duplicates(subset=['head_entity', 'relation', 'tail_entity', 'probability'], keep='first',
                                  inplace=True)

    # Save the combined and deduplicated data to a new file
    path = file_path.split('/')[1]
    filename = 'combined_data_' + path + '.csv'
    full_path = os.path.join(file_path, filename)
    combined_data.to_csv(full_path, sep='\t', index=False, header=False)


def kg_split(file_name):
    # Load the combined KG into a DataFrame
    path = os.path.dirname(file_name)
    combined_kg = pd.read_csv(file_name, sep='\t', header=None,
                              names=['head_entity', 'relation', 'tail_entity', 'probability'])
    print(combined_kg.dtypes)
    # Convert the DataFrame columns to appropriate types
    combined_kg['head_entity'] = pd.to_numeric(combined_kg['head_entity'], errors='coerce')
    combined_kg['relation'] = pd.to_numeric(combined_kg['relation'], errors='coerce')
    combined_kg['tail_entity'] = pd.to_numeric(combined_kg['tail_entity'], errors='coerce')
    combined_kg['probability'] = pd.to_numeric(combined_kg['probability'], errors='coerce')

    # Drop any rows that have NaN values after the conversion
    combined_kg.dropna(inplace=True)

    # Create a graph using PyTorch Geometric
    edge_index = torch.tensor(combined_kg[['head_entity', 'tail_entity']].values, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(combined_kg[['probability']].values, dtype=torch.float)
    graph = Data(edge_index=edge_index, edge_attr=edge_attr)

    # Partition the graph into two subgraphs (half)
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges

    # Create masks for the two subgraphs
    mask1 = torch.zeros(num_edges, dtype=torch.bool)
    mask1[:num_edges // 2] = 1
    mask2 = ~mask1

    # Create the subgraphs
    edge_index1 = graph.edge_index[:, mask1]
    edge_attr1 = graph.edge_attr[mask1]
    edge_index2 = graph.edge_index[:, mask2]
    edge_attr2 = graph.edge_attr[mask2]
    graph1 = Data(edge_index=edge_index1, edge_attr=edge_attr1)
    graph2 = Data(edge_index=edge_index2, edge_attr=edge_attr2)

    # Create a mapping for the entities to new entity names
    unique_nodes = torch.unique(torch.cat([graph1.edge_index.flatten(), graph2.edge_index.flatten()]))
    entity_mapping = {int(node): 10000 + i for i, node in enumerate(unique_nodes)}

    # Apply the mapping to the subgraphs
    graph1.edge_index = torch.tensor([[entity_mapping[int(node)] for node in graph1.edge_index[0]],
                                      [entity_mapping[int(node)] for node in graph1.edge_index[1]]])

    graph2.edge_index = torch.tensor([[entity_mapping[int(node)] for node in graph2.edge_index[0]],
                                      [entity_mapping[int(node)] for node in graph2.edge_index[1]]])

    # Save the subgraphs and entity mapping to files
    kg1_df = pd.DataFrame({'head_entity': graph1.edge_index[0], 'tail_entity': graph1.edge_index[1],
                           'probability': graph1.edge_attr.flatten()})
    kg2_df = pd.DataFrame({'head_entity': graph2.edge_index[0], 'tail_entity': graph2.edge_index[1],
                           'probability': graph2.edge_attr.flatten()})

    kg1_df.to_csv(path + '/KG1.csv', index=False)
    kg2_df.to_csv(path + '/KG2.csv', index=False)

    entity_pair_df = pd.DataFrame(list(entity_mapping.items()), columns=['Original', 'Mapped'])
    entity_pair_df.to_csv(path +'/entity_pair.csv', index=False)


def entity_mapping(file):
    # Read the data into a DataFrame
    df = pd.read_csv(file, sep='\t', header=None,
                     names=['head_entity', 'relation', 'tail_entity', 'probability'])
    # Create mappings for entities and relations
    entity_mapping = {entity: i for i, entity in enumerate(pd.concat([df['head_entity'], df['tail_entity']]).unique())}
    relation_mapping = {relation: i for i, relation in enumerate(df['relation'].unique())}

    # Create mapping DataFrames
    entity_mapping_df = pd.DataFrame(list(entity_mapping.items()), columns=['Original_Entity', 'Mapped_ID'])
    relation_mapping_df = pd.DataFrame(list(relation_mapping.items()), columns=['Original_Relation', 'Mapped_ID'])

    # Save mapping DataFrames to CSV files
    entity_mapping_df.to_csv('entity_mapping.csv', index=False)
    relation_mapping_df.to_csv('relation_mapping.csv', index=False)

    # Map the original entities and relations to their corresponding IDs
    df['head_entity'] = df['head_entity'].map(entity_mapping)
    df['tail_entity'] = df['tail_entity'].map(entity_mapping)
    df['relation'] = df['relation'].map(relation_mapping)

    # Save the mapped DataFrame to a new CSV file
    df.to_csv('mapped_data.csv', sep='\t', index=False)


if __name__ == '__main__':
    path = 'data'
    # List all folders in the directory
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    for folder in folders:
        path = 'data/' + folder
        print(path)
        #combine(path)
    #file = 'data/nl27k/combined_data_nl27k.csv'
    #entity_mapping(file)
    KGs = ['data/cn15k/combined_data_cn15k.csv', 'data/nl27k/mapped_data.csv', 'data/ppi5k/combined_data_ppi5k.csv']
    for KG in KGs:
        print(KG)
        kg_split(KG)

