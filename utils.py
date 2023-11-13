# Import necessary libraries
import pandas as pd
import torch
from torch_geometric.data import Data

# Load the combined KG into a DataFrame
# This assumes that the file 'combined_data.txt' contains the KG data separated by tabs
combined_kg = pd.read_csv('combined_data.txt', sep='\t', header=None, names=['head_entity', 'relation', 'tail_entity', 'probability'])

# Create a graph using PyTorch Geometric
# Convert the DataFrame columns to a tensor and transpose it to match the expected shape
edge_index = torch.tensor(combined_kg[['head_entity', 'tail_entity']].values, dtype=torch.long).t().contiguous()

# Convert the 'probability' column to a tensor
edge_attr = torch.tensor(combined_kg[['probability']].values, dtype=torch.float)

# Create a PyTorch Geometric Data object
graph = Data(edge_index=edge_index, edge_attr=edge_attr)

# Partition the graph into two subgraphs
# For demonstration, we'll just split it in half

# Get the number of edges in the graph
num_edges = graph.num_edges

# Create masks for the two subgraphs
# Initialize a boolean tensor of zeros with the same length as the number of edges
mask1 = torch.zeros(num_edges, dtype=torch.bool)

# Set the first half of the mask to True
mask1[:num_edges // 2] = 1

# Create the second mask as the negation of the first mask
mask2 = ~mask1

# Create the subgraphs based on the masks
# Extract edges and attributes for the first subgraph
edge_index1 = graph.edge_index[:, mask1]
edge_attr1 = graph.edge_attr[mask1]

# Extract edges and attributes for the second subgraph
edge_index2 = graph.edge_index[:, mask2]
edge_attr2 = graph.edge_attr[mask2]

# Create Data objects for the subgraphs
graph1 = Data(edge_index=edge_index1, edge_attr=edge_attr1)
graph2 = Data(edge_index=edge_index2, edge_attr=edge_attr2)

# Create a mapping for the entities to new entity names
# Get unique nodes from both subgraphs and enumerate them starting from 10000
unique_nodes = torch.unique(torch.cat([graph1.edge_index.flatten(), graph2.edge_index.flatten()]))
entity_mapping = {int(node): 10000 + i for i, node in enumerate(unique_nodes)}

# Apply the mapping to the subgraphs
# Update the edge indices based on the new entity names
graph1.edge_index = torch.tensor([[entity_mapping[int(node)] for node in graph1.edge_index[0]],
                                  [entity_mapping[int(node)] for node in graph1.edge_index[1]]])
graph2.edge_index = torch.tensor([[entity_mapping[int(node)] for node in graph2.edge_index[0]],
                                  [entity_mapping[int(node)] for node in graph2.edge_index[1]]])

# Save the subgraphs and entity mapping to files
# Convert the Data objects to DataFrames and save them as CSV files
kg1_df = pd.DataFrame({'head_entity': graph1.edge_index[0], 'tail_entity': graph1.edge_index[1], 'probability': graph1.edge_attr.flatten()})
kg2_df = pd.DataFrame({'head_entity': graph2.edge_index[0], 'tail_entity': graph2.edge_index[1], 'probability': graph2.edge_attr.flatten()})
kg1_df.to_csv('KG1.csv', index=False)
kg2_df.to_csv('KG2.csv', index=False)

# Save the entity mapping as a CSV file
entity_pair_df = pd.DataFrame(list(entity_mapping.items()), columns=['Original', 'Mapped'])
entity_pair_df.to_csv('entity_pair.csv', index=False)
