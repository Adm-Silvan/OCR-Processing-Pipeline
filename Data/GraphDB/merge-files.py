import os
import rdflib

input_folder = 'C:/Users/Silvan/Data/AIS'
output_file = 'C:/Users/Silvan/Data/AIS/ais-combined.ttl'
graph = rdflib.Graph()

def merge_turtle_batches(input_folder, output_folder, batch_size_mb=750):
    batch_size_bytes = batch_size_mb * 1024 * 1024
    current_graph = rdflib.Graph()
    current_batch_size = 0
    batch_index = 1
    
    ttl_files = [f for f in os.listdir(input_folder) if f.endswith('.ttl')]
    ttl_files.sort()  # Sort files for consistency
    
    for filename in ttl_files:
        filepath = os.path.join(input_folder, filename)
        file_size = os.path.getsize(filepath)
        
        # If adding this file exceeds batch size, serialize current batch and start new graph
        if current_batch_size + file_size > batch_size_bytes and len(current_graph) > 0:
            output_file = os.path.join(output_folder, f'combined_batch_{batch_index}.ttl')
            current_graph.serialize(destination=output_file, format='turtle')
            print(f'Written batch {batch_index} to {output_file}')
            batch_index += 1
            current_graph = rdflib.Graph()
            current_batch_size = 0
        
        # Parse and add the current file's triples to the current graph
        current_graph.parse(filepath, format='turtle')
        current_batch_size += file_size
    
    # Serialize any remaining triples in the last batch
    if len(current_graph) > 0:
        output_file = os.path.join(output_folder, f'combined_batch_{batch_index}.ttl')
        current_graph.serialize(destination=output_file, format='turtle')
        print(f'Written final batch {batch_index} to {output_file}')

output_folder = 'C:/Users/Silvan/Data/AIS/combined'
os.makedirs(output_folder, exist_ok=True)
merge_turtle_batches(input_folder, output_folder, batch_size_mb=750)
