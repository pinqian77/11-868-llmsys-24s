from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


# ASSIGNMENT 4.1
class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN SOLUTION
        data_idx = self.index[index]
        return self.data[data_idx]
        # END SOLUTION

# ASSIGNMENT 4.1
class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN SOLUTION
        # Create a list of shuffled indices
        data_len = len(data)
        indices = list(range(data_len))
        rng.shuffle(indices)

        # Partition the indices
        self.partitions = self.partition_indices(indices, sizes)
        # END SOLUTION
    
    def partition_indices(self, indices, sizes):
        """ Splits indices according to specified sizes. """
        partitions = []
        data_len = len(indices)
        indices_per_partition = [int(round(size * data_len)) for size in sizes]
        
        current = 0
        for size in indices_per_partition:
            if current + size > data_len:
                size = data_len - current  # Adjust size if it goes beyond data length
            partitions.append(indices[current:current + size])
            current += size
        return partitions

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN SOLUTION
        return Partition(self.data, self.partitions[partition])
        # END SOLUTION

# ASSIGNMENT 4.1
def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN SOLUTION
    partitioned_batch_size = batch_size // world_size
    partition_sizes = [1.0 / world_size for _ in range(world_size)]

    partitioner = DataPartitioner(dataset, sizes=partition_sizes)
    partitioned_dataset = partitioner.use(partition=rank)

    dataloader = DataLoader(partitioned_dataset, batch_size=partitioned_batch_size, collate_fn=collate_fn)
    return dataloader
    
    # END SOLUTION
