from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    total_clock_cycles = num_batches + num_partitions - 1

    schedule = []
    for clock_cycle in range(total_clock_cycles):
        jobs = []
        for partition_idx in range(num_partitions):
            micro_batch_idx = clock_cycle - partition_idx
            if micro_batch_idx >= 0 and micro_batch_idx < num_batches:
                jobs.append((micro_batch_idx, partition_idx))
        schedule.append(jobs)
    
    return schedule
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        '''
        # BEGIN SOLUTION
        # Calculate the number of micro-batches
        num_micro_batches = (x.size(0) + self.split_size - 1) // self.split_size
        
        # Split the input tensor into micro-batches
        micro_batches = [x[i * self.split_size:(i + 1) * self.split_size] for i in range(num_micro_batches)]

        # Generate a schedule for processing the micro-batches
        schedule = list(_clock_cycles(num_micro_batches, len(self.partitions)))

        # Process the micro-batches according to the schedule
        self.compute(micro_batches, schedule)

        # Concatenate and return the processed micro-batches as the final output
        return torch.cat(micro_batches, dim=0)
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        print(f"devices:{devices}")

        # BEGIN SOLUTION
        for clock_cycle in schedule:
            tasks = []

            # Schedule tasks for the current clock cycle
            for batch_index, partition_index in clock_cycle:
                partition = partitions[partition_index]
                device = devices[partition_index]
                micro_batch = batches[batch_index]

                def compute_fn(micro_batch=micro_batch, partition=partition, device=device):
                    if isinstance(micro_batch, tuple):
                        # print(f"tuple: {len(batch)}")
                        # print(f"batch[0]: {batch[0].shape}")
                        micro_batch = micro_batch[0].to(device)
                    else:
                        # print(batch.shape)
                        micro_batch = micro_batch.to(device)

                    return partition(micro_batch)

                # Create a Task object for the computation
                task = Task(compute_fn)
                self.in_queues[partition_index].put(task)
                tasks.append((batch_index, partition_index))

            # Collect results for all tasks scheduled in this clock cycle
            for batch_index, partition_index in tasks:
                success, payload = self.out_queues[partition_index].get()
                if success:
                    # If the task was successful, payload is a tuple of (task, batch)
                    _, result_batch = payload
                    batches[batch_index] = result_batch
                else:
                    raise Exception(f"A task in the pipeline failed on partition")
        # END SOLUTION

