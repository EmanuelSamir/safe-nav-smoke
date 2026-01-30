import numpy as np
from collections import deque
from typing import Dict, Any, List

class GenericReplayBuffer:
    """
    A generic replay buffer that stores samples as a dictionary of deques.
    Allows handling a variable number of data types (keys).
    """
    
    def __init__(self, buffer_size: int, data_keys: List[str]):
        """
        Initializes the buffer.
        
        :param buffer_size: The maximum size of the buffer.
        :param data_keys: A list of strings with the keys of the data to store.
        """
        self.buffer_size = int(buffer_size)
        self.data_keys = data_keys
        self.buffer: Dict[str, deque] = {}
        self.global_data: Dict[str, Any] = {}
        self.current_size = 0
        self.reset()

    def reset(self):
        """Initializes all deques for the specified keys."""
        print("Resetting replay buffer")
        self.buffer = {key: deque(maxlen=self.buffer_size) for key in self.data_keys}
        self.current_size = 0
        print("Replay buffer reset")

    def add(self, **kwargs: Any):
        """
        Adds a sample to the buffer using keyword arguments.
        Example: buffer.add(state=s, action=a, done=d)
        
        :param kwargs: The key-value pairs to be added to the buffer.
                       Keys must match self.data_keys.
        """
        # Ensure only defined keys are passed
        if set(kwargs.keys()) != set(self.data_keys):
            raise ValueError(f"The provided keys do not match the buffer's defined keys: {self.data_keys}")
            
        for key, value in kwargs.items():
            self.buffer[key].append(value)
            
        # Update current size (only needed once per addition)
        self.current_size = len(self.buffer[self.data_keys[0]])
        
    def get_from_index(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a full sample from a specific index.
        """
        if index >= self.current_size:
            raise IndexError("Index out of current buffer limits.")

        data = {key: self.buffer[key][index] for key in self.data_keys}
        return data

    def save_to_file(self, filepath: str, **kwargs):
        """
        Saves the buffer content to an .npz file.
        Additional keyword arguments can be passed to save global/static data.
        """
        data_to_save = {}
        for key in self.data_keys:
            # Try to save as a regular numpy array for efficiency
            try:
                # np.array(deque) usually works and stacks efficiently if shapes are consistent
                temp_array = np.array(self.buffer[key])
            except Exception:
                # Fallback to object array if shapes/types are inconsistent
                temp_array = np.array(list(self.buffer[key]), dtype=object) 
            
            data_to_save[key] = temp_array

        np.savez(filepath, **data_to_save, buffer_size=self.buffer_size, data_keys=self.data_keys, **kwargs)
        print(f"Buffer successfully saved to: {filepath}")

    def load_from_file(self, filepath: str):
        """
        Loads the buffer content from an .npz file.
        """
        # Load with mmap_mode=None to read into memory (standard) 
        # or we could let user decide. For now standard load.
        npzfile = np.load(filepath, allow_pickle=True)
        
        # 1. Load metadata
        if 'data_keys' in npzfile and 'buffer_size' in npzfile:
            loaded_keys = list(npzfile['data_keys'])
            loaded_buffer_size = npzfile['buffer_size'].item()
        else:
            print("The file does not contain 'data_keys' and 'buffer_size' keys.")
            loaded_keys = list(npzfile.keys())
            loaded_buffer_size = len(npzfile[loaded_keys[0]])
        
        if loaded_keys != self.data_keys or loaded_buffer_size != self.buffer_size:
             print(f"WARNING: Loaded keys ({loaded_keys}) or size ({loaded_buffer_size}) \n"
                   f"do not match the current configuration ({self.data_keys}, {self.buffer_size}). \n"
                   f"Using loaded configuration and data.")
             self.data_keys = loaded_keys
             self.buffer_size = loaded_buffer_size

        # 2. Rebuild deques
        self.reset()
        self.global_data = {}  # Store global/static data here

        print(f"Loading keys: {npzfile.files}")
        
        for key in npzfile.files:
            if key in ('data_keys', 'buffer_size'):
                continue
                
            if key in self.data_keys:
                print(f"Loading sequence data: {key}")
                data_array = npzfile[key]
                print(f"Data '{key}' shape: {data_array.shape}, dtype: {data_array.dtype}")
                
                if data_array.dtype == object:
                     # Legacy object-array support
                     # The old save method created arrays of objects (even for valid float matrices),
                     # which causes 1) huge file size/load time, 2) compatibility issues with torch.from_numpy.
                     print(f"Detected legacy object array for '{key}'. Attempting to optimize...")
                     try:
                         # Attempt 1: Cast the entire array to float32. 
                         # This works if the object array actually contains a regular matrix of numbers.
                         # This is much faster and fixes the memory layout.
                         converted_array = data_array.astype(np.float32)
                         print(f"Successfully cast '{key}' to float32.")
                         self.buffer[key] = deque(converted_array, maxlen=self.buffer_size)
                     except Exception as e:
                         print(f"Global cast failed ({e}). Fallback to row-wise conversion.")
                         # Attempt 2: If jagged or mixed, convert each element individually.
                         # We ensure each element becomes a numpy array (float) for the dataset.
                         # data_array is likely (N, ...) or (N,). Iterating yields rows.
                         # We force each row to be a float array.
                         cleaned_deque = deque()
                         for item in data_array:
                             # item is likely a 1D object array or list.
                             # np.array(item, dtype=np.float32) converts it.
                             cleaned_deque.append(np.array(item, dtype=np.float32))
                         self.buffer[key] = cleaned_deque
                else:
                     # Optimize loading: avoid tolist() conversion for native arrays
                     # deque can ingest the numpy array (iterating over the first dimension)
                     self.buffer[key] = deque(data_array, maxlen=self.buffer_size)
            else:
                # If key is effectively global/static (e.g. smoke_value_positions stored once)
                self.global_data[key] = npzfile[key]
                print(f"Loaded global data key: '{key}' with shape {self.global_data[key].shape}")

        self.current_size = len(self.buffer[self.data_keys[0]]) if self.data_keys else 0
        print(f"Buffer successfully loaded. Current size: {self.current_size}")


    def full(self) -> bool:
        """Checks if the buffer is full."""
        return self.current_size >= self.buffer_size
    
    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return self.current_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Allows accessing the buffer as buffer[index]."""
        return self.get_from_index(index)