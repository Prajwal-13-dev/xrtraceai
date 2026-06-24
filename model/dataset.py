import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class XRTraceDataset(Dataset):
    def __init__(self, data_dir, seq_length=45, stride=15):
        """
        PyTorch Dataset for sliding-window sequence classification.
        Features memory-efficient lazy loading and mathematically bounded window labeling.
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.stride = stride
        
        # Storing only metadata (paths and indices) instead of raw matrices
        self.samples = [] 
        self.labels = []
        
        self._build_index()

    def _build_index(self):
        print(f"Building memory-mapped index from {self.data_dir}...")
        brv_files = sorted(Path(self.data_dir).glob("*_brv.npy"))
        
        for brv_path in brv_files:
            session_id = brv_path.stem.replace("_brv", "")
            label_path = brv_path.parent / f"{session_id}_labels.npy"
            
            if not label_path.exists():
                continue
                
            # Load labels into RAM and instantly check the BRV shape via mmap
            label_data = np.load(label_path).astype(np.int64)
            brv_mmap_check = np.load(brv_path, mmap_mode='r')
            
            # Catches stale files from old debugging cycles before they crash PyTorch
            assert len(label_data) == len(brv_mmap_check), (
                f"Data mismatch in {session_id}! "
                f"Labels: {len(label_data)} frames | BRV: {len(brv_mmap_check)} frames. "
                "Delete these stale files and re-run your extraction pipeline."
            )
            
            num_frames = len(label_data)
            
            # Slide the window across the session
            for start_idx in range(0, num_frames - self.seq_length + 1, self.stride):
                end_idx = start_idx + self.seq_length
                window_labels = label_data[start_idx:end_idx]
                
                # We require at least 15 anomalous frames in the window to label it as an anomaly.
                if (window_labels == 3).sum() >= 15:
                    window_label = 3
                else:
                    # Otherwise, use strict majority voting (the mode) for normal classes
                    counts = np.bincount(window_labels)
                    window_label = np.argmax(counts)
                
                # Store only the string path and the integer index
                self.samples.append((str(brv_path), start_idx))
                self.labels.append(window_label)
                
        print(f"Index complete. Total memory-mapped windows: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        brv_path, start_idx = self.samples[idx]
        end_idx = start_idx + self.seq_length
        
        brv_mmap = np.load(brv_path, mmap_mode='r')
        
        # Slice the specific window and convert to a concrete numpy array
        window_brv = np.array(brv_mmap[start_idx:end_idx], dtype=np.float32)
        
        # Convert to PyTorch tensors
        x = torch.tensor(window_brv)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y
    
if __name__ == "__main__":
    TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
    # Using the exact parameters your mentor calculated
    dataset = XRTraceDataset(TRAIN_DIR, seq_length=60, stride=15)
    
    # Grab the first batch to verify shape
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    
    print("\n=== DATASET VERIFICATION ===")
    print(f"X Batch Shape: {x_batch.shape} -> (Batch, Seq_Len, Features)")
    print(f"Y Batch Shape: {y_batch.shape} -> (Batch)")