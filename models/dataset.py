import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class XRTraceDataset(Dataset):
    def __init__(self, data_dir, seq_length=60, stride=15):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.stride = stride
        
        self.samples = [] 
        self.window_labels = [] 
        
        self._build_index()

    def _build_index(self):
        print(f"Building memory-mapped index from {self.data_dir}...")
        brv_files = sorted(Path(self.data_dir).glob("*_brv.npy"))
        
        for brv_path in brv_files:
            session_id = brv_path.stem.replace("_brv", "")
            label_path = brv_path.parent / f"{session_id}_labels.npy"
            
            if not label_path.exists():
                continue
                
            label_data = np.load(label_path).astype(np.int64)
            brv_mmap_check = np.load(brv_path, mmap_mode='r')
            
            assert len(label_data) == len(brv_mmap_check), (
                f"Data mismatch in {session_id}! "
                f"Labels: {len(label_data)} frames | BRV: {len(brv_mmap_check)} frames. "
                "Delete these stale files and re-run your extraction pipeline."
            )
            
            num_frames = len(label_data)
            
            for start_idx in range(0, num_frames - self.seq_length + 1, self.stride):
                end_idx = start_idx + self.seq_length
                window_labels_slice = label_data[start_idx:end_idx]
                
                # Minimum 15 anomalous frames required to flag the window
                if (window_labels_slice == 3).sum() >= 15:
                    window_label = 3
                else:
                    counts = np.bincount(window_labels_slice)
                    window_label = np.argmax(counts)
                
                self.samples.append((str(brv_path), start_idx))
                # UPDATE 2: Appending to the new variable name
                self.window_labels.append(window_label)
                
        print(f"Index complete. Total memory-mapped windows: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        brv_path, start_idx = self.samples[idx]
        end_idx = start_idx + self.seq_length
        
        brv_mmap = np.load(brv_path, mmap_mode='r')
        window_brv = np.array(brv_mmap[start_idx:end_idx], dtype=np.float32)
        
        x = torch.tensor(window_brv)
        y = torch.tensor(self.window_labels[idx], dtype=torch.long)
        
        return x, y