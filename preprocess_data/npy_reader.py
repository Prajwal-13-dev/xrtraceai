import numpy as np
import pandas as pd

# 1. Load the data
data = np.load(r'C:\Users\Student3\Documents\xrtraceai\preprocess_data\train\UG_00_R040-13July-Nespresso_labels.npy')

# 2. Provide exactly 22 column names
columns = [
    # --- 11 SPATIAL FEATURES ---
    'Head_Y', 'Head_Qx', 'Head_Qy', 'Head_Qz', 'Head_Qw',
    'L_PosX', 'L_PosY', 'L_PosZ', 'R_PosX', 'R_PosY', 'R_PosZ',
    
    # --- 11 VELOCITY FEATURES ---
    'Head_Y_Vel', 'Head_Qx_Vel', 'Head_Qy_Vel', 'Head_Qz_Vel', 'Head_Qw_Vel', 
    'L_VelX', 'L_VelY', 'L_VelZ', 'R_VelX', 'R_VelY', 'R_VelZ'
]

# Convert and save
df = pd.DataFrame(data, columns=columns)
df.to_csv('R073-20July-GoPro_Readable.csv', index=False)

print(f"Success! Data shape: {data.shape}")
print("Saved to CSV! You can now open R073-20July-GoPro_Readable.csv in Excel.")
