
import numpy as np
import nrrd
import matplotlib.pyplot as plt

dir_data = "Y:/yskim/BoundingBox/data/processed/Validation/val/"
file_name = "ANO_0050_0108.npz"
seg_colon = f"{file_name.split('_')[1]}"
slice_num = int(file_name.split('_')[2][:-4])
seg_data, header = nrrd.read(f"Y:/yskim/BoundingBox/data/raw/Colorectal/sin_fast_colon/{seg_colon}.seg.nrrd")
target_seg = seg_data[:,:,slice_num]
target_seg = np.flip(target_seg, axis=0)
target_seg = np.rot90(target_seg, 3)


target_seg = seg_data[:,:,slice_num]
target_seg = np.flip(target_seg, axis=0)
target_seg = np.rot90(target_seg, 3)


dir_save = f"Y:/yskim/BoundingBox/{file_name.split('_')[0]}_{seg_colon}_{slice_num}_colon.tiff"

plt.imshow(target_seg, cmap='gray')
plt.tight_layout()
plt.axis('off')
plt.savefig(dir_save)
plt.close()