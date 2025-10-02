import os
import numpy as np
import nibabel as nib
import torch
from numba.cpython.slicing import slice_indices
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import random
import config

class MultiViewSplit(Dataset):
    def __init__(self, data_dir, mode="train", file_extension=".nii"):
        self.data_dir = data_dir
        self.modalities = ["flair", "t1", "t1ce", "t2"]
        self.mode = mode
        self.file_extension = file_extension
        
        inner_dir = None
        has_patient_folders = any(
            d.startswith("BraTS") and os.path.isdir(os.path.join(data_dir, d))
            for d in os.listdir(data_dir)
        )

        if has_patient_folders:
            inner_dir = data_dir
        else:
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and "MICCAI_BraTS" in item:
                    inner_dir = item_path
                    break

        if inner_dir is None:
            raise ValueError(f"Could not find patient folders (BraTS*) in {data_dir} or any MICCAI_BraTS subdirectory")

        self.inner_dir = inner_dir
        self.patients = sorted([
            d for d in os.listdir(inner_dir)
            if os.path.isdir(os.path.join(inner_dir, d)) and d.startswith("BraTS")
        ])

        self._create_slice_index()

        print(f"Dataset initialized with {len(self.patients)} patients")
        print(f"Total slices: {len(self.slice_indices)}")
        print(
            f"Axial: {self.orientation_counts['axial']}, Coronal: {self.orientation_counts['coronal']}, Sagittal: {self.orientation_counts['sagittal']}")


    def _create_slice_index(self):
        self.slice_indices = []
        self.orientation_counts = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        
        orientations = ['axial', 'coronal', 'sagittal']
        slice_axes = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        
        for patient_idx, patient_id in enumerate(self.patients):
            try:
                patient_path = os.path.join(self.inner_dir, patient_id)
                sample_path = os.path.join(patient_path, f"{patient_id}_{self.modalities[0]}{self.file_extension}")
                
                if not os.path.exists(sample_path):
                    print(f"Warning: File not found {sample_path}")
                    continue
                
                img = nib.load(sample_path)
                volume_shape = img.shape  # (X, Y, Z)
                
                for orientation in orientations:
                    axis = slice_axes[orientation]
                    num_slices = volume_shape[axis]
                    
                    for slice_idx in range(num_slices):
                        self.slice_indices.append((patient_idx, orientation, slice_idx))
                        self.orientation_counts[orientation] += 1
                        
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                continue


    def load_nifti(self, path):
        return np.array(nib.load(path).get_fdata(), dtype=np.float32)


    def normalize_image(self, img):
        mean_val = np.mean(img)
        std_val = np.std(img)
        normalized = (img - mean_val) / (std_val + 1e-6)
        return np.nan_to_num(normalized)


    def extract_slice(self, img, orientation, slice_idx):
        if orientation == 'sagittal':
            return img[slice_idx, :, :]      # left-to-right slices
        elif orientation == 'coronal':
            return img[:, slice_idx, :]      # front-to-back slices
        else:  # axial
            return img[:, :, slice_idx]      # horizontal slices


    def __getitem__(self, idx):
        try:
            patient_idx, orientation, slice_idx = self.slice_indices[idx]
            patient_id = self.patients[patient_idx]
            patient_path = os.path.join(self.inner_dir, patient_id)
            
            modalities = []
            for modality in self.modalities:
                file_path = os.path.join(patient_path, f"{patient_id}_{modality}{self.file_extension}")
                img = self.load_nifti(file_path)
                img = self.normalize_image(img)
                modalities.append(img)
            
            slice_data = []
            for img in modalities:
                slice_2d = self.extract_slice(img, orientation, slice_idx)
                slice_data.append(slice_2d)
            
            multi_modal_slice = np.stack(slice_data, axis=-1)
            multi_modal_slice = torch.tensor(multi_modal_slice, dtype=torch.float32)
            
            if self.mode == "train":
                seg_path = os.path.join(patient_path, f"{patient_id}_seg{self.file_extension}")
                if os.path.exists(seg_path):
                    seg = self.load_nifti(seg_path)
                    
                    seg_slice = self.extract_slice(seg, orientation, slice_idx)

                    seg_slice[seg_slice == 4] = 3
                    seg_slice = torch.tensor(seg_slice.astype(np.uint8), dtype=torch.long)
                    
                    return multi_modal_slice, seg_slice, orientation
                else:
                    print(f"Warning: Segmentation file not found for {patient_id}")
                    dummy_seg = torch.zeros(multi_modal_slice.shape[:2], dtype=torch.long)
                    return multi_modal_slice, dummy_seg, orientation
            else:
                return multi_modal_slice, orientation
                
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            dummy_slice = torch.zeros((240, 240, 4), dtype=torch.float32)
            if self.mode == "train":
                dummy_seg = torch.zeros((240, 240), dtype=torch.long)
                return dummy_slice, dummy_seg, "axial"
            return dummy_slice, "axial"


    def __len__(self):
        return len(self.slice_indices)


    def get_slice_info(self, idx):
        patient_idx, orientation, slice_idx = self.slice_indices[idx]
        patient_id = self.patients[patient_idx]
        return {
            'patient_id': patient_id,
            'patient_idx': patient_idx,
            'orientation': orientation,
            'slice_idx': slice_idx
        }


    def get_orientation_indices(self, orientation):
        indices = []
        for i, (_, ori, _) in enumerate(self.slice_indices):
            if ori == orientation:
                indices.append(i)
        return indices


def create_patient_level_split(dataset, val_split=0.2, seed=42):
    num_patients = len(dataset.patients)
    patients = dataset.patients.copy()

    random.seed(seed)
    random.shuffle(patients)

    num_val = int(num_patients * val_split)
    val_patients = set(patients[:num_val])
    train_patients = set(patients[num_val:])

    print(f"\nPatient split: {len(train_patients)} train, {len(val_patients)} val")

    train_indices = []
    val_indices = []

    for idx, (patient_idx,orientation, slice_idx) in enumerate(dataset.slice_indices):
        patient_id = dataset.patients[patient_idx]
        if patient_id in train_patients:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    print(f"Total slices: {len(train_indices)} train, {len(val_indices)} val")

    return train_indices, val_indices


def create_orientation_datasets_with_split(full_dataset, orientation, train_indices, val_indices):
    orientation_indices = full_dataset.get_orientation_indices(orientation)
    orientation_set = set(orientation_indices)

    train_ori_indices = [idx for idx in train_indices if idx in orientation_set]
    val_ori_indices = [idx for idx in val_indices if idx in orientation_set]

    train_dataset = Subset(full_dataset, train_ori_indices)
    val_dataset = Subset(full_dataset, val_ori_indices)

    print(f"{orientation.capitalize():9} - Train: {len(train_ori_indices):6}, Val: {len(val_ori_indices):6}")

    return train_dataset, val_dataset

#Test code
if __name__ == "__main__":
    print("=" * 40)
    print("Loading BraTS Training Data")
    print("=" * 40)

    config.print_config()

    if not config.check_data_paths():
        exit(1)


    # Load full training dataset
    full_train_ds = MultiViewSplit(
        data_dir=str(config.TRAIN_DATA_DIR),
        mode="train",
        file_extension=".nii"
    )

    train_indices, val_indices = create_patient_level_split(full_train_ds, val_split=0.2, seed=42)

    print("\n" + "=" * 40)
    print("Creating Orientation-Specific Datasets")
    print("=" * 40)

    # Create datasets for each orientation
    train_axial, val_axial = create_orientation_datasets_with_split(
        full_train_ds, 'axial', train_indices, val_indices
    )

    train_coronal, val_coronal = create_orientation_datasets_with_split(
        full_train_ds, 'coronal', train_indices, val_indices
    )

    train_sagittal, val_sagittal = create_orientation_datasets_with_split(
        full_train_ds, 'sagittal', train_indices, val_indices
    )

    print("\n" + "=" * 40)
    print("Testing Data Loading")
    print("=" * 40)

    # Test loading from each orientation
    if len(train_axial) > 0:
        sample_data, sample_mask, orientation = train_axial[0]
        print(f"\nAxial sample: {sample_data.shape}, mask: {sample_mask.shape}, orientation: {orientation}")

    if len(train_coronal) > 0:
        sample_data, sample_mask, orientation = train_coronal[0]
        print(f"Coronal sample: {sample_data.shape}, mask: {sample_mask.shape}, orientation: {orientation}")

    if len(train_sagittal) > 0:
        sample_data, sample_mask, orientation = train_sagittal[0]
        print(f"Sagittal sample: {sample_data.shape}, mask: {sample_mask.shape}, orientation: {orientation}")

    print("\n" + "=" * 40)
    print("Setup Complete - Ready for Training!")
    print("=" * 40)
    print("\nDatasets created:")
    print(f"  train_axial, val_axial")
    print(f"  train_coronal, val_coronal")
    print(f"  train_sagittal, val_sagittal")

