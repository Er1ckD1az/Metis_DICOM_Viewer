import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import segmentation_models_pytorch as smp
from tqdm import tqdm
import tempfile
import re

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    ClientError = None
    NoCredentialsError = None


class BraTSSegmentationModel:
    #Single-model brain tumor segmentation for BraTS MRI data
    def __init__(self, model_checkpoint_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._temp_files = []

        print(f"Loading segmentation model on {self.device}...")
        try:
            self.model = self._load_model(model_checkpoint_path)
            print(f"Model loaded successfully")
        except Exception as e:
            self._cleanup_temp_files()
            raise RuntimeError(f"Failed to load model: {e}")

    def __del__(self):
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        #Clean up temporary downloaded files
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass

    def _is_s3_path(self, path):
        #Check if path is S3 URI
        return str(path).startswith("s3://")

    def _parse_s3_path(self, s3_uri):
        #Parse S3 URI into bucket and key
        match = re.match(r"s3://([^/]+)/(.+)", s3_uri)
        if not match:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        return match.group(1), match.group(2)

    def _download_from_s3(self, s3_uri):
        #Download model checkpoint from S3
        if boto3 is None:
            raise ImportError("boto3 is required for S3. Install with: pip install boto3")

        try:
            bucket, key = self._parse_s3_path(s3_uri)
            tmp_dir = Path(tempfile.gettempdir()) / "brats_models"
            tmp_dir.mkdir(exist_ok=True)
            local_path = tmp_dir / Path(key).name

            if local_path.exists():
                print(f"Using cached model: {local_path}")
                return local_path

            print(f"Downloading model from S3...")
            s3 = boto3.client("s3")
            s3.download_file(bucket, key, str(local_path))
            print(f"Downloaded to {local_path}")

            self._temp_files.append(local_path)
            return local_path

        except NoCredentialsError:
            raise RuntimeError("AWS credentials not found. Please configure AWS credentials.")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                raise FileNotFoundError(f"S3 object not found: {s3_uri}")
            elif error_code == '403':
                raise PermissionError(f"Access denied to S3 object: {s3_uri}")
            else:
                raise RuntimeError(f"S3 error ({error_code}): {e}")

    def _load_model(self, checkpoint_path):
        #Load PSPNet model from checkpoint
        if self._is_s3_path(checkpoint_path):
            checkpoint_path = self._download_from_s3(checkpoint_path)

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=4,
            classes=4,
            activation=None
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        dice_score = checkpoint.get('val_dice', 'N/A')
        if isinstance(dice_score, (int, float)):
            dice_score = f"{dice_score:.4f}"
        print(f"  Model Dice Score: {dice_score}")

        return model

    def _normalize_slice(self, slice_2d):
        #Normalize 2D slice using z-score normalization
        mean_val = np.mean(slice_2d)
        std_val = np.std(slice_2d)
        normalized = (slice_2d - mean_val) / (std_val + 1e-6)
        return np.nan_to_num(normalized)

    def _predict_slice(self, slice_2d):
        #Predict segmentation for a single 2D slice
        slice_tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)
        slice_tensor = slice_tensor.permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            output = self.model(slice_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        return pred

    def predict_volume(self, volume_data):
        #Predict segmentation for entire 3D volume
        volume_shape = volume_data.shape
        predictions = np.zeros(volume_shape, dtype=np.uint8)

        print(f"Predicting segmentation for volume shape: {volume_shape}")
        print("Note: Using single modality data for all 4 input channels")

        # Process axial slices (z-dimension)
        for z in tqdm(range(volume_shape[2]), desc="Processing slices"):
            # Normalize the slice
            slice_normalized = self._normalize_slice(volume_data[:, :, z])

            # Duplicate the same slice for all 4 modality channels
            multi_modal = np.stack([slice_normalized] * 4, axis=-1)

            predictions[:, :, z] = self._predict_slice(multi_modal)

        print("\nPrediction complete!")
        self._print_prediction_summary(predictions)

        return predictions

    def _print_prediction_summary(self, predictions):
        #Print summary statistics of predictions
        print(f"\nPrediction Summary:")
        print(f"  Shape: {predictions.shape}")
        print(f"  Classes found: {np.unique(predictions)}")

        class_names = {
            0: 'Background',
            1: 'NCR/NET (Necrotic)',
            2: 'Edema',
            3: 'Enhancing Tumor'
        }

        for class_id in np.unique(predictions):
            count = np.sum(predictions == class_id)
            percentage = 100 * count / predictions.size
            print(f"    Class {class_id} ({class_names.get(class_id, 'Unknown')}): "
                  f"{count:,} voxels ({percentage:.2f}%)")

    def save_prediction(self, prediction, reference_nifti_path, output_path):
        #Save prediction as NIfTI file using reference file's header/affine

        reference_img = nib.load(str(reference_nifti_path))
        pred_img = nib.Nifti1Image(prediction, reference_img.affine, reference_img.header)
        nib.save(pred_img, str(output_path))
        print(f"✅ Saved segmentation to: {output_path}")


def load_nifti_volume(file_path):
    #Load a single NIfTI file into memory

    img = nib.load(str(file_path))
    volume = np.array(img.get_fdata(), dtype=np.float32)
    print(f"✅ Loaded volume: {volume.shape}")
    return volume

def detect_modality_from_filename(filename):
    filename_lower = filename.lower()
    match = re.search(r'_(flair|t1ce|t1|t2)\.nii', filename_lower)
    if match:
        return match.group(1)
    return None

def find_sibling_files(file_path, detected_modality):
    file_path = Path(file_path)
    parent_dir = file_path.parent
    
    # Extract patient ID
    filename = file_path.name
    name_without_ext = filename.replace('.nii.gz', '').replace('.nii', '')
    patient_id = name_without_ext.rsplit('_', 1)[0]
    
    modalities = ['flair', 't1', 't1ce', 't2']
    sibling_paths = {}
    missing = []
    
    for modality in modalities:
        # Try both .nii and .nii.gz extensions
        for ext in ['.nii.gz', '.nii']:
            candidate = parent_dir / f"{patient_id}_{modality}{ext}"
            if candidate.exists():
                sibling_paths[modality] = str(candidate)
                break
        
        if modality not in sibling_paths:
            missing.append(f"{patient_id}_{modality}.nii[.gz]")
    
    if missing:
        raise FileNotFoundError(f"Missing required modality files: {', '.join(missing)}")
    
    return sibling_paths


def load_all_modalities(modality_paths):
    import numpy as np
    import nibabel as nib
    
    modality_order = ['flair', 't1', 't1ce', 't2']
    volumes = []
    
    for modality in modality_order:
        if modality not in modality_paths:
            raise ValueError(f"Missing modality: {modality}")
        
        img = nib.load(modality_paths[modality])
        volume = np.array(img.get_fdata(), dtype=np.float32)
        volumes.append(volume)
    
    multi_modal_volume = np.stack(volumes, axis=-1)
    
    print(f"✅ Loaded all modalities: {multi_modal_volume.shape}")
    
    return multi_modal_volume