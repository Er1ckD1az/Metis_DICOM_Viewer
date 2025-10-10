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

try:
    from scipy import stats
except ImportError:
    stats = None


class MultiViewEnsemble:
    def __init__(
            self,
            axial_checkpoint_path,
            coronal_checkpoint_path,
            sagittal_checkpoint_path,
            device=None
    ):

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._temp_files = []  # Track temporary files for cleanup

        print("Loading models...")

        try:
            self.axial_model = self._load_model(axial_checkpoint_path, "Axial")
            self.coronal_model = self._load_model(coronal_checkpoint_path, "Coronal")
            self.sagittal_model = self._load_model(sagittal_checkpoint_path, "Sagittal")
            print(f"All models loaded on {self.device}")
        except Exception as e:
            self._cleanup_temp_files()
            raise RuntimeError(f"Failed to load models: {e}")

    def __del__(self):
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass


    def _is_s3_path(self, path):
        return str(path).startswith("s3://")

    def _parse_s3_path(self, s3_uri):
        match = re.match(r"s3://([^/]+)/(.+)", s3_uri)
        if not match:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        return match.group(1), match.group(2)

    def _download_from_s3(self, s3_uri):
        if boto3 is None:
            raise ImportError(
                "boto3 is required for loading from S3. "
                "Install with: pip install boto3"
            )

        try:
            bucket, key = self._parse_s3_path(s3_uri)
            tmp_dir = Path(tempfile.gettempdir()) / "brats_models"
            tmp_dir.mkdir(exist_ok=True)
            local_path = tmp_dir / Path(key).name

            if local_path.exists():
                print(f"  ℹ️  Using cached model: {local_path}")
                return local_path

            print(f"⬇️  Downloading {key} from bucket {bucket}...")
            s3 = boto3.client("s3")
            s3.download_file(bucket, key, str(local_path))
            print(f"  ✅ Downloaded to {local_path}")

            self._temp_files.append(local_path)
            return local_path

        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. Please configure AWS credentials:\n"
                "  - Run 'aws configure'\n"
                "  - Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                raise FileNotFoundError(f"S3 object not found: {s3_uri}")
            elif error_code == '403':
                raise PermissionError(f"Access denied to S3 object: {s3_uri}")
            else:
                raise RuntimeError(f"S3 error ({error_code}): {e}")

    def _load_model(self, checkpoint_path, model_name):
        """Load a single model from checkpoint (supports local + S3)"""
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
        print(f"  ✅ {model_name} model loaded (Dice: {dice_score})")

        return model

    def _normalize_slice(self, slice_2d):
        mean_val = np.mean(slice_2d)
        std_val = np.std(slice_2d)
        normalized = (slice_2d - mean_val) / (std_val + 1e-6)
        return np.nan_to_num(normalized)

    def _pad_to_160(self, array):
        if array.shape[1] >= 160:
            return array
        pad_total = 160 - array.shape[1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if array.ndim == 2:
            return np.pad(array, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        else:
            return np.pad(array, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    def _unpad_from_160(self, array, original_width=155):
        if array.shape[1] <= original_width:
            return array
        pad_total = array.shape[1] - original_width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return array[:, pad_left:-pad_right if pad_right > 0 else None]


    def _predict_slice(self, slice_2d, model):
        slice_tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)
        slice_tensor = slice_tensor.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            output = model(slice_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        return pred

    def _majority_voting(self, axial_pred, coronal_pred, sagittal_pred):
        if stats is None:
            raise ImportError(
                "scipy is required for ensemble voting. "
                "Install with: pip install scipy"
            )

        stacked = np.stack([axial_pred, coronal_pred, sagittal_pred], axis=0)
        final_pred, _ = stats.mode(stacked, axis=0, keepdims=False)
        return final_pred.astype(np.uint8)

    def predict_volume(self, volume_dict, orientation='axial'):
        """
        Predicts the segmentation using only one orientation (default: axial)
        to reduce runtime.
        """
        volume_shape = volume_dict['flair'].shape
        prediction = np.zeros(volume_shape, dtype=np.uint8)

        print(f"Predicting {orientation} slices only...")

        if orientation == 'axial':
            for z in tqdm(range(volume_shape[2]), desc="Axial"):
                slice_data = [self._normalize_slice(volume_dict[m][:, :, z])
                              for m in ['flair', 't1', 't1ce', 't2']]
                multi_modal = np.stack(slice_data, axis=-1)
                prediction[:, :, z] = self._predict_slice(multi_modal, self.axial_model)

        elif orientation == 'coronal':
            for y in tqdm(range(volume_shape[1]), desc="Coronal"):
                slice_data = [self._normalize_slice(volume_dict[m][:, y, :])
                              for m in ['flair', 't1', 't1ce', 't2']]
                multi_modal = np.stack(slice_data, axis=-1)
                multi_modal = self._pad_to_160(multi_modal)
                pred = self._predict_slice(multi_modal, self.coronal_model)
                prediction[:, y, :] = self._unpad_from_160(pred)

        elif orientation == 'sagittal':
            for x in tqdm(range(volume_shape[0]), desc="Sagittal"):
                slice_data = [self._normalize_slice(volume_dict[m][x, :, :])
                              for m in ['flair', 't1', 't1ce', 't2']]
                multi_modal = np.stack(slice_data, axis=-1)
                multi_modal = self._pad_to_160(multi_modal)
                pred = self._predict_slice(multi_modal, self.sagittal_model)
                prediction[x, :, :] = self._unpad_from_160(pred)

        else:
            raise ValueError(f"Unknown orientation '{orientation}'. Choose 'axial', 'coronal', or 'sagittal'.")

        print(f"\nPrediction complete using {orientation} model.")
        return prediction

    def predict_patient(self, patient_dir, save_path=None):
        patient_dir = Path(patient_dir)
        patient_id = patient_dir.name

        print(f"\n{'=' * 60}")
        print(f"Processing patient: {patient_id}")
        print(f"{'=' * 60}")

        volume_dict = {}
        for modality in ['flair', 't1', 't1ce', 't2']:
            # Try both .nii and .nii.gz extensions
            file_candidates = list(patient_dir.glob(f"{patient_id}_{modality}.nii*"))

            if not file_candidates:
                raise FileNotFoundError(
                    f"Could not find {modality} file for patient {patient_id}\n"
                    f"Expected: {patient_dir}/{patient_id}_{modality}.nii or .nii.gz"
                )

            file_path = file_candidates[0]
            img = nib.load(str(file_path))
            volume_dict[modality] = np.array(img.get_fdata(), dtype=np.float32)
            print(f"  ✅ Loaded {modality}: {volume_dict[modality].shape}")

        prediction = self.predict_volume(volume_dict)

        # Save if requested
        if save_path:
            flair_files = list(patient_dir.glob(f"{patient_id}_flair.nii*"))
            if not flair_files:
                raise FileNotFoundError(f"Could not find FLAIR file for saving header info")

            original_img = nib.load(str(flair_files[0]))
            pred_img = nib.Nifti1Image(prediction, original_img.affine, original_img.header)
            nib.save(pred_img, str(save_path))
            print(f"\n✅ Saved prediction to: {save_path}")

        print(f"\n Prediction Summary:")
        print(f"  Shape: {prediction.shape}")
        print(f"  Classes found: {np.unique(prediction)}")
        for class_id in np.unique(prediction):
            count = np.sum(prediction == class_id)
            percentage = 100 * count / prediction.size
            class_names = {0: 'Background', 1: 'NCR/NET', 2: 'Edema', 3: 'Enhancing'}
            print(f"    Class {class_id} ({class_names.get(class_id, 'Unknown')}): "
                  f"{count:,} voxels ({percentage:.2f}%)")

        return prediction
