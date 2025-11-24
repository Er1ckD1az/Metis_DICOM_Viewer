import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
from pathlib import Path
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import argparse
import os
import time
import warnings

warnings.filterwarnings('ignore')


class MultiViewSegmentationModel(nn.Module):
    def __init__(self, axial_checkpoint, sagittal_checkpoint, coronal_checkpoint,
                 num_classes=4, fusion_method='learned', use_refinement=False, device='cuda'):
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.use_refinement = use_refinement

        print("Loading view-specific models...")
        self.axial_model = self._load_view_model(axial_checkpoint, 'Axial')
        self.sagittal_model = self._load_view_model(sagittal_checkpoint, 'Sagittal')
        self.coronal_model = self._load_view_model(coronal_checkpoint, 'Coronal')

        #Freeze base models during meta-learning
        self._freeze_base_models()

        #Meta-learning components
        if fusion_method == 'learned':
            #Learnable attention weights for each view
            self.view_attention = nn.Sequential(
                nn.Linear(num_classes * 3, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3),
                nn.Softmax(dim=-1)
            )
        elif fusion_method == 'weighted':
            #Simple learnable weights
            self.view_weights = nn.Parameter(torch.ones(3) / 3)

        #Optional refinement network
        if use_refinement:
            print("  ✓ Refinement network enabled")
            self.refinement = nn.Sequential(
                nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, num_classes, kernel_size=1)
            )
        else:
            self.refinement = None

        self.to(device)

    def _load_view_model(self, checkpoint_path, view_name):
        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=4,
            classes=self.num_classes,
            activation=None
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        dice = checkpoint.get('val_dice', 'N/A')
        print(f"  {view_name} model loaded - Dice: {dice}")

        return model

    def _freeze_base_models(self):
        for model in [self.axial_model, self.sagittal_model, self.coronal_model]:
            for param in model.parameters():
                param.requires_grad = False

    def _unfreeze_base_models(self):
        for model in [self.axial_model, self.sagittal_model, self.coronal_model]:
            for param in model.parameters():
                param.requires_grad = True

    def _predict_view(self, volume, model, view_axis):
        H, W, D = volume.shape[:3]
        num_slices = volume.shape[view_axis]

        #Initialize prediction volume
        if view_axis == 0:  # Axial
            pred_shape = (H, W, D, self.num_classes)
        elif view_axis == 1:  # Sagittal
            pred_shape = (H, W, D, self.num_classes)
        else:  #Coronal
            pred_shape = (H, W, D, self.num_classes)

        predictions = np.zeros(pred_shape, dtype=np.float32)

        with torch.no_grad():
            for i in range(num_slices):
                #Extract slice
                if view_axis == 0:
                    slice_4d = volume[:, :, i, :]
                elif view_axis == 1:
                    slice_4d = volume[i, :, :, :]
                else:
                    slice_4d = volume[:, i, :, :]

                #Normalize
                normalized = np.zeros_like(slice_4d)
                for ch in range(4):
                    channel = slice_4d[:, :, ch]
                    mean, std = channel.mean(), channel.std()
                    normalized[:, :, ch] = (channel - mean) / (std + 1e-6)
                    normalized[:, :, ch] = np.nan_to_num(normalized[:, :, ch])

                #Predict
                tensor = torch.from_numpy(normalized).float()
                tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

                output = model(tensor)
                probs = torch.softmax(output, dim=1).squeeze(0)
                probs = probs.cpu().numpy().transpose(1, 2, 0)

                #Place back in volume
                if view_axis == 0:  # Axial
                    predictions[:, :, i, :] = probs
                elif view_axis == 1:  # Sagittal
                    predictions[i, :, :, :] = probs
                else:  #Coronal
                    predictions[:, i, :, :] = probs

        return predictions

    def predict_volume(self, volume_data):
        print("Predicting from all three views...")

        #Get predictions from each view
        axial_probs = self._predict_view(volume_data, self.axial_model, view_axis=2)
        sagittal_probs = self._predict_view(volume_data, self.sagittal_model, view_axis=0)
        coronal_probs = self._predict_view(volume_data, self.coronal_model, view_axis=1)

        #Fuse predictions
        if self.fusion_method == 'average':
            fused_probs = (axial_probs + sagittal_probs + coronal_probs) / 3

        elif self.fusion_method == 'weighted':
            weights = torch.softmax(self.view_weights, dim=0).cpu().numpy()
            fused_probs = (weights[0] * axial_probs +
                           weights[1] * sagittal_probs +
                           weights[2] * coronal_probs)

        elif self.fusion_method == 'learned':
            #Stack all predictions
            all_probs = np.stack([axial_probs, sagittal_probs, coronal_probs], axis=-1)
            H, W, D = all_probs.shape[:3]

            # Reshape for attention mechanism
            flat_probs = all_probs.reshape(-1, self.num_classes * 3)
            flat_tensor = torch.from_numpy(flat_probs).float().to(self.device)

            #Calculate attention weights
            with torch.no_grad():
                attention = self.view_attention(flat_tensor)  # (N, 3)

            attention = attention.cpu().numpy().reshape(H, W, D, 3)

            #Apply attention
            fused_probs = (attention[..., 0:1] * axial_probs +
                           attention[..., 1:2] * sagittal_probs +
                           attention[..., 2:3] * coronal_probs)

        #Apply optional refinement network (slice by slice)
        if self.use_refinement and self.refinement is not None:
            print("Applying refinement network to predictions...")
            refined_probs = np.zeros_like(fused_probs)

            with torch.no_grad():
                for z in tqdm(range(fused_probs.shape[2]), desc="Refining slices"):
                    #Get slice: (H, W, num_classes)
                    slice_probs = fused_probs[:, :, z, :]

                    #Convert to tensor: (1, num_classes, H, W)
                    slice_tensor = torch.from_numpy(slice_probs).float()
                    slice_tensor = slice_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

                    #Apply refinement
                    refined_slice = self.refinement(slice_tensor)
                    refined_slice = torch.softmax(refined_slice, dim=1)

                    #Convert back: (H, W, num_classes)
                    refined_slice = refined_slice.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    refined_probs[:, :, z, :] = refined_slice

            fused_probs = refined_probs

        #Convert to final segmentation
        final_prediction = np.argmax(fused_probs, axis=-1).astype(np.uint8)

        return final_prediction

    def forward(self, axial_input, sagittal_input, coronal_input):
        #Get predictions from each view
        with torch.no_grad():
            axial_out = self.axial_model(axial_input)
            sagittal_out = self.sagittal_model(sagittal_input)
            coronal_out = self.coronal_model(coronal_input)

        #Apply softmax to get probabilities
        axial_probs = torch.softmax(axial_out, dim=1)
        sagittal_probs = torch.softmax(sagittal_out, dim=1)
        coronal_probs = torch.softmax(coronal_out, dim=1)

        #Get target spatial dimensions from axial view (most common reference)
        target_size = axial_probs.shape[2:]  # (H, W)

        #Resize sagittal and coronal to match axial dimensions
        if sagittal_probs.shape[2:] != target_size:
            sagittal_probs = torch.nn.functional.interpolate(
                sagittal_probs, size=target_size, mode='bilinear', align_corners=False
            )

        if coronal_probs.shape[2:] != target_size:
            coronal_probs = torch.nn.functional.interpolate(
                coronal_probs, size=target_size, mode='bilinear', align_corners=False
            )

        if self.fusion_method == 'average':
            fused = (axial_probs + sagittal_probs + coronal_probs) / 3

        elif self.fusion_method == 'weighted':
            weights = torch.softmax(self.view_weights, dim=0)
            fused = (weights[0] * axial_probs +
                     weights[1] * sagittal_probs +
                     weights[2] * coronal_probs)

        elif self.fusion_method == 'learned':
            B, C, H, W = axial_probs.shape

            #Concatenate all view predictions
            all_probs = torch.cat([axial_probs, sagittal_probs, coronal_probs], dim=1)
            flat = all_probs.permute(0, 2, 3, 1).reshape(B * H * W, -1)

            attention = self.view_attention(flat).reshape(B, H, W, 3)
            attention = attention.permute(0, 3, 1, 2)  # (B, 3, H, W)

            fused = (attention[:, 0:1] * axial_probs +
                     attention[:, 1:2] * sagittal_probs +
                     attention[:, 2:3] * coronal_probs)

        if self.use_refinement and self.refinement is not None:
            fused = self.refinement(fused)

        return fused


class BraTSMetaDataset(Dataset):
    def __init__(self, data_dir, patient_ids, file_extension=".nii"):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.file_extension = file_extension
        self.modalities = ["flair", "t1", "t1ce", "t2"]
        self.samples = self._prepare_samples()
        self._create_slice_index()

    def _prepare_samples(self):
        samples = []
        for pid in self.patient_ids:
            patient_path = self.data_dir / pid
            if not patient_path.exists():
                continue

            modality_paths = {}
            all_exist = True
            for mod in self.modalities:
                mod_path = patient_path / f"{pid}_{mod}{self.file_extension}"
                if mod_path.exists():
                    modality_paths[mod] = mod_path
                else:
                    all_exist = False
                    break

            seg_path = patient_path / f"{pid}_seg{self.file_extension}"

            if all_exist and seg_path.exists():
                samples.append({
                    'patient_id': pid,
                    'patient_path': patient_path,
                    'modalities': modality_paths,
                    'segmentation': seg_path
                })

        return samples

    def _create_slice_index(self):
        self.slice_indices = []

        for sample_idx, sample in enumerate(self.samples):
            img = nib.load(sample['modalities']['flair'])
            volume_shape = img.shape

            x_start, x_end = int(volume_shape[0] * 0.1), int(volume_shape[0] * 0.9)
            y_start, y_end = int(volume_shape[1] * 0.1), int(volume_shape[1] * 0.9)
            z_start, z_end = int(volume_shape[2] * 0.1), int(volume_shape[2] * 0.9)

            for x in range(x_start, x_end, 5):  #Sample every 5 slices
                for y in range(y_start, y_end, 5):
                    for z in range(z_start, z_end, 5):
                        self.slice_indices.append((sample_idx, x, y, z))

        print(f"Created {len(self.slice_indices)} multi-view samples from {len(self.samples)} patients")

    def _normalize_slice(self, slice_2d):
        mean, std = slice_2d.mean(), slice_2d.std()
        normalized = (slice_2d - mean) / (std + 1e-6)
        return np.nan_to_num(normalized)

    def _pad_to_divisible(self, slice_4d, divisor=8):
        H, W = slice_4d.shape[:2]

        #Calculate padding needed
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor

        #Pad bottom and right
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(slice_4d,
                            ((0, pad_h), (0, pad_w), (0, 0)),
                            mode='constant',
                            constant_values=0)
            return padded, (pad_h, pad_w)

        return slice_4d, (0, 0)

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        sample_idx, x, y, z = self.slice_indices[idx]
        sample = self.samples[sample_idx]

        volumes = []
        for mod in self.modalities:
            img = nib.load(sample['modalities'][mod])
            vol = np.array(img.get_fdata(), dtype=np.float32)
            volumes.append(vol)

        volume = np.stack(volumes, axis=-1)

        seg_img = nib.load(sample['segmentation'])
        seg = np.array(seg_img.get_fdata(), dtype=np.int64)
        seg[seg == 4] = 3  # Map label 4 to 3

        sagittal_slice = volume[x, :, :, :]
        sagittal_seg = seg[x, :, :]

        coronal_slice = volume[:, y, :, :]
        coronal_seg = seg[:, y, :]

        axial_slice = volume[:, :, z, :]
        axial_seg = seg[:, :, z]

        #Pad slices to ensure dimensions are divisible by 8
        axial_slice, axial_pad = self._pad_to_divisible(axial_slice)
        sagittal_slice, sagittal_pad = self._pad_to_divisible(sagittal_slice)
        coronal_slice, coronal_pad = self._pad_to_divisible(coronal_slice)

        # Pad segmentation masks too
        if axial_pad[0] > 0 or axial_pad[1] > 0:
            axial_seg = np.pad(axial_seg, ((0, axial_pad[0]), (0, axial_pad[1])),
                               mode='constant', constant_values=0)
        if sagittal_pad[0] > 0 or sagittal_pad[1] > 0:
            sagittal_seg = np.pad(sagittal_seg, ((0, sagittal_pad[0]), (0, sagittal_pad[1])),
                                  mode='constant', constant_values=0)
        if coronal_pad[0] > 0 or coronal_pad[1] > 0:
            coronal_seg = np.pad(coronal_seg, ((0, coronal_pad[0]), (0, coronal_pad[1])),
                                 mode='constant', constant_values=0)

        #Normalize each view independently
        def normalize_and_convert(slice_4d):
            normalized = np.zeros_like(slice_4d)
            for ch in range(4):
                normalized[:, :, ch] = self._normalize_slice(slice_4d[:, :, ch])
            return torch.from_numpy(normalized).float().permute(2, 0, 1)

        axial_tensor = normalize_and_convert(axial_slice)
        sagittal_tensor = normalize_and_convert(sagittal_slice)
        coronal_tensor = normalize_and_convert(coronal_slice)

        axial_seg_tensor = torch.from_numpy(axial_seg).long()

        return {
            'axial': axial_tensor,
            'sagittal': sagittal_tensor,
            'coronal': coronal_tensor,
            'segmentation': axial_seg_tensor,
            'patient_id': sample['patient_id'],
            'position': (x, y, z)
        }


def dice_coefficient(pred, target, num_classes=4):
    dice_scores = []

    for c in range(1, num_classes):  #Skip background
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())

    return np.mean(dice_scores)


def train_meta_learner(model, train_loader, val_loader, num_epochs=1,
                       learning_rate=1e-4, save_dir='./meta_model',
                       early_stopping_patience=5):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     patience=3, factor=0.5)

    best_dice = 0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_dice': [],
        'epoch_times': [],
        'total_time': 0
    }
    epochs_without_improvement = 0

    #Start total training timer
    training_start_time = time.perf_counter()

    for epoch in range(num_epochs):
        #Start epoch timer
        epoch_start_time = time.perf_counter()

        model.train()
        train_loss = 0
        train_dice = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            axial_inputs = batch['axial'].to(model.device)
            sagittal_inputs = batch['sagittal'].to(model.device)
            coronal_inputs = batch['coronal'].to(model.device)
            targets = batch['segmentation'].to(model.device)

            outputs = model(axial_inputs, sagittal_inputs, coronal_inputs)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            train_loss += loss.item()

            pred = outputs.argmax(dim=1)
            dice = dice_coefficient(pred, targets)
            train_dice += dice
            num_batches += 1

            pbar.set_postfix({'loss': loss.item(), 'dice': dice})

        train_loss /= num_batches
        train_dice /= num_batches

        model.eval()
        val_dice = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                axial_inputs = batch['axial'].to(model.device)
                sagittal_inputs = batch['sagittal'].to(model.device)
                coronal_inputs = batch['coronal'].to(model.device)
                targets = batch['segmentation'].to(model.device)

                outputs = model(axial_inputs, sagittal_inputs, coronal_inputs)
                pred = outputs.argmax(dim=1)
                val_dice += dice_coefficient(pred, targets)
                val_batches += 1

        val_dice /= val_batches

        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['epoch_times'].append(epoch_duration)

        print(f'Epoch {epoch + 1}: Loss={train_loss:.4f}, '
              f'Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}, '
              f'Time={epoch_duration:.2f}s')

        #Early stopping check
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'fusion_method': model.fusion_method,
                'use_refinement': model.use_refinement,
                'training_time': time.perf_counter() - training_start_time
            }
            torch.save(checkpoint, save_dir / 'best_meta_model.pth')
            print(f'Saved best model with Dice={best_dice:.4f}')
        else:
            epochs_without_improvement += 1
            print(f'No improvement for {epochs_without_improvement} epochs')

            if epochs_without_improvement >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

        scheduler.step(val_dice)

    #End total training timer
    training_end_time = time.perf_counter()
    total_training_time = training_end_time - training_start_time
    history['total_time'] = total_training_time

    #Save training history with timing info
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


    print("\n" + "-" * 45)
    print("TRAINING TIME SUMMARY")
    print("-" * 45)
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time / 60:.2f} minutes)")
    print(f"Average time per epoch: {np.mean(history['epoch_times']):.2f}s")
    print(f"Fastest epoch: {np.min(history['epoch_times']):.2f}s")
    print(f"Slowest epoch: {np.max(history['epoch_times']):.2f}s")
    print("-" * 45)

    return history


def main():
    parser = argparse.ArgumentParser(description='Meta-learning for multi-view MRI segmentation')
    parser.add_argument('--axial_model', type=str, required=True,
                        help='Path to axial view model checkpoint')
    parser.add_argument('--sagittal_model', type=str, required=True,
                        help='Path to sagittal view model checkpoint')
    parser.add_argument('--coronal_model', type=str, required=True,
                        help='Path to coronal view model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--fusion_method', type=str, default='learned',
                        choices=['average', 'weighted', 'learned'],
                        help='Fusion method for combining views')
    parser.add_argument('--use_refinement', action='store_true',
                        help='Enable refinement network after fusion (adds trainable parameters)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 15 for meta-learning)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--file_extension', type=str, default='.nii',
                        help='File extension for NIfTI files (.nii or .nii.gz)')
    parser.add_argument('--save_dir', type=str, default='./meta_model',
                        help='Directory to save trained model')

    args = parser.parse_args()

    # Start overall timer
    overall_start_time = time.perf_counter()

    print("\n" + "-" * 45)
    print("INITIALIZING MULTI-VIEW META-LEARNING MODEL")
    print("-" * 45)
    print(f"Axial model: {args.axial_model}")
    print(f"Sagittal model: {args.sagittal_model}")
    print(f"Coronal model: {args.coronal_model}")
    print(f"Fusion method: {args.fusion_method}")
    print(f"Refinement network: {'ENABLED' if args.use_refinement else 'DISABLED'}")
    print("-" * 45)

    model_init_start = time.perf_counter()
    model = MultiViewSegmentationModel(
        axial_checkpoint=args.axial_model,
        sagittal_checkpoint=args.sagittal_model,
        coronal_checkpoint=args.coronal_model,
        fusion_method=args.fusion_method,
        use_refinement=args.use_refinement
    )
    model_init_time = time.perf_counter() - model_init_start

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model initialization time: {model_init_time:.2f}s")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Prepare datasets
    print("\n" + "-" * 45)
    print("PREPARING DATASETS")
    print("-" * 45)

    data_prep_start = time.perf_counter()
    data_dir = Path(args.data_dir)

    all_patients = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(data_dir / d) and d.startswith("BraTS")
    ])

    if not all_patients:
        raise ValueError(f"No patient directories found in {data_dir}. "
                         f"Expected directories starting with 'BraTS'")

    np.random.shuffle(all_patients)
    split_idx = int(0.8 * len(all_patients))
    train_ids = all_patients[:split_idx]
    val_ids = all_patients[split_idx:]

    print(f"Training patients: {len(train_ids)}")
    print(f"Validation patients: {len(val_ids)}")

    train_dataset = BraTSMetaDataset(data_dir, train_ids, file_extension=args.file_extension)
    val_dataset = BraTSMetaDataset(data_dir, val_ids, file_extension=args.file_extension)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    data_prep_time = time.perf_counter() - data_prep_start
    print(f"Data preparation time: {data_prep_time:.2f}s")

    # Train
    print("\n" + "-" * 45)
    print("TRAINING META-LEARNER")
    print("-" * 45)
    print(f"Using conservative training: {args.epochs} max epochs with early stopping")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("-" * 45)

    history = train_meta_learner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping
    )

    # End overall timer
    overall_end_time = time.perf_counter()
    total_time = overall_end_time - overall_start_time

    print("\n" + "-" * 45)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("-" * 45)
    print(f"Best validation Dice: {max(history['val_dice']):.4f}")
    print(f"Model saved to: {args.save_dir}")
    print("\n--- TIME BREAKDOWN ---")
    print(f"Model initialization: {model_init_time:.2f}s")
    print(f"Data preparation: {data_prep_time:.2f}s")
    print(f"Training: {history['total_time']:.2f}s ({history['total_time'] / 60:.2f} min)")
    print(f"Total execution time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("-" * 45)

if __name__ == '__main__':
    main()