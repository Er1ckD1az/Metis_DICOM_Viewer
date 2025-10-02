import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from segmentation_models_pytorch.metrics import get_stats, iou_score
import os
from DataPREP import (MultiViewSplit, create_patient_level_split, create_orientation_datasets_with_split)
from DataVisualization import visualize_predictions, plot_training_history, calculate_detailed_metrics, print_metrics_report
import config

class PSPNetTrainer:
    def __init__(
            self,
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=4,
            num_classes=4,
            lr=1e-4,
            device=None,
            save_dir="./checkpoints"
    ):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.num_classes = num_classes

        os.makedirs(save_dir, exist_ok=True)

        # Initialize model
        self.model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        self.model = self.model.to(self.device)

        # Loss function
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass')

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # Tracking
        self.best_val_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_dices = []

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model: PSPNet with {encoder_name} encoder")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def loss_fn(self, outputs, targets):
        return self.dice_loss(outputs, targets) + self.focal_loss(outputs, targets)

    def train_epoch(self, train_loader, epoch, num_epochs):
        self.model.train()
        train_loss = 0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            images, masks, orientations = batch_data

            images = images.to(self.device)
            masks = masks.to(self.device).long()
            images = images.permute(0, 3, 1, 2)  #data is formatted as [B, H, W, C] this fixes that

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / num_batches
        return avg_train_loss

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        num_batches = 0

        # Metrics tracking
        all_tp = []
        all_fp = []
        all_fn = []
        all_tn = []

        with torch.no_grad():
            for batch_data in val_loader:
                images, masks, orientations = batch_data

                images = images.to(self.device)
                masks = masks.to(self.device).long()
                images = images.permute(0, 3, 1, 2)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

                val_loss += loss.item()
                num_batches += 1

                preds = outputs.argmax(dim=1)
                tp, fp, fn, tn = get_stats(
                    preds,
                    masks,
                    mode='multiclass',
                    num_classes=self.num_classes
                )

                all_tp.append(tp)
                all_fp.append(fp)
                all_fn.append(fn)
                all_tn.append(tn)

        all_tp = torch.cat(all_tp, dim=0)
        all_fp = torch.cat(all_fp, dim=0)
        all_fn = torch.cat(all_fn, dim=0)
        all_tn = torch.cat(all_tn, dim=0)

        dice_micro = iou_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")
        dice_macro = iou_score(all_tp, all_fp, all_fn, all_tn, reduction="macro")
        dice_per_class = iou_score(all_tp, all_fp, all_fn, all_tn, reduction=None)

        avg_val_loss = val_loss / num_batches

        return avg_val_loss, dice_micro, dice_macro, dice_per_class

    def save_checkpoint(self, epoch, train_loss, val_loss, val_dice, model_name):
        checkpoint_path = os.path.join(self.save_dir, f"{model_name}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'best_val_dice': self.best_val_dice,
        }, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best Val Dice: {self.best_val_dice:.4f}")

    def train(
            self,
            train_loader,
            val_loader,
            num_epochs=100,
            model_name="best_model",
            save_every=10
    ):

        print("\n" + "=" * 60)
        print(f"Starting Training: {model_name}")
        print("=" * 60)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Epochs: {num_epochs}")
        print("=" * 60 + "\n")

        for epoch in range(num_epochs):
            avg_train_loss = self.train_epoch(train_loader, epoch, num_epochs)
            self.train_losses.append(avg_train_loss)

            avg_val_loss, dice_micro, dice_macro, dice_per_class = self.validate(val_loader)
            self.val_losses.append(avg_val_loss)
            self.val_dices.append(dice_micro.item())

            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs} Summary")
            print(f"{'=' * 60}")
            print(f"Train Loss:        {avg_train_loss:.4f}")
            print(f"Val Loss:          {avg_val_loss:.4f}")
            print(f"Val Dice (micro):  {dice_micro:.4f}")
            print(f"Val Dice (macro):  {dice_macro:.4f}")
            print(f"Val Dice per class:")
            print(f"  Background:      {dice_per_class[0]:.4f}")
            print(f"  NCR/NET:         {dice_per_class[1]:.4f}")
            print(f"  Edema:           {dice_per_class[2]:.4f}")
            print(f"  Enhancing:       {dice_per_class[3]:.4f}")
            print(f"Learning Rate:     {self.optimizer.param_groups[0]['lr']:.6f}")

            self.scheduler.step(dice_micro)

            # Save best model
            if dice_micro > self.best_val_dice:
                self.best_val_dice = dice_micro
                self.save_checkpoint(
                    epoch, avg_train_loss, avg_val_loss,
                    dice_micro.item(), f"{model_name}_best"
                )

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    epoch, avg_train_loss, avg_val_loss,
                    dice_micro.item(), f"{model_name}_epoch_{epoch + 1}"
                )

            print(f"{'=' * 60}\n")

        print("Training complete!")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")

        return self.best_val_dice


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config.print_config()

    print("Loading data...")
    full_train_ds = MultiViewSplit(
        data_dir=str(config.TRAIN_DATA_DIR),
        mode="train",
        file_extension=".nii"
    )

    # Create patient-level split
    train_indices, val_indices = create_patient_level_split(full_train_ds, val_split=0.2, seed=42)

    # Create orientation-specific datasets
    train_axial, val_axial = create_orientation_datasets_with_split(
        full_train_ds, 'axial', train_indices, val_indices
    )
    train_coronal, val_coronal = create_orientation_datasets_with_split(
        full_train_ds, 'coronal', train_indices, val_indices
    )
    train_sagittal, val_sagittal = create_orientation_datasets_with_split(
        full_train_ds, 'sagittal', train_indices, val_indices
    )

    batch_size = 10
    num_epochs = 3

    # Create DataLoaders for Axial view
    axial_train_loader = DataLoader(
        train_axial, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    axial_val_loader = DataLoader(
        val_axial, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Create DataLoaders for Coronal view
    coronal_train_loader = DataLoader(
        train_coronal, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    coronal_val_loader = DataLoader(
        val_coronal, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Create DataLoaders for Sagittal view
    sagittal_train_loader = DataLoader(
        train_sagittal, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    sagittal_val_loader = DataLoader(
        val_sagittal, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    #Train Axial
    print("\n" + "=" * 40)
    print("TRAINING AXIAL MODEL")
    print("=" * 40)

    axial_trainer = PSPNetTrainer(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=4,
        num_classes=4,
        lr=1e-4,
        save_dir="./checkpoints/axial"
    )

    axial_trainer.train(
        train_loader=axial_train_loader,
        val_loader=axial_val_loader,
        num_epochs=num_epochs,
        model_name="axial_model"
    )

    print("\n" + "=" * 40)
    print("TRAINING CORONAL MODEL")
    print("=" * 40)

    #Train Coronal
    coronal_trainer = PSPNetTrainer(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=4,
        num_classes=4,
        lr=1e-4,
        save_dir="./checkpoints/coronal"
    )

    coronal_trainer.train(
        train_loader=coronal_train_loader,
        val_loader=coronal_val_loader,
        num_epochs=num_epochs,
        model_name="coronal_model"
    )

    print("\n" + "=" * 40)
    print("TRAINING SAGITTAL MODEL")
    print("=" * 40)

    #Train Sagittal
    sagittal_trainer = PSPNetTrainer(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=4,
        num_classes=4,
        lr=1e-4,
        save_dir="./checkpoints/sagittal"
    )

    sagittal_trainer.train(
        train_loader=sagittal_train_loader,
        val_loader=sagittal_val_loader,
        num_epochs=num_epochs,
        model_name="sagittal_model"
    )

    print("\n" + "=" * 40)
    print("ALL MODELS TRAINED!")
    print("=" * 40)
    print(f"Axial model best Dice:     {axial_trainer.best_val_dice:.4f}")


    print(f"Coronal model best Dice:   {coronal_trainer.best_val_dice:.4f}")
    print(f"Sagittal model best Dice:  {sagittal_trainer.best_val_dice:.4f}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 40)
    print("Generating visualizations and metrics")
    print("=" * 40)

    os.makedirs('./results', exist_ok = True)


    # === Axial Results ===
    print("\n--- Axial Model Analysis ---")

    print("Plotting axial training history")
    plot_training_history(axial_trainer, save_path='./results/axial_training_history.png')

    print("visualizing axial predictions")
    visualize_predictions(
        axial_trainer.model,
        axial_train_loader,
        device,
        num_samples=10,
        save_dir='./results/axial_predictions'
    )

    print("Calculating axial metrics")
    axial_metrics = calculate_detailed_metrics(
        axial_trainer.model,
        axial_train_loader,
        device
    )
    print_metrics_report(axial_metrics)


    # === Coronal Results ===
    print("\n--- Coronal Model Analysis ---")

    print("Plotting Coronal training history")
    plot_training_history(coronal_trainer, save_path='./results/coronal_training_history.png')

    print("visualizing Coronal predictions")
    visualize_predictions(
        coronal_trainer.model,
        coronal_train_loader,
        device,
        num_samples=10,
        save_dir='./results/coronal_predictions'
    )

    print("Calculating Coronal metrics")
    coronal_metrics = calculate_detailed_metrics(
        coronal_trainer.model,
        coronal_train_loader,
        device
    )
    print_metrics_report(coronal_metrics)


    # === Sagittal Results ===
    print("\n--- Sagittal Model Analysis ---")

    print("Plotting Sagittal training history")
    plot_training_history(sagittal_trainer, save_path='./results/sagittal_training_history.png')

    print("visualizing Sagittal predictions")
    visualize_predictions(
        sagittal_trainer.model,
        sagittal_train_loader,
        device,
        num_samples=10,
        save_dir='./results/sagittal_predictions'
    )

    print("Calculating Sagittal metrics")
    sagittal_metrics = calculate_detailed_metrics(
        sagittal_trainer.model,
        sagittal_train_loader,
        device
    )
    print_metrics_report(sagittal_metrics)


    print("\n" + "=" * 40)
    print("Final Comparative Summary")
    print("=" * 40)

    print("\nDice Scores (Mirco):")
    print(f" Axial: {axial_metrics['dice_micro']:.4f}")
    print(f" Coronal: {coronal_metrics['dice_micro']:.4f}")
    print(f" Sagittal: {sagittal_metrics['dice_micro']:.4f}")

    print("\nPer-Class Dice Scores:")
    class_names = ['Background', 'NCR/NET', 'Edema', 'Enhancing']
    for i, name in enumerate(class_names):
        print(f"\n{name}:")
        print(f" Axial: {axial_metrics['dice_per_class']:.4f}")
        print(f" Coronal: {coronal_metrics['dice_per_class']:.4f}")
        print(f" Sagittal: {sagittal_metrics['dice_per_class']:.4f}")

    print("\n" + "=" * 40)
    print("All results saved to ./results/")
    print("Checkpoints saved to ./checkpoints/")
    print("=" * 40)

