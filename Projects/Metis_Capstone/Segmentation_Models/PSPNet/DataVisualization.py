import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os


def visualize_predictions(model, dataloader, device, num_samples=5, save_dir=None):
    model.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch_data in enumerate(dataloader):
            if idx >= num_samples:
                break

            images, masks, orientations = batch_data
            images = images.to(device)
            masks = masks.to(device).long()
            images_input = images.permute(0, 3, 1, 2)

            # Get predictions
            outputs = model(images_input)
            preds = outputs.argmax(dim=1)  # [B, H, W]

            # Visualize first image in batch
            img = images[0].cpu().numpy()  # [H, W, 4]
            mask_gt = masks[0].cpu().numpy()  # [H, W]
            mask_pred = preds[0].cpu().numpy()  # [H, W]
            orientation = orientations[0]

            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'Sample {idx + 1} - {orientation.capitalize()} View', fontsize=16)

            # Show all 4 MRI modalities
            modality_names = ['FLAIR', 'T1', 'T1ce', 'T2']
            for i in range(4):
                axes[0, i].imshow(img[:, :, i], cmap='gray')
                axes[0, i].set_title(f'{modality_names[i]}')
                axes[0, i].axis('off')

            # Show ground truth segmentation
            axes[1, 0].imshow(mask_gt, cmap='tab10', vmin=0, vmax=3)
            axes[1, 0].set_title('Ground Truth')
            axes[1, 0].axis('off')

            # Show prediction
            axes[1, 1].imshow(mask_pred, cmap='tab10', vmin=0, vmax=3)
            axes[1, 1].set_title('Prediction')
            axes[1, 1].axis('off')

            # Show overlay on FLAIR
            axes[1, 2].imshow(img[:, :, 0], cmap='gray')
            axes[1, 2].imshow(mask_gt, cmap='tab10', alpha=0.4, vmin=0, vmax=3)
            axes[1, 2].set_title('GT Overlay')
            axes[1, 2].axis('off')

            # Show error map
            error_map = (mask_gt != mask_pred).astype(float)
            axes[1, 3].imshow(error_map, cmap='Reds')
            axes[1, 3].set_title(f'Errors (red)\nAccuracy: {100 * (1 - error_map.mean()):.1f}%')
            axes[1, 3].axis('off')

            plt.tight_layout()

            if save_dir:
                plt.savefig(os.path.join(save_dir, f'prediction_{idx + 1}.png'), dpi=150, bbox_inches='tight')
                print(f"Saved visualization to {save_dir}/prediction_{idx + 1}.png")

            plt.show()
            plt.close()


def visualize_single_prediction(model, dataset, idx, device, save_path=None):
    model.eval()

    # Get single sample
    image, mask_gt, orientation = dataset[idx]

    # Prepare for model
    image_input = image.unsqueeze(0).to(device)  # [1, H, W, C]
    image_input = image_input.permute(0, 3, 1, 2)  # [1, C, H, W]

    with torch.no_grad():
        output = model(image_input)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    image = image.cpu().numpy()
    mask_gt = mask_gt.cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'{orientation.capitalize()} View - Slice Index {idx}', fontsize=16)

    modality_names = ['FLAIR', 'T1', 'T1ce', 'T2']
    for i in range(4):
        axes[0, i].imshow(image[:, :, i], cmap='gray')
        axes[0, i].set_title(f'{modality_names[i]}')
        axes[0, i].axis('off')

    # Segmentation results
    axes[1, 0].imshow(mask_gt, cmap='tab10', vmin=0, vmax=3)
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred, cmap='tab10', vmin=0, vmax=3)
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')

    # Overlay
    axes[1, 2].imshow(image[:, :, 0], cmap='gray')
    axes[1, 2].imshow(pred, cmap='tab10', alpha=0.4, vmin=0, vmax=3)
    axes[1, 2].set_title('Prediction Overlay')
    axes[1, 2].axis('off')

    # Per-class comparison
    class_names = ['Background', 'NCR/NET', 'Edema', 'Enhancing']
    class_colors = ['black', 'red', 'green', 'blue']

    # Show which classes are present
    gt_classes = np.unique(mask_gt)
    pred_classes = np.unique(pred)

    info_text = "Classes present:\n"
    info_text += f"GT: {[class_names[c] for c in gt_classes]}\n"
    info_text += f"Pred: {[class_names[c] for c in pred_classes]}\n"
    info_text += f"\nAccuracy: {100 * np.mean(pred == mask_gt):.2f}%"

    axes[1, 3].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    axes[1, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    plt.close()


def plot_training_history(trainer, save_path=None):
    epochs = range(1, len(trainer.train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, trainer.train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, trainer.val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Dice score plot
    ax2.plot(epochs, trainer.val_dices, 'g-', label='Val Dice Score', linewidth=2)
    ax2.axhline(y=trainer.best_val_dice, color='r', linestyle='--',
                label=f'Best: {trainer.best_val_dice:.4f}', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice Score', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")

    plt.show()
    plt.close()


def calculate_detailed_metrics(model, dataloader, device, num_classes=4):
    from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, accuracy, precision, recall

    model.eval()

    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    print("Calculating detailed metrics...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            images, masks, orientations = batch_data
            images = images.to(device)
            masks = masks.to(device).long()
            images = images.permute(0, 3, 1, 2)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            tp, fp, fn, tn = get_stats(preds, masks, mode='multiclass', num_classes=num_classes)
            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            all_tn.append(tn)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    all_tp = torch.cat(all_tp, dim=0)
    all_fp = torch.cat(all_fp, dim=0)
    all_fn = torch.cat(all_fn, dim=0)
    all_tn = torch.cat(all_tn, dim=0)

    metrics = {
        'dice_per_class': iou_score(all_tp, all_fp, all_fn, all_tn, reduction=None),
        'dice_micro': iou_score(all_tp, all_fp, all_fn, all_tn, reduction="micro"),
        'dice_macro': iou_score(all_tp, all_fp, all_fn, all_tn, reduction="macro"),
        'f1_per_class': f1_score(all_tp, all_fp, all_fn, all_tn, reduction=None),
        'precision_per_class': precision(all_tp, all_fp, all_fn, all_tn, reduction=None),
        'recall_per_class': recall(all_tp, all_fp, all_fn, all_tn, reduction=None),
        'accuracy': accuracy(all_tp, all_fp, all_fn, all_tn, reduction="micro"),
    }

    return metrics


def print_metrics_report(metrics, class_names=['Background', 'NCR/NET', 'Edema', 'Enhancing']):
    print("\n" + "=" * 70)
    print("DETAILED METRICS REPORT")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Dice Score (micro):  {metrics['dice_micro']:.4f}")
    print(f"  Dice Score (macro):  {metrics['dice_macro']:.4f}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Dice':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 70)

    for i, name in enumerate(class_names):
        dice = metrics['dice_per_class'][i].item()
        f1 = metrics['f1_per_class'][i].item()
        prec = metrics['precision_per_class'][i].item()
        rec = metrics['recall_per_class'][i].item()
        print(f"{name:<15} {dice:<10.4f} {f1:<10.4f} {prec:<12.4f} {rec:<10.4f}")

    print("=" * 70 + "\n")
