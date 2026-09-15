import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def plot_convergence_curves(log_file, output_dir):
    """
    Read training log and plot convergence curves.
    log_file: path to training_log.json
    output_dir: where to save the plots
    """
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    epochs = [entry['epoch'] for entry in logs]
    train_losses = [entry['train_loss'] for entry in logs]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Training Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in logs[0]:
        val_losses = [entry['val_loss'] for entry in logs]
        axes[0].plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training / Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: AP / MPJPE if available
    if 'AP25' in logs[0]:
        ap25 = [entry['AP25'] for entry in logs]
        axes[1].plot(epochs, ap25, 'g-', linewidth=2, label='AP@25mm')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('AP (%)', fontsize=12)
        axes[1].set_title('Validation AP@25mm', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    if 'MPJPE' in logs[0]:
        # Create secondary y-axis for MPJPE
        ax_mpjpe = axes[1].twinx()
        mpjpe = [entry['MPJPE'] for entry in logs]
        ax_mpjpe.plot(epochs, mpjpe, 'm--', linewidth=2, label='MPJPE (mm)')
        ax_mpjpe.set_ylabel('MPJPE (mm)', fontsize=12, color='m')
        ax_mpjpe.legend(fontsize=11, loc='center right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'convergence_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Convergence curves saved to {save_path}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python plot_curves.py <log_file> <output_dir>')
        sys.exit(1)
    plot_convergence_curves(sys.argv[1], sys.argv[2])
