import matplotlib.pyplot as plt
from Model_B import train_cnn_model

def run_analysis():
    print("Starting CNN Analysis Experiments")
    epochs = 15
    
    # --- Experiment 1 & 3: Augmentation & Training Budget ---
    # (Already existing, we keep it)
    print("1. Analyzing Augmentation & Budget")
    hist_base = train_cnn_model(augment_mode=False, epochs=epochs)
    hist_aug = train_cnn_model(augment_mode=True, epochs=epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), hist_base, label='Baseline', marker='o', linestyle='--')
    plt.plot(range(1, epochs+1), hist_aug, label='Augmented', marker='s')
    plt.title('CNN Augmentation & Training Budget Analysis')
    plt.xlabel('Training Budget (Epochs)')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_aug_budget_analysis.png')
    
    # --- Experiment 2: Model Capacity (NEW) ---
    print("2. Analyzing Model Capacity (Network Width)")
    # Compare 0.5x width vs 1.0x width
    # Note: Training small model might vary, run carefully
    hist_small = train_cnn_model(augment_mode=True, epochs=epochs, capacity_mult=0.5)
    hist_normal = train_cnn_model(augment_mode=True, epochs=epochs, capacity_mult=1.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), hist_small, label='Low Capacity (0.5x Filters)', marker='x')
    plt.plot(range(1, epochs+1), hist_normal, label='Normal Capacity (1.0x Filters)', marker='o')
    plt.title('CNN Model Capacity Analysis')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_capacity_analysis.png')
    print("Saved cnn_capacity_analysis.png")

if __name__ == '__main__':
    run_analysis()