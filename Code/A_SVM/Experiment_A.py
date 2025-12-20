import matplotlib.pyplot as plt
import numpy as np
from Model_A import train_svm_model, get_data_loader

def run_analysis():
    print("Starting SVM Analysis Experiments")
    
    # --- Experiment 1: Model Capacity ---
    print("1. Analyzing C Parameter Capacity")
    c_values = [0.01, 0.1, 1, 10, 100]
    accuracies = []
    for c in c_values:
        acc, _, _, _ = train_svm_model(c_value=c, augment=False)
        accuracies.append(acc)
        
    plt.figure(figsize=(10, 6))
    plt.semilogx(c_values, accuracies, marker='o')
    plt.title('SVM Accuracy vs Model Capacity C')
    plt.xlabel('C Parameter Log Scale')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig('svm_capacity_analysis.png')
    
    # --- Experiment 2: Data Augmentation ---
    print("2. Analyzing Data Augmentation Impact")
    acc_base, _, _, _ = train_svm_model(c_value=1.0, augment=False)
    acc_aug, _, _, _ = train_svm_model(c_value=1.0, augment=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Baseline', 'Augmented'], [acc_base, acc_aug], color=['blue', 'orange'])
    plt.title('SVM Performance with Augmentation')
    plt.ylabel('Accuracy')
    # Add text labels
    plt.text(0, acc_base, f"{acc_base:.4f}", ha='center', va='bottom')
    plt.text(1, acc_aug, f"{acc_aug:.4f}", ha='center', va='bottom')
    plt.savefig('svm_augmentation_analysis.png')
    
    # --- Experiment 3: Training Budget (Data Size) ---
    # NEW ADDITION HERE
    print("3. Analyzing Training Budget (Dataset Size)")
    fractions = [0.1, 0.5, 1.0] # 10%, 50%, 100% data
    budget_accs = []
    
    # Get full data first
    X_full, y_full = get_data_loader('train', False)
    # We need to manually slice data here because Model_A doesn't support fraction directly
    # But we can cheat by passing sliced data? 
    # Actually simpler: modify Model_A slightly OR implement slicing here.
    # Let's verify Model_A structure. It calls get_data_loader internally.
    # To avoid changing Model_A again, let's just create a quick local training loop here.
    
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    X_test, y_test = get_data_loader('test', False)
    scaler = StandardScaler()
    
    for frac in fractions:
        # Slice data
        n_samples = int(len(y_full) * frac)
        X_sub = X_full[:n_samples]
        y_sub = y_full[:n_samples]
        
        # Train
        scaler.fit(X_sub)
        X_sub_s = scaler.transform(X_sub)
        X_test_s = scaler.transform(X_test)
        
        clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_sub_s, y_sub)
        acc = accuracy_score(y_test, clf.predict(X_test_s))
        budget_accs.append(acc)
        print(f"Data Fraction {frac*100}% Accuracy {acc:.4f}")
        
    plt.figure(figsize=(8, 6))
    plt.plot([f*100 for f in fractions], budget_accs, marker='s', color='green')
    plt.title('SVM Performance vs Training Data Budget')
    plt.xlabel('Percentage of Training Data Used')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig('svm_budget_analysis.png')
    print("Saved svm_budget_analysis.png")

if __name__ == '__main__':
    run_analysis()