import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

try:
    from Model_A import get_data_loader, extract_features, train_and_predict
except ImportError:
    from .Model_A import get_data_loader, extract_features, train_and_predict

RESULT_DIR = 'Results_A'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path='./'):
    # compute confusion matrix from ground truth and predictions
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    # visualize the matrix as a heatmap for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filename = os.path.join(save_path, f'confusion_matrix_{model_name}.png')
    plt.savefig(filename)
    plt.close()

def run_experiments():
    report_file = open(os.path.join(RESULT_DIR, 'svm_final_report.txt'), 'w')

    """initialize data loaders for training and validation sets
     training set is used for parameter learning
     validation set is strictly for hyperparameter tuning"""
    X_train_raw, y_train_raw = get_data_loader('train', augment_mode='none')
    X_val_raw, y_val = get_data_loader('val', augment_mode='none')

    C_values = [0.1, 0.5, 1, 2, 5, 10, 50, 100]
    
    # combined experiment 1 and 2
    # find optimal c for both raw and hog features
    
    # evaluate performance using raw pixel features
    acc_scores_raw = []
    for c in C_values:
        acc, _ = train_and_predict(X_train_raw, y_train_raw, X_val_raw, y_val, C_value=c)
        acc_scores_raw.append(acc)

    best_c_raw_idx = np.argmax(acc_scores_raw)
    best_c_raw = C_values[best_c_raw_idx]
    best_acc_raw = acc_scores_raw[best_c_raw_idx]
    print(f"best C for RAW: {best_c_raw} with acc {best_acc_raw:.4f}")

    # evaluate performance using hog features
    X_train_hog = extract_features(X_train_raw, 'hog')
    X_val_hog = extract_features(X_val_raw, 'hog')
    
    acc_scores_hog = []
    for c in C_values:
        acc, _ = train_and_predict(X_train_hog, y_train_raw, X_val_hog, y_val, C_value=c)
        acc_scores_hog.append(acc)
        
    best_c_hog_idx = np.argmax(acc_scores_hog)
    best_c_hog = C_values[best_c_hog_idx]
    best_acc_hog = acc_scores_hog[best_c_hog_idx]
    print(f"best C for HOG: {best_c_hog} with acc {best_acc_hog:.4f}")

    plt.figure(figsize=(7, 5))
    C_labels = [str(c) for c in C_values]
    plt.plot(range(len(C_values)), acc_scores_raw, marker='o', label='Raw', color='gray')
    plt.plot(range(len(C_values)), acc_scores_hog, marker='s', label='HOG', color='blue')
    plt.xticks(range(len(C_values)), C_labels)
    plt.title('SVM Capacity Selection')
    plt.xlabel('C Parameter')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULT_DIR, 'SVM_exp1_capacity_comparison.png'))
    plt.close()

    # select the superior feature extraction method based on validation accuracy
    if best_acc_hog > best_acc_raw:
        winner_type = 'hog'
        best_c = best_c_hog
        best_val_acc = best_acc_hog
        print(f"feature winner: HOG")
    else:
        winner_type = 'raw'
        best_c = best_c_raw
        best_val_acc = best_acc_raw
        print(f"feature winner: RAW")

    report_file.write("Experiment Feature pipelines (Validation Results)\n")
    report_file.write(f"Raw Best Acc: {best_acc_raw:.4f} (C={best_c_raw})\n")
    report_file.write(f"HOG Best Acc: {best_acc_hog:.4f} (C={best_c_hog})\n")
    report_file.write(f"Selected winner: {winner_type}\n\n")

    # experiment 2 investigates data augmentation impact
    # rotation and noise are applied to training data to assess validation improvement
    aug_modes = ['none', 'rotation', 'noise', 'parallel']
    aug_val_accs = []
    
    best_aug_mode = 'none'
    best_val_acc_so_far = 0.0

    # set validation features based on the winner from previous experiment
    if winner_type == 'hog':
        X_val_feat = X_val_hog
    else:
        X_val_feat = X_val_raw

    for mode in aug_modes:
        X_train_aug_raw, y_train_aug = get_data_loader('train', augment_mode=mode) # retrieve fresh loader with specific augmentation strategy applied
        
        # apply feature extraction to the augmented training set
        X_train_aug_feat = extract_features(X_train_aug_raw, feature_type=winner_type)

        # train and validate using optimal c parameter found earlier
        acc, _ = train_and_predict(X_train_aug_feat, y_train_aug, X_val_feat, y_val, C_value=best_c)
        aug_val_accs.append(acc)

        # track the best performing augmentation strategy
        if acc > best_val_acc_so_far:
            best_val_acc_so_far = acc
            best_aug_mode = mode

    print(f"best augmentation mode: {best_aug_mode}")

    report_file.write(f"Experiment Data augmentation on {winner_type} (Validation)\n")
    for i, mode in enumerate(aug_modes):
        report_file.write(f"Mode {mode}: {aug_val_accs[i]:.4f}\n")

    plt.figure()
    plt.bar(aug_modes, aug_val_accs, color=['gray', 'orange', 'purple', 'red'])
    plt.title(f'SVM Augmentation Impact ({winner_type})')
    plt.ylabel('Validation Accuracy')
    
    # adjust y axis limits for better visual differentiation
    min_val, max_val = min(aug_val_accs), max(aug_val_accs)
    margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.05
    plt.ylim(max(0, min_val - margin), min(1.0, max_val + margin))
    
    for i, v in enumerate(aug_val_accs):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    plt.savefig(os.path.join(RESULT_DIR, 'SVM_exp2_augmentation_val.png'))
    plt.close()

    # final evaluation phase on test data
    X_test_raw, y_test = get_data_loader('test', augment_mode='none')
    X_test_feat = extract_features(X_test_raw, feature_type=winner_type)
    X_train_final_raw, y_train_final = get_data_loader('train', augment_mode=best_aug_mode)
    X_train_final_feat = extract_features(X_train_final_raw, feature_type=winner_type)

    # train final model and obtain performance metrics
    final_acc, final_preds = train_and_predict(
        X_train_final_feat, y_train_final, 
        X_test_feat, y_test, 
        C_value=best_c
    )

    print(f"final test accuracy: {final_acc:.4f}")

    # generate comprehensive final report
    report_file.write("\n==========================================\n")
    report_file.write("FINAL EVALUATION ON TEST SET\n")
    report_file.write("==========================================\n")
    report_file.write(f"Selected Parameters:\n")
    report_file.write(f" - C: {best_c}\n")
    report_file.write(f" - Features: {winner_type}\n")
    report_file.write(f" - Augmentation: {best_aug_mode}\n\n")
    report_file.write(f"TEST ACCURACY: {final_acc:.4f}\n\n")
    report_file.write("Classification Report:\n")
    report_file.write(classification_report(y_test, final_preds, target_names=['Benign', 'Malignant']))

    # visualize confusion matrix for test results
    plot_confusion_matrix(y_test, final_preds, "SVM", save_path=RESULT_DIR)

    # experiment 3 learning curve analysis
    # evaluates true performance on test set across different data budgets
    ratios = [0.1, 0.2,0.3,0.4, 0.5,0.6, 0.7,0.8, 0.9, 1.0]
    budget_accs = []

    for r in ratios:
        X_sub_raw, y_sub = get_data_loader('train', augment_mode='none', max_samples_ratio=r)
        X_sub_feat = extract_features(X_sub_raw, feature_type=winner_type)
        
        # evaluate performance on test set
        acc, _ = train_and_predict(X_sub_feat, y_sub, X_test_feat, y_test, C_value=best_c)
        budget_accs.append(acc)

    plt.figure()
    plt.plot(ratios, budget_accs, marker='s', color='green')
    plt.title(f'SVM Learning Curve')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'SVM_exp3_budget.png'))
    plt.close()

    report_file.close()

if __name__ == '__main__':
    run_experiments()