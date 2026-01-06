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
    # calculate the confusion matrix from true and predicted labels
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    # use a heatmap because it makes reading the errors much easier
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    filename = f'{save_path}confusion_matrix_{model_name}.png'
    plt.savefig(filename)
    plt.close()

def run_experiments():
    report_file = open(os.path.join(RESULT_DIR, 'svm_final_report.txt'), 'w')

    # load the three dataset splits
    # train is for learning parameters
    # val is for tuning hyperparameters like C or features
    # test is locked away until the very end
    X_train_raw, y_train_raw = get_data_loader('train', augment_mode='none')
    X_val_raw, y_val = get_data_loader('val', augment_mode='none')
    X_test_raw, y_test = get_data_loader('test', augment_mode='none')

    # first experiment is checking model capacity
    # we need to find the best regularization parameter c
    C_values = [0.1, 0.5, 1, 2, 5, 10, 50, 100]
    acc_scores_val = []

    for c in C_values:
        # train on train set and evaluate on validation set
        acc, _ = train_and_predict(X_train_raw, y_train_raw, X_val_raw, y_val, C_value=c)
        acc_scores_val.append(acc)

    # find the c value that gave the highest validation accuracy
    best_c_idx = np.argmax(acc_scores_val)
    best_c = C_values[best_c_idx]
    print(f"selected C = {best_c}")

    # plot the results so we can see the trend
    plt.figure(figsize=(7, 5))
    C_labels = [str(c) for c in C_values]
    plt.plot(range(len(C_values)), acc_scores_val, marker='o', color='#1f77b4')
    plt.xticks(range(len(C_values)), C_labels)
    plt.title('SVM Capacity')
    plt.xlabel('C Parameter')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULT_DIR, 'exp1_capacity_val.png'))
    plt.close()

    # second experiment compares feature extraction methods
    # we check if raw pixels or hog features work better
    # check performance of raw pixels on validation set
    acc_raw_val, _ = train_and_predict(X_train_raw, y_train_raw, X_val_raw, y_val, C_value=best_c)

    # check performance of hog features on validation set
    X_train_hog = extract_features(X_train_raw, 'hog')
    X_val_hog = extract_features(X_val_raw, 'hog')
    acc_hog_val, _ = train_and_predict(X_train_hog, y_train_raw, X_val_hog, y_val, C_value=best_c)

    # decide the winner based purely on validation scores
    if acc_hog_val > acc_raw_val:
        winner_type = 'hog'
        print(f"feature winner: HOG")
    else:
        winner_type = 'raw'
        print(f"feature winner: RAW")

    # save these stats to the report file
    report_file.write("Experiment Feature pipelines (Validation Results)\n")
    report_file.write(f"Raw Val Accuracy: {acc_raw_val:.4f}\n")
    report_file.write(f"HOG Val Accuracy: {acc_hog_val:.4f}\n")
    report_file.write(f"Selected winner: {winner_type}\n\n")

    # bar chart for visual comparison
    plt.figure()
    bars= plt.bar(['Raw Pixels', 'HOG Features'], [acc_raw_val, acc_hog_val], color=['gray', 'blue'])
    plt.title('Feature Pipeline Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    # loop over the bars to add the accuracy numbers
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    plt.savefig(os.path.join(RESULT_DIR, 'exp2_features_val.png'))
    plt.close()

    # third experiment checks if data augmentation helps
    # we apply rotation and noise to training data and check validation score
    aug_modes = ['none', 'rotation', 'noise', 'parallel']
    aug_val_accs = []
    
    best_aug_mode = 'none'
    best_val_acc_so_far = 0.0

    # prepare validation features using the winner from the previous step
    if winner_type == 'hog':
        X_val_feat = X_val_hog
    else:
        X_val_feat = X_val_raw

    for mode in aug_modes:
        # get a fresh loader with the specific augmentation applied
        X_train_aug_raw, y_train_aug = get_data_loader('train', augment_mode=mode)
        
        # extract features for this new training set
        X_train_aug_feat = extract_features(X_train_aug_raw, feature_type=winner_type)

        # train and validate
        acc, _ = train_and_predict(X_train_aug_feat, y_train_aug, X_val_feat, y_val, C_value=best_c)
        aug_val_accs.append(acc)

        # keep track of the best performing mode
        if acc > best_val_acc_so_far:
            best_val_acc_so_far = acc
            best_aug_mode = mode

    print(f"best augmentation mode: {best_aug_mode}")

    report_file.write(f"Experiment Data augmentation on {winner_type} (Validation)\n")
    for i, mode in enumerate(aug_modes):
        report_file.write(f"Mode {mode}: {aug_val_accs[i]:.4f}\n")

    plt.figure()
    plt.bar(aug_modes, aug_val_accs, color=['gray', 'orange', 'purple', 'red'])
    plt.title(f'Augmentation Impact')
    plt.ylabel('Accuracy')
    min_val, max_val = min(aug_val_accs), max(aug_val_accs)
    margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.05
    plt.ylim(max(0, min_val - margin), min(1.0, max_val + margin))
    for i, v in enumerate(aug_val_accs):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    plt.savefig(os.path.join(RESULT_DIR, 'exp3_augmentation_val.png'))
    plt.close()

    """" finally we run the evaluation on the test set
     this is done only once with the best parameters we found 
     load the full training set with the best augmentation """
    X_train_final_raw, y_train_final = get_data_loader('train', augment_mode=best_aug_mode)
    
    # extract features for both train and test
    X_train_final_feat = extract_features(X_train_final_raw, feature_type=winner_type)
    X_test_feat = extract_features(X_test_raw, feature_type=winner_type)

    # train the final model and get the score
    final_acc, final_preds = train_and_predict(
        X_train_final_feat, y_train_final, 
        X_test_feat, y_test, 
        C_value=best_c
    )

    print(f"final test accuracy: {final_acc:.4f}")

    # write all the final details to the report
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

    # save the confusion matrix image
    plot_confusion_matrix(y_test, final_preds, "SVM", save_path=os.path.join(RESULT_DIR, ''))

    # last experiment checks how the model learns with different amounts of data
    # this is just for analysis and doesn't change our model choice
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    budget_accs = []

    for r in ratios:
        """ load a subset of the training data
         we stick to no augmentation here to keep the comparison clean"""
        X_sub_raw, y_sub = get_data_loader('train', augment_mode='none', max_samples_ratio=r)
        X_sub_feat = extract_features(X_sub_raw, feature_type=winner_type)
        
        acc, _ = train_and_predict(X_sub_feat, y_sub, X_test_feat, y_test, C_value=best_c)
        budget_accs.append(acc)

    plt.figure()
    plt.plot(ratios, budget_accs, marker='s', color='green')
    plt.title(f'Learning Curve (Test Set)')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'exp4_budget.png'))
    plt.close()

    report_file.close()

if __name__ == '__main__':
    run_experiments()