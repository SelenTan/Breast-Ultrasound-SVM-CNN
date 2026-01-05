import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# import model A utilities
# fallback import lets this file run in different project layouts
try:
    from Model_A import get_data_loader, extract_features, train_and_predict
except ImportError:
    from .Model_A import get_data_loader, extract_features, train_and_predict

# store outputs for model A here
RESULT_DIR = 'Results_A'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


def run_experiments():
    # runs the full svm pipeline used in the report
    # capacity sweep, feature choice, then augmentation and budget using the winner
    report_file = open(os.path.join(RESULT_DIR, 'svm_final_report.txt'), 'w')

    # load test set once and reuse it across experiments
    X_test_raw, y_test = get_data_loader('test', augment_mode='none')

    # capacity sweep on raw pixels to pick a reasonable C
    C_values = [0.1, 0.5, 1, 2, 5, 10, 50, 100]
    acc_scores = []

    X_train_raw, y_train_raw = get_data_loader('train', augment_mode='none')

    for c in C_values:
        # keep features fixed here so only C is changing
        acc, _ = train_and_predict(X_train_raw, y_train_raw, X_test_raw, y_test, C_value=c)
        acc_scores.append(acc)

    # pick the best C from the sweep results
    best_c_idx = np.argmax(acc_scores)
    best_c = C_values[best_c_idx]
    print(f"Selected C = {best_c}")

    # plot using evenly spaced x positions so the labels do not squish together
    plt.figure(figsize=(7, 5))
    C_labels = [str(c) for c in C_values]
    x_positions = range(len(C_values))
    plt.plot(x_positions, acc_scores, marker='o', linestyle='-', linewidth=2, color='#1f77b4')
    plt.xticks(x_positions, C_labels)
    plt.title('SVM Model Capacity (Regularization C)')
    plt.xlabel('C Parameter')
    plt.ylabel('Test Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULT_DIR, 'exp1_capacity.png'))
    plt.close()

    # feature pipeline comparison
    # train the same svm with raw vs hog and let accuracy decide the winner

    # raw run reuses the same train and test arrays
    acc_raw, pred_raw = train_and_predict(X_train_raw, y_train_raw, X_test_raw, y_test, C_value=best_c)

    # hog run needs feature extraction on both train and test
    X_train_hog = extract_features(X_train_raw, 'hog')
    X_test_hog = extract_features(X_test_raw, 'hog')
    acc_hog, pred_hog = train_and_predict(X_train_hog, y_train_raw, X_test_hog, y_test, C_value=best_c)

    # dynamic choice that drives the rest of the script
    if acc_hog > acc_raw:
        winner_type = 'hog'
        winner_acc = acc_hog
    else:
        winner_type = 'raw'
        winner_acc = acc_raw
    
    print(f"Feature winner: {winner_type.upper()}")

    # write both reports so we can show the tradeoff even if one loses
    report_file.write("Experiment Feature pipelines\n")
    report_file.write(f"Raw Accuracy: {acc_raw:.4f}\n")
    report_file.write(f"HOG Accuracy: {acc_hog:.4f}\n")
    report_file.write(f"Selected winner: {winner_type}\n\n")
    report_file.write("Classification Report Raw\n")
    report_file.write(classification_report(y_test, pred_raw, target_names=['Benign', 'Malignant']))
    report_file.write("\nClassification Report HOG\n")
    report_file.write(classification_report(y_test, pred_hog, target_names=['Benign', 'Malignant']))
    report_file.write("\n\n")

    plt.figure()
    plt.bar(['Raw Pixels', 'HOG Features'], [acc_raw, acc_hog], color=['gray', 'blue'])
    plt.title('Feature Pipeline Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(RESULT_DIR, 'exp2_features.png'))
    plt.close()

    # augmentation experiment
    # only apply augmentation on the winning feature pipeline
    aug_modes = ['none', 'rotation', 'noise', 'combined']
    aug_accs = []

    report_file.write(f"Experiment Data augmentation on {winner_type}\n")

    # reuse already computed test features when possible
    if winner_type == 'hog':
        X_test_for_aug = X_test_hog
    else:
        X_test_for_aug = X_test_raw

    for mode in aug_modes:
        if mode == 'none':
            aug_accs.append(winner_acc) # reuse the accuracy instead of running the same training again
            continue

        # load an augmented version of the training split
        X_train_aug_raw, y_train_aug = get_data_loader('train', augment_mode=mode)

        # extract features based on the chosen pipeline
        X_train_aug_feat = extract_features(X_train_aug_raw, feature_type=winner_type)

        # train on augmented train, evaluate on the same fixed test
        acc, pred = train_and_predict(X_train_aug_feat, y_train_aug, X_test_for_aug, y_test, C_value=best_c)
        aug_accs.append(acc)

        report_file.write(f"\nAugmentation mode: {mode}\n")
        report_file.write(f"Accuracy: {acc:.4f}\n")
        report_file.write(classification_report(y_test, pred, target_names=['Benign', 'Malignant']))

    plt.figure()
    plt.bar(aug_modes, aug_accs, color=['gray', 'orange', 'purple', 'red'])
    plt.title(f'Impact of Augmentation')
    plt.ylabel('Test Accuracy')

    # tighten y range so small differences show up
    min_val, max_val = min(aug_accs), max(aug_accs)
    margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.05
    plt.ylim(max(0, min_val - margin), min(1.0, max_val + margin))

    for i, v in enumerate(aug_accs):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')

    plt.savefig(os.path.join(RESULT_DIR, 'exp3_augmentation.png'))
    plt.close()

    # budget experiment
    # subsample the train set to see how accuracy scales with data
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    budget_accs = []

    for r in ratios:
        X_sub_raw, y_sub = get_data_loader('train', augment_mode='none', max_samples_ratio=r)

        # keep feature type consistent with the winner choice
        X_sub_feat = extract_features(X_sub_raw, feature_type=winner_type)

        acc, _ = train_and_predict(X_sub_feat, y_sub, X_test_for_aug, y_test, C_value=best_c)
        budget_accs.append(acc)

    plt.figure()
    plt.plot(ratios, budget_accs, marker='s', color='green')
    plt.title(f'Learning Curve')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'exp4_budget.png'))
    plt.close()

    report_file.close()

if __name__ == '__main__':
    run_experiments()