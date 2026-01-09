import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

try:
    from Model_B import train_and_validate, run_test_evaluation
except ImportError:
    from .Model_B import train_and_validate, run_test_evaluation

def plot_confusion_matrix(y_true, y_pred, model_name, save_path='./'):
    # calculate the confusion matrix from true and predicted labels
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    # plot a heatmap since it's easier to spot misclassifications
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filename = os.path.join(save_path, f'confusion_matrix_{model_name}.png')
    plt.savefig(filename)
    plt.close()

RESULT_DIR = "Results_B"
os.makedirs(RESULT_DIR, exist_ok=True)
EPOCHS = 15

def run_experiments():
    report_path = os.path.join(RESULT_DIR, "cnn_final_report.txt")
    report_file = open(report_path, "w", encoding="utf-8")

    """ 
     first experiment checks model capacity by varying the network width
     we use the validation set to pick the best size
    """
    width_options = [1, 2, 3, 4, 5, 6, 7, 8]
    val_accs = []
    
    # keep track of the best model weights so we don't have to retrain later
    best_val_acc = 0.0
    winner_width = 1
    winner_state = None

    for width_mult in width_options:
        # to get the best epoch's weights
        v_acc, state = train_and_validate(
            width_mult=width_mult,
            augment_mode="none",
            epochs=EPOCHS
        )
        val_accs.append(v_acc)
        
        # update our winner if this width performed better
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            winner_width = width_mult
            winner_state = state

    print(f"selected width: x{winner_width}")

    report_file.write("Experiment Model capacity (Validation Results)\n")
    for width_mult, acc in zip(width_options, val_accs):
        report_file.write(f"Width x{width_mult} Val Accuracy: {acc:.4f}\n")
    report_file.write(f"Selected width x{winner_width}\n\n")

    # visualize the capacity sweep
    plt.figure()
    plt.plot(width_options, val_accs, marker="o")
    plt.xticks(width_options)
    plt.title("CNN Capacity Selection")
    plt.xlabel("Width multiplier")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "exp1_capacity_val.png"))
    plt.close()

    # experiment 2 checks if data augmentation helps
    # we test rotation, noise, and both combined
    aug_modes = ["none", "rotation", "noise", "parallel"]
    aug_val_accs = []
    
    best_aug_mode = "none"
    best_aug_val_acc = 0.0
    # we need the absolute best weights found here for the final test run
    best_overall_state = None 

    report_file.write(f"Experiment Data augmentation width x{winner_width} (Validation)\n")

    for mode in aug_modes:
        # if mode is 'none', we already ran it in exp 1
        if mode == "none":
            acc = best_val_acc 
            state = winner_state
        else:
            acc, state = train_and_validate(
                width_mult=winner_width,
                augment_mode=mode,
                epochs=EPOCHS
            )

        aug_val_accs.append(acc)
        report_file.write(f"Mode {mode}: {acc:.4f}\n")

        # save the best performing model configuration
        if acc > best_aug_val_acc:
            best_aug_val_acc = acc
            best_aug_mode = mode
            best_overall_state = state

    print(f"best augmentation mode: {best_aug_mode}")

    plt.figure()
    plt.bar(aug_modes, aug_val_accs, color=['gray', 'orange', 'purple', 'red'])
    plt.title(f"Augmentation Impact")
    plt.ylabel("Accuracy")
    
    # format the plot to make differences visible
    min_v, max_v = min(aug_val_accs), max(aug_val_accs)
    margin = (max_v - min_v) * 0.2 if max_v != min_v else 0.05
    plt.ylim(max(0, min_v - margin), min(1.0, max_v + margin))
    
    for i, v in enumerate(aug_val_accs):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    plt.savefig(os.path.join(RESULT_DIR, "exp2_augmentation_val.png"))
    plt.close()

    """ final step: run the absolute best model on the test set
    # this is the only time the test set is used
    # load the weights from the best run (best width + best augmentation) """
    final_acc, final_y_true, final_y_pred = run_test_evaluation(
        width_mult=winner_width,
        model_state=best_overall_state
    )

    print(f"final test accuracy: {final_acc:.4f}")

    # write the final report
    report_file.write("\n==========================================\n")
    report_file.write("FINAL EVALUATION ON TEST SET\n")
    report_file.write("==========================================\n")
    report_file.write(f"Parameters: Width x{winner_width}, Augmentation: {best_aug_mode}\n")
    report_file.write(f"TEST ACCURACY: {final_acc:.4f}\n\n")
    report_file.write("Classification Report:\n")
    report_file.write(
        classification_report(
            final_y_true,
            final_y_pred,
            target_names=["Benign", "Malignant"],
            zero_division=0
        )
    )

    # generate the confusion matrix
    plot_confusion_matrix(
        final_y_true,  final_y_pred, f"CNN", 
        save_path=RESULT_DIR
    )

    # experiment 3 generates a learning curve
    # this shows how performance scales with data size
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    budget_accs = []

    for train_fraction in ratios:
        # train a fresh model with limited data
        # finding the best epoch on val, then testing on test
        _, state = train_and_validate(
            width_mult=winner_width,
            augment_mode="none", # keep augmentation off for a clean comparison
            max_samples_ratio=train_fraction,
            epochs=EPOCHS,
        )
        
        acc, _, _ = run_test_evaluation(winner_width, state)
        budget_accs.append(acc)

    plt.figure()
    plt.plot(ratios, budget_accs, marker="s", color='green')
    plt.title(f"Learning curve")
    plt.xlabel("Fraction of training data")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "exp3_budget.png"))
    plt.close()

    report_file.close()

if __name__ == "__main__":
    run_experiments()