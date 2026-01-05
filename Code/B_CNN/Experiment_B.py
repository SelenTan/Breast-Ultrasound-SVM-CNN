import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# import the cnn training runner
# fallback import makes it work both as a script and as a module
try:
    from Model_B import train_eval_cnn
except ImportError:
    from .Model_B import train_eval_cnn


# store all model B outputs here
# figures and logs end up in this folder
RESULT_DIR = "Results_B"
os.makedirs(RESULT_DIR, exist_ok=True)

# keep epochs fixed so comparisons are about the thing we changed
EPOCHS = 15


def run_experiments():
    # runs the full set of cnn experiments for the report
    # we test capacity, augmentation, and training budget in separate blocks

    # write a detailed log for later copy paste into the report
    report_path = os.path.join(RESULT_DIR, "cnn_final_report.txt")
    report_file = open(report_path, "w", encoding="utf-8")

    # capacity experiment
    # width_mult scales channel count so it is basically model size
    width_options = [1, 2, 3, 4, 5, 6, 7, 8]
    capacity_accs = []

    # loop over widths and collect the test accuracy for each one
    for width_mult in width_options:
        acc, _, _ = train_eval_cnn(
            width_mult=width_mult,
            augment_mode="none",
            epochs=EPOCHS,
        )
        capacity_accs.append(acc)

    # pick the best width based on test accuracy from the sweep
    winner_idx = int(np.argmax(capacity_accs))
    winner_width = width_options[winner_idx]
    winner_acc_base = capacity_accs[winner_idx]
    
    print(f"Capacity Analysis. Winner Width = x{winner_width} (Acc: {winner_acc_base:.4f})")

    # log the sweep results so we can quote exact numbers later
    report_file.write("Experiment Model capacity\n")
    for width_mult, acc in zip(width_options, capacity_accs):
        report_file.write(f"width x{width_mult} Accuracy: {acc:.4f}\n")
    report_file.write(f"selected width x{winner_width}\n\n")

    # plot capacity vs accuracy
    plt.figure()
    plt.plot(width_options, capacity_accs, marker="o")
    plt.xticks(width_options)
    plt.title("CNN Model Capacity (Channel Width)")
    plt.xlabel("Width multiplier")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "exp1_capacity.png"))
    plt.close()

    # Augmentation experiment
    # reuse the winner width so we are only testing augmentation effects
    aug_modes = ["none", "rotation", "noise", "combined"]
    aug_accs = []

    report_file.write(f"Experiment Data augmentation width x{winner_width}\n")

    for mode in aug_modes:
        # for none, reuse the best result from the capacity sweep
        # avoids retraining the exact same setup twice
        if mode == "none":
            aug_accs.append(winner_acc_base)
            report_file.write(f"\nmode {mode}\n")
            report_file.write(f" Accuracy: {winner_acc_base:.4f}\n")
            continue

        # train with the chosen augmentation and dump a full classification report
        acc, y_true, y_pred = train_eval_cnn(
            width_mult=winner_width,
            augment_mode=mode,
            epochs=EPOCHS,
        )
        aug_accs.append(acc)

        report_file.write(f"\nmode {mode}\n")
        report_file.write(f" Accuracy: {acc:.4f}\n")
        report_file.write(
            classification_report(
                y_true,
                y_pred,
                target_names=["Benign", "Malignant"],
            )
        )
        report_file.write("\n")

    print(f"Augmentation Analysis. Best Mode Acc: {max(aug_accs):.4f}")

    # bar chart is clearer here since modes are discrete categories
    plt.figure()
    plt.bar(aug_modes, aug_accs)
    plt.title(f"augmentation impact")
    plt.ylabel("Test accuracy")

    # tighten y range so small gaps between modes are visible
    min_v, max_v = min(aug_accs), max(aug_accs)
    margin = (max_v - min_v) * 0.2 if max_v != min_v else 0.05
    plt.ylim(max(0, min_v - margin), min(1.0, max_v + margin))

    # add numbers on bars so the figure can be read without the log file
    for i, v in enumerate(aug_accs):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.savefig(os.path.join(RESULT_DIR, "exp2_augmentation.png"))
    plt.close()

    # training budget experiment
    # vary how much training data we allow, everything else stays the same
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    budget_accs = []

    for train_fraction in ratios:
        acc, _, _ = train_eval_cnn(
            width_mult=winner_width,
            augment_mode="none",
            max_samples_ratio=train_fraction,
            epochs=EPOCHS,
        )
        budget_accs.append(acc)
        
    print(f"Budget Analysis. Full Data Acc: {budget_accs[-1]:.4f}")

    # learning curve style plot
    plt.figure()
    plt.plot(ratios, budget_accs, marker="s")
    plt.title(f"Learning curve")
    plt.xlabel("Fraction of training data")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "exp3_budget.png"))
    plt.close()

    report_file.close()

    # summary to know what got picked and where outputs went
    print(f"capacity winner width x{winner_width}  Accuracy: {winner_acc_base:.4f}")


if __name__ == "__main__":
    run_experiments()
