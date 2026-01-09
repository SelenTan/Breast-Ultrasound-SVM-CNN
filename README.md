## Project Overview
This project benchmarks two machine learning models on the BreastMNIST dataset. The goal is to compare a classical approach against a deep learning approach under various conditions.

* **Task A:** Support Vector Machine (SVM)
* **Task B:** Convolutional Neural Network (SimpleCNN)

The project analyzes the impact of model capacity, data augmentation (random rotation, Gaussian noise), and training budget(different fraction of training data) on performance.

## Student Information
**Student Number:** 25187537

## Repository Structure
The project is organized as follows:
```
AMLS_25_26_SN25187537/
├── Code/
│   ├── A_SVM/
│   │   ├── Model_A.py       # Contains SVM data loading and training logic.
│   │   └── Experiment_A.py  # Runs experiments on C-parameter, augmentation, and data budget.
│   │
│   └── B_CNN/
│       ├── Model_B.py       # Contains SimpleCNN architecture and training loops.
│       └── Experiment_B.py  # Runs experiments on width, augmentation, and data budget.
│
├── Datasets/                # Folder where BreastMNIST data will be stored.
├── main.py                  # The entry point script. Orchestrates the execution of both Task A and Task B.
└── README.md                # Project documentation and setup instructions.
```
## Installation and Setup
To reproduce the results, please set up the environment using the provided configuration file.

**Step 1 Create the environment:**
Use the command: conda env create -f environment.yml

**Step 2 Activate the environment:**
Use the command: conda activate AMLS_Final

**Step 3 Run the pipeline:**
Execute the following command in terminal: python main.py

## Execution Process:
- The script first checks if the Datasets folder exists.
- It first executes Task A. Results and plots are saved to Code/A_SVM/Results_A/.
- Then executes Task B. Results and plots are saved to Code/B_CNN/Results_B/.

## Required Packages
This project uses the following libraries:
- torch
- torchvision
- medmnist
- scikit-learn
- matplotlib
- numpy
- seaborn



