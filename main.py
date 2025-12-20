import sys
import os
import medmnist
from medmnist import INFO

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'Datasets')
MODEL_A_DIR = os.path.join(BASE_DIR, 'Code', 'A_SVM')
MODEL_B_DIR = os.path.join(BASE_DIR, 'Code', 'B_CNN')
DATA_FLAG = 'breastmnist'

def prepare_data():
    print("System Checking Dataset")
    info = INFO[DATA_FLAG]
    file_name = f"{DATA_FLAG}.npz"
    file_path = os.path.join(DATA_ROOT, file_name)
    
    if os.path.exists(file_path):
        print(f"Dataset found at {file_path}")
    else:
        print("Dataset not found downloading...")
        try:
            DataClass = getattr(medmnist, info['python_class'])
            DataClass(split='train', download=True, root=DATA_ROOT)
            print("Download complete")
        except Exception as e:
            print(f"Error downloading data {e}")
            sys.exit(1)

def run_script(folder, script_name, description):
    print(f"\n--- {description} ---")
    original_cwd = os.getcwd()
    try:
        os.chdir(folder)
        exit_code = os.system(f"python {script_name}")
        if exit_code != 0:
            print(f"Error {script_name} failed")
    except Exception as e:
        print(f"Execution Error {e}")
    finally:
        os.chdir(original_cwd)

def main():
    print("AMLS Assignment 25 26 Execution Pipeline")
    
    # 1 Centralized Data Prep
    prepare_data()
    
    # 2 Run Model A Sequence
    run_script(MODEL_A_DIR, "Model_A.py", "Running Best Model A SVM")
    run_script(MODEL_A_DIR, "Experiment_A.py", "Generating SVM Analysis Graphs")
    
    # 3 Run Model B Sequence
    # Assuming you have similar files for B
    # If Experiment_Augmentation.py is what you have use that
    # Ideally Model B should also follow the import structure
    run_script(MODEL_B_DIR, "Model_B.py", "Running Best Model B CNN")
    
    # Note If you renamed your experiment file update the name below
    # Using the name from our previous conversation
    run_script(MODEL_B_DIR, "Experiment_B.py", "Generating CNN Analysis Graphs")

    print("\nAll Tasks Completed Successfully")

if __name__ == "__main__":
    main()