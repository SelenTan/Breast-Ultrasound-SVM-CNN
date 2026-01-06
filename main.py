import sys
import os
import subprocess

# get the project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# setting up paths to match your specific folder structure
CODE_DIR = os.path.join(BASE_DIR, 'Code')
DATASETS_DIR = os.path.join(BASE_DIR, 'Datasets')

# mapping to the folders shown in your screenshot
MODEL_A_DIR = os.path.join(CODE_DIR, 'A_SVM') 
MODEL_B_DIR = os.path.join(CODE_DIR, 'B_CNN')

def check_structure():
    # just checking if the folders are where they should be
    if not os.path.exists(CODE_DIR):
        print(f"error: code folder not found at {CODE_DIR}")
        return False

    if not os.path.exists(MODEL_A_DIR):
        print(f"error: model a folder not found at {MODEL_A_DIR}")
        print("expected folder: A_SVM")
        return False

    if not os.path.exists(MODEL_B_DIR):
        print(f"error: model b folder not found at {MODEL_B_DIR}")
        print("expected folder: B_CNN")
        return False
    
    # create the datasets folder if it is missing so scripts don't crash
    if not os.path.exists(DATASETS_DIR):
        print("info: datasets folder missing, creating it now...")
        os.makedirs(DATASETS_DIR)
        
    return True

def run_script(folder_path, script_name, description):
    # runs the script using subprocess to ensure the environment is correct
    # we also set the working directory so relative paths work inside the scripts
    script_path = os.path.join(folder_path, script_name)
    
    if not os.path.exists(script_path):
        print(f"error: script not found at {script_path}")
        return

    print(f"\nstarting: {description}")

    try:
        # use sys.executable to make sure we use the same python interpreter
        subprocess.run(
            [sys.executable, script_name], 
            cwd=folder_path, 
            check=True
        )
        print(f"finished: {description}")
    except subprocess.CalledProcessError as e:
        print(f"error: execution failed for {script_name}")
        print(f"details: {e}")
    except Exception as e:
        print(f"error: unexpected issue {e}")

def main():
    if not check_structure():
        sys.exit(1)

    # run the svm experiment
    # assuming the script inside A_SVM is named Experiment_A.py
    run_script(MODEL_A_DIR, "Experiment_A.py", "Task A (SVM)")

    # run the cnn experiment
    # assuming the script inside B_CNN is named Experiment_B.py
    run_script(MODEL_B_DIR, "Experiment_B.py", "Task B (CNN)")

    print("\nall tasks completed")

if __name__ == "__main__":
    main()