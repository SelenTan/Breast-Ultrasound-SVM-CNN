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
    print(f"\n-- {description} --")
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
    # 1 Centralized Data Prep
    prepare_data()
    
    # 2 Run Model A Sequence
    run_script(MODEL_A_DIR, "Model_A.py", "Running Model A SVM")
    run_script(MODEL_A_DIR, "Experiment_A.py", "Generating SVM Analysis Graphs")
    
    # 3 Run Model B Sequence
    # Assuming you have similar files for B
    # If Experiment_Augmentation.py is what you have use that
    # Ideally Model B should also follow the import structure
    run_script(MODEL_B_DIR, "Model_B.py", "Running Model B CNN")
    
    # Note If you renamed your experiment file update the name below
    # Using the name from our previous conversation
    run_script(MODEL_B_DIR, "Experiment_B.py", "Generating CNN Analysis Graphs")

    #Generate SVM and CNN comparison graph
    print("\n--Generating Final Comparison--")
    
    def get_max_accuracy(file_path):
        """
        读取报告文件，寻找最高的 Accuracy 数值
        """
        max_acc = 0.0
        if not os.path.exists(file_path):
            print(f"Warning: Report file not found at {file_path}")
            return 0.0
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 统一转成小写
                    line_lower = line.lower()
                    if "accuracy" in line_lower:
                        # 把:和=都替换成空格
                        cleaned_line = line_lower.replace(":", " ").replace("=", " ")
                        
                        # 按空格切分，尝试把每一个片段转成数字
                        parts = cleaned_line.split()
                        for part in parts:
                            try:
                                val = float(part)
                                # 简单的过滤器：只取合理的准确率范围 (0.0 到 1.0)
                                if 0.0 < val <= 1.0 and val > max_acc:
                                    max_acc = val
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return max_acc

    # 1. 定义报告文件的绝对路径
    # 注意：这里假设 Experiment_A 生成的是 Results_A/svm_final_report.txt
    svm_report_path = os.path.join(MODEL_A_DIR, 'Results_A', 'svm_final_report.txt')
    cnn_report_path = os.path.join(MODEL_B_DIR, 'Results_B', 'cnn_final_report.txt')
    
    # 2. 自动读取分数
    svm_acc = get_max_accuracy(svm_report_path)
    cnn_acc = get_max_accuracy(cnn_report_path)
    
    print(f"   > Detected Best SVM Accuracy: {svm_acc:.4f}")
    print(f"   > Detected Best CNN Accuracy: {cnn_acc:.4f}")

    # 3. 画图
    import matplotlib.pyplot as plt
    
    if svm_acc > 0 or cnn_acc > 0:
        plt.figure(figsize=(6, 5))
        # 定义柱状图
        bars = plt.bar(['SVM', 'CNN'], [svm_acc, cnn_acc], color=["#52a8e7", "#e8da6f"])
        
        plt.title('Final Benchmark: SVM vs CNN')
        plt.ylabel('Test Accuracy (Best Config)')
        plt.ylim(0, 1.1) # 留出顶部空间写数字
        
        # 在柱子上方标注数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, 
                     f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
            
        # 保存图片到根目录
        save_path = os.path.join(BASE_DIR, 'final_comparison.png')
        plt.savefig(save_path)
    else:
        print("   ! No valid accuracy data found. Please check if report text files exist.")

    print("\nAll Tasks Completed Successfully")

if __name__ == "__main__":
    main()