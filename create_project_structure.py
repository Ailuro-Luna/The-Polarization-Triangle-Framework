import os

def create_directory(path):
    """创建目录（如果不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")

def create_file(path, content=""):
    """创建文件（如果不存在）"""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"创建文件: {path}")

def create_init_file(directory):
    """在指定目录中创建__init__.py文件"""
    create_file(os.path.join(directory, "__init__.py"))

# 项目根目录
base_dir = "polarization_triangle"
create_directory(base_dir)

# 创建模块目录
modules = [
    "core",
    "utils",
    "visualization", 
    "analysis",
    "experiments",
    "scripts"
]

for module in modules:
    module_path = os.path.join(base_dir, module)
    create_directory(module_path)
    create_init_file(module_path)

# 创建core模块文件
core_files = ["simulation.py", "config.py", "dynamics.py"]
for file in core_files:
    create_file(os.path.join(base_dir, "core", file))

# 创建utils模块文件
utils_files = ["network.py", "data_manager.py"]
for file in utils_files:
    create_file(os.path.join(base_dir, "utils", file))

# 创建visualization模块文件
viz_files = ["network_viz.py", "opinion_viz.py", "activation_viz.py", "rule_viz.py"]
for file in viz_files:
    create_file(os.path.join(base_dir, "visualization", file))

# 创建analysis模块文件
analysis_files = ["trajectory.py", "activation.py"]
for file in analysis_files:
    create_file(os.path.join(base_dir, "analysis", file))

# 创建experiments模块文件
experiment_files = ["batch_runner.py", "morality_test.py", "model_params_test.py", "activation_analysis.py"]
for file in experiment_files:
    create_file(os.path.join(base_dir, "experiments", file))

# 创建脚本文件
script_files = ["run_basic_simulation.py", "run_morality_test.py", "run_model_params_test.py"]
for file in script_files:
    create_file(os.path.join(base_dir, "scripts", file))

# 创建主入口文件
create_file(os.path.join(base_dir, "main.py"))

print("项目结构创建完成！") 