# 极化三角框架 (The Polarization Triangle Framework)

## 项目介绍

极化三角框架是一个用于社会极化现象研究的计算模型和模拟工具。该框架基于三角关系理论，通过模拟代理(agent)之间的交互，研究意见极化、身份极化和道德化三者之间的动态关系。

本框架可以帮助研究人员：
- 模拟不同网络结构下的社会极化过程
- 分析身份认同、道德判断与意见形成之间的相互作用
- 探索不同参数设置对社会极化程度的影响
- 可视化模拟结果，便于直观理解

## 项目结构

```
polarization_triangle/
├── core/                     # 核心模块
│   ├── config.py             # 配置类和预设配置
│   ├── dynamics.py           # 动力学方程和更新规则
│   ├── simulation.py         # 模拟核心类
├── utils/                    # 工具模块
│   ├── network.py            # 网络创建和处理工具
│   ├── data_manager.py       # 数据管理和保存工具
├── analysis/                 # 分析模块
├── visualization/            # 可视化模块
├── experiments/              # 实验模块
│   ├── batch_runner.py       # 批量运行模拟
│   ├── morality_test.py      # 道德化率测试
│   ├── model_params_test.py  # 模型参数测试
│   ├── activation_analysis.py# 激活组件分析
├── scripts/                  # 脚本模块
│   ├── run_basic_simulation.py # 运行基本模拟
│   ├── run_model_params_test.py # 运行模型参数测试
│   ├── run_morality_test.py  # 运行道德化率测试
├── main.py                   # 主入口文件
```

## 核心模块说明

### core/config.py

定义了`SimulationConfig`类，包含所有模拟所需的配置参数，如：
- 代理数量
- 网络类型及参数
- 意见分布
- 道德化率
- 极化三角框架相关参数

同时提供了多种预设配置如：高极化配置、低极化配置、随机网络配置等。

### core/dynamics.py

实现了极化三角框架的动力学方程和更新规则，包括：
- 意见变化方程
- 社会影响计算
- 自我激活计算
- 道德化影响计算

### core/simulation.py

实现了`Simulation`类，作为模拟的核心类，包括：
- 初始化网络、代理属性
- 执行模拟步骤
- 收集和分析模拟数据
- 提供访问模拟状态的接口

## 工具模块说明

### utils/network.py

提供网络创建和处理相关的工具函数：
- `create_network`: 创建不同类型的网络（随机网络、小世界网络、无标度网络、社区网络等）
- `handle_isolated_nodes`: 处理网络中的孤立节点

### utils/data_manager.py

提供数据管理和保存相关的工具函数：
- `save_trajectory_to_csv`: 将轨迹数据保存为CSV文件
- `save_simulation_data`: 将模拟数据保存为多个CSV文件（轨迹、最终状态、网络结构、配置）

## 实验模块说明

- `batch_runner.py`: 实现批量运行多种预设配置的模拟，并比较结果
- `morality_test.py`: 实现对不同道德化率的参数扫描测试
- `model_params_test.py`: 实现对模型关键参数的参数扫描测试
- `activation_analysis.py`: 实现对激活组件（自我激活和社会影响）的分析

## 使用方法

### 安装依赖

首先克隆本仓库，然后安装所需依赖：

```bash
git clone https://github.com/yourusername/polarization-triangle-framework.git
cd polarization-triangle-framework
pip install -r requirements.txt
```

主要依赖包括：
- numpy
- networkx
- matplotlib
- numba
- scipy
- pandas
- tqdm

### 基本用法

#### 运行基本模拟

```bash
python -m polarization_triangle.main --test-type basic
```

这将使用默认参数运行一组预设配置的模拟，并将结果保存在`batch_results`目录下。

#### 运行道德化率测试

```bash
python -m polarization_triangle.main --test-type morality --output-dir morality_results --steps 300 --morality-rates 0.1 0.3 0.5 0.7 0.9
```

这将使用不同的道德化率运行多次模拟，分析道德化率对极化程度的影响，并将结果保存在指定目录。

#### 运行模型参数测试

```bash
python -m polarization_triangle.main --test-type model-params --output-dir params_results --steps 300
```

这将测试不同的模型参数（如alpha、beta、gamma等）对极化程度的影响。

#### 运行激活组件分析

```bash
python -m polarization_triangle.main --test-type activation --output-dir activation_results --steps 300
```

这将分析自我激活和社会影响组件在极化过程中的作用。

### 使用脚本

也可以直接运行各个脚本：

```bash
python -m polarization_triangle.scripts.run_basic_simulation
python -m polarization_triangle.scripts.run_morality_test
python -m polarization_triangle.scripts.run_model_params_test
```

### 自定义配置

如需使用自定义配置，可以修改`core/config.py`中的预设配置，或者在代码中创建新的`SimulationConfig`实例：

```python
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.simulation import Simulation

# 创建自定义配置
my_config = SimulationConfig(
    num_agents=200,
    network_type="ba",
    network_params={"m": 3},
    opinion_distribution="single_peak",
    morality_rate=0.4,
    delta=0.1,
    alpha=0.3,
    beta=0.2,
    gamma=0.5
)

# 创建模拟实例
sim = Simulation(my_config)

# 运行模拟
for _ in range(100):
    sim.step()

# 保存结果
sim.save_simulation_data("my_results", "custom_sim")
```

## 结果分析

模拟结果将保存为CSV文件，包括：
- `*_trajectory.csv`: 包含每个步骤每个代理的意见、身份、道德化状态等
- `*_final_state.csv`: 包含模拟结束时每个代理的最终状态
- `*_network.csv`: 包含网络结构信息
- `*_config.csv`: 包含使用的配置参数

这些数据可以用于后续分析和可视化。

## 贡献与开发

欢迎贡献代码、报告问题或提出改进建议。如需开发，请遵循以下建议：
- 使用类型注解来提高代码可读性
- 添加详细的文档字符串
- 编写测试用例确保功能正确
- 遵循项目的代码风格和结构

## 许可证

本项目采用[MIT许可证](LICENSE)。 