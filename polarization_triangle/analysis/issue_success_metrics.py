"""
议题成功度量模块 (Issue Success Measurement Module)

该模块提供一套综合指标来衡量新议题在社会网络中的传播效果：
- 成功推广：形成新的社会共识，跨越身份边界
- 引起极化：导致意见分化，加剧身份对立

核心理念：区分consensus-building vs polarization-inducing的议题传播模式
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from polarization_triangle.core.simulation import Simulation


class IssueSuccessMetrics:
    """议题成功度量指标类"""
    
    def __init__(self):
        self.name = "Issue Success Measurement System"
        self.version = "1.0"
    
    def calculate_issue_success_index(self, sim: Simulation, 
                                    moral_innovator_opinion: Optional[float] = None,
                                    trajectory: Optional[List[np.ndarray]] = None,
                                    weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        计算议题成功度量指数 (Issue Success Measurement Index, ISMI)
        
        参数:
        sim -- Simulation实例
        moral_innovator_opinion -- 道德创新者的意见值（如Zealot意见）
        trajectory -- 意见演化轨迹（用于稳定性分析）
        weights -- 各子指标的权重
        
        返回:
        包含所有指标的字典
        """
        if weights is None:
            weights = {
                'consensus_convergence': 0.3,    # 共识收敛权重
                'directional_influence': 0.25,   # 方向性影响权重
                'identity_bridging': 0.25,       # 身份跨越权重
                'temporal_stability': 0.2        # 时间稳定性权重
            }
        
        results = {}
        
        # 1. 共识收敛指数 (Consensus Convergence Index)
        cci = self.calculate_consensus_convergence_index(sim, moral_innovator_opinion)
        results['consensus_convergence_index'] = cci
        
        # 2. 方向性影响指数 (Directional Influence Index)
        if moral_innovator_opinion is not None and hasattr(sim, 'initial_opinions'):
            dii = self.calculate_directional_influence_index(sim, moral_innovator_opinion)
            results['directional_influence_index'] = dii
        else:
            dii = 0.0
            results['directional_influence_index'] = dii
            warnings.warn("无法计算方向性影响指数：缺少道德创新者意见或初始状态")
        
        # 3. 身份跨越指数 (Identity Bridging Index)
        ibi = self.calculate_identity_bridging_index(sim)
        results['identity_bridging_index'] = ibi
        
        # 4. 时间稳定性指数 (Temporal Stability Index)
        if trajectory is not None:
            tsi = self.calculate_temporal_stability_index(trajectory)
            results['temporal_stability_index'] = tsi
        else:
            tsi = 0.0
            results['temporal_stability_index'] = tsi
            warnings.warn("无法计算时间稳定性指数：缺少轨迹数据")
        
        # 5. 综合议题成功指数 (Overall Issue Success Index)
        ismi = (weights['consensus_convergence'] * cci + 
                weights['directional_influence'] * dii + 
                weights['identity_bridging'] * ibi + 
                weights['temporal_stability'] * tsi)
        
        results['issue_success_index'] = ismi
        
        # 6. 极化对比指数
        polarization_index = self._get_polarization_index(sim)
        results['polarization_index'] = polarization_index
        
        # 7. 成功vs极化比率
        if polarization_index > 0:
            success_polarization_ratio = ismi / polarization_index
        else:
            success_polarization_ratio = float('inf') if ismi > 0 else 0.0
        results['success_polarization_ratio'] = success_polarization_ratio
        
        # 8. 议题传播类型判断
        results['issue_outcome_type'] = self._classify_issue_outcome(ismi, polarization_index)
        
        return results
    
    def calculate_consensus_convergence_index(self, sim: Simulation, 
                                            target_opinion: Optional[float] = None) -> float:
        """
        计算共识收敛指数 (Consensus Convergence Index, CCI)
        
        衡量意见向特定目标（如道德创新者观点）收敛的程度
        """
        opinions = sim.opinions.copy()
        
        # 排除Zealot以避免偏差
        if hasattr(sim, 'zealot_ids') and sim.zealot_ids:
            non_zealot_mask = np.ones(len(opinions), dtype=bool)
            non_zealot_mask[sim.zealot_ids] = False
            opinions = opinions[non_zealot_mask]
        
        if len(opinions) == 0:
            return 0.0
        
        # 如果没有指定目标，使用当前系统的主导意见
        if target_opinion is None:
            # 使用众数或中位数作为主导意见
            target_opinion = np.median(opinions)
        
        # 计算意见与目标的距离
        distances = np.abs(opinions - target_opinion)
        
        # 收敛指数：使用指数衰减函数，距离越小收敛度越高
        convergence_scores = np.exp(-2 * distances)
        
        # 计算整体收敛指数
        cci = np.mean(convergence_scores)
        
        return float(cci)
    
    def calculate_directional_influence_index(self, sim: Simulation, 
                                            moral_innovator_opinion: float) -> float:
        """
        计算方向性影响指数 (Directional Influence Index, DII)
        
        衡量意见变化是否朝着道德创新者的方向移动
        """
        if not hasattr(sim, 'initial_opinions'):
            warnings.warn("Simulation缺少initial_opinions属性，无法计算方向性影响指数")
            return 0.0
        
        initial_opinions = sim.initial_opinions.copy()
        current_opinions = sim.opinions.copy()
        
        # 排除Zealot以避免偏差
        if hasattr(sim, 'zealot_ids') and sim.zealot_ids:
            non_zealot_mask = np.ones(len(current_opinions), dtype=bool)
            non_zealot_mask[sim.zealot_ids] = False
            initial_opinions = initial_opinions[non_zealot_mask]
            current_opinions = current_opinions[non_zealot_mask]
        
        if len(current_opinions) == 0:
            return 0.0
        
        # 计算每个agent的意见变化
        opinion_changes = current_opinions - initial_opinions
        
        # 计算初始状态与目标的距离方向
        initial_distances = moral_innovator_opinion - initial_opinions
        
        # 避免除零
        initial_distances = np.where(np.abs(initial_distances) < 1e-6, 1e-6, initial_distances)
        
        # 方向一致性评分
        alignment_scores = []
        for i in range(len(opinion_changes)):
            change = opinion_changes[i]
            target_direction = initial_distances[i]
            
            if abs(change) < 1e-6:  # 没有变化
                score = 0.0
            elif change * target_direction > 0:  # 方向一致
                score = min(abs(change), abs(target_direction)) / abs(target_direction)
            else:  # 方向相反
                score = -min(abs(change), abs(target_direction)) / abs(target_direction)
            
            alignment_scores.append(score)
        
        dii = np.mean(alignment_scores)
        
        return float(dii)
    
    def calculate_identity_bridging_index(self, sim: Simulation) -> float:
        """
        计算身份跨越指数 (Identity Bridging Index, IBI)
        
        衡量不同身份群体是否在新议题上形成共识
        """
        opinions = sim.opinions.copy()
        identities = sim.identities.copy()
        
        # 排除Zealot以避免偏差
        if hasattr(sim, 'zealot_ids') and sim.zealot_ids:
            non_zealot_mask = np.ones(len(opinions), dtype=bool)
            non_zealot_mask[sim.zealot_ids] = False
            opinions = opinions[non_zealot_mask]
            identities = identities[non_zealot_mask]
        
        if len(opinions) == 0:
            return 0.0
        
        # 分别计算不同身份群体的意见统计
        unique_identities = np.unique(identities)
        
        if len(unique_identities) < 2:
            return 1.0  # 只有一个身份群体时，默认为完全跨越
        
        identity_groups = {}
        for identity_val in unique_identities:
            mask = identities == identity_val
            group_opinions = opinions[mask]
            if len(group_opinions) > 0:
                identity_groups[identity_val] = group_opinions
        
        if len(identity_groups) < 2:
            return 1.0
        
        # 计算各组的统计指标
        group_means = []
        group_variances = []
        
        for group_opinions in identity_groups.values():
            group_means.append(np.mean(group_opinions))
            group_variances.append(np.var(group_opinions) if len(group_opinions) > 1 else 0.0)
        
        # 组间相似性：均值差异越小，相似性越高
        mean_diff = np.std(group_means)
        similarity = 1 / (1 + 2 * mean_diff)  # 调整敏感度
        
        # 组内一致性：各组内部方差越小，一致性越高
        avg_internal_variance = np.mean(group_variances)
        consistency = 1 / (1 + 2 * avg_internal_variance)
        
        # 身份跨越指数：组间相似 + 组内一致
        ibi = 0.6 * similarity + 0.4 * consistency
        
        return float(ibi)
    
    def calculate_temporal_stability_index(self, trajectory: List[np.ndarray], 
                                         window_size: int = 50) -> float:
        """
        计算时间稳定性指数 (Temporal Stability Index, TSI)
        
        衡量共识的时间稳定性
        """
        if len(trajectory) < window_size:
            window_size = len(trajectory)
        
        if window_size < 2:
            return 0.0
        
        # 取最后一段时间的数据
        recent_trajectory = trajectory[-window_size:]
        
        # 计算每个时间步的系统均值和方差
        system_means = []
        system_variances = []
        
        for step_opinions in recent_trajectory:
            system_means.append(np.mean(step_opinions))
            system_variances.append(np.var(step_opinions))
        
        # 均值稳定性：均值变化的标准差越小，稳定性越高
        mean_stability = 1 / (1 + np.std(system_means))
        
        # 方差稳定性：方差变化的标准差越小，稳定性越高
        variance_stability = 1 / (1 + np.std(system_variances))
        
        # 综合稳定性指数
        tsi = 0.7 * mean_stability + 0.3 * variance_stability
        
        return float(tsi)
    
    def _get_polarization_index(self, sim: Simulation) -> float:
        """获取极化指数"""
        if hasattr(sim, 'calculate_polarization_index'):
            return float(sim.calculate_polarization_index())
        else:
            # 简化的极化指数计算
            opinions = sim.opinions
            
            # 排除Zealot
            if hasattr(sim, 'zealot_ids') and sim.zealot_ids:
                non_zealot_mask = np.ones(len(opinions), dtype=bool)
                non_zealot_mask[sim.zealot_ids] = False
                opinions = opinions[non_zealot_mask]
            
            if len(opinions) == 0:
                return 0.0
            
            # 使用方差作为简化的极化指标
            return float(np.var(opinions))
    
    def _classify_issue_outcome(self, ismi: float, polarization_index: float) -> str:
        """
        根据ISMI和极化指数分类议题传播结果
        
        返回值:
        - "High Success": 高成功度，低极化
        - "Moderate Success": 中等成功度
        - "Consensus Building": 共识建构中
        - "Polarizing": 引起极化
        - "High Polarization": 严重极化
        - "Failed": 传播失败
        """
        # 定义阈值
        HIGH_SUCCESS_THRESHOLD = 0.7
        MODERATE_SUCCESS_THRESHOLD = 0.5
        LOW_POLARIZATION_THRESHOLD = 0.3
        HIGH_POLARIZATION_THRESHOLD = 0.6
        
        if ismi >= HIGH_SUCCESS_THRESHOLD and polarization_index <= LOW_POLARIZATION_THRESHOLD:
            return "High Success"
        elif ismi >= MODERATE_SUCCESS_THRESHOLD and polarization_index <= LOW_POLARIZATION_THRESHOLD:
            return "Moderate Success"
        elif ismi >= MODERATE_SUCCESS_THRESHOLD and polarization_index <= HIGH_POLARIZATION_THRESHOLD:
            return "Consensus Building"
        elif ismi < MODERATE_SUCCESS_THRESHOLD and polarization_index >= HIGH_POLARIZATION_THRESHOLD:
            return "High Polarization"
        elif polarization_index >= HIGH_POLARIZATION_THRESHOLD:
            return "Polarizing"
        else:
            return "Failed"
    
    def generate_success_report(self, results: Dict[str, float]) -> str:
        """
        生成议题成功度分析报告
        
        参数:
        results -- calculate_issue_success_index的返回结果
        
        返回:
        格式化的报告字符串
        """
        report = f"""
=== 议题成功度量分析报告 ===

【核心指标】
议题成功指数 (ISMI): {results['issue_success_index']:.4f}
极化指数: {results['polarization_index']:.4f}
成功/极化比率: {results.get('success_polarization_ratio', 'N/A')}

【子指标分解】
1. 共识收敛指数: {results['consensus_convergence_index']:.4f}
   - 衡量意见向目标观点的收敛程度
   
2. 方向性影响指数: {results['directional_influence_index']:.4f}
   - 衡量变化是否朝着道德创新者方向
   
3. 身份跨越指数: {results['identity_bridging_index']:.4f}
   - 衡量是否跨越身份边界形成共识
   
4. 时间稳定性指数: {results['temporal_stability_index']:.4f}
   - 衡量共识的时间稳定性

【议题传播结果分类】
传播类型: {results['issue_outcome_type']}

【解释指南】
- ISMI > 0.7: 议题成功推广，形成广泛共识
- ISMI 0.5-0.7: 议题部分成功，需要进一步推广
- ISMI < 0.5: 议题推广效果有限

- 极化指数 > 0.6: 引起显著极化
- 极化指数 0.3-0.6: 中等程度分化
- 极化指数 < 0.3: 极化程度较低
        """
        
        return report.strip()


def analyze_issue_success_across_conditions(simulations: List[Simulation],
                                           conditions: List[str],
                                           moral_innovator_opinion: float = 1.0) -> Dict[str, Dict[str, float]]:
    """
    分析不同条件下的议题成功度
    
    参数:
    simulations -- 不同条件下的仿真结果列表
    conditions -- 对应的条件描述列表
    moral_innovator_opinion -- 道德创新者的意见值
    
    返回:
    各条件下的成功度指标字典
    """
    metrics = IssueSuccessMetrics()
    results = {}
    
    for i, (sim, condition) in enumerate(zip(simulations, conditions)):
        try:
            condition_results = metrics.calculate_issue_success_index(
                sim, moral_innovator_opinion=moral_innovator_opinion
            )
            results[condition] = condition_results
        except Exception as e:
            warnings.warn(f"分析条件 '{condition}' 时发生错误: {e}")
            results[condition] = {}
    
    return results


def find_optimal_conditions_for_success(results: Dict[str, Dict[str, float]],
                                       success_metric: str = 'issue_success_index') -> List[Tuple[str, float]]:
    """
    找到最有利于议题成功推广的条件
    
    参数:
    results -- analyze_issue_success_across_conditions的结果
    success_metric -- 用于排序的成功指标
    
    返回:
    按成功度排序的条件列表
    """
    condition_scores = []
    
    for condition, metrics in results.items():
        if success_metric in metrics:
            score = metrics[success_metric]
            condition_scores.append((condition, score))
    
    # 按成功度降序排序
    condition_scores.sort(key=lambda x: x[1], reverse=True)
    
    return condition_scores 