import logging
from pathlib import Path
import yaml
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EvolutionReporter:
    """进化报告生成器：生成详细的进化分析报告"""
    
    def __init__(self, analytics, config_path="config.yaml"):
        """初始化报告生成器
        
        Args:
            analytics: EvolutionAnalytics实例
            config_path: 配置文件路径
        """
        self.analytics = analytics
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['monitor']
        
        # 初始化存储路径
        self.report_path = Path(self.config['storage']['base_path']) / self.config['storage']['reports']
        self.report_path.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, report_type='full'):
        """生成进化报告
        
        Args:
            report_type: 报告类型 ('full', 'summary', 'metrics')
        """
        if len(self.analytics.history) == 0:
            return "No evolution data available for reporting."
        
        report_methods = {
            'full': self._generate_full_report,
            'summary': self._generate_summary_report,
            'metrics': self._generate_metrics_report
        }
        
        report = report_methods.get(report_type, self._generate_full_report)()
        self._save_report(report, report_type)
        return report
    
    def _generate_full_report(self):
        """生成完整报告"""
        stats = self.analytics.get_statistics()
        
        report = []
        report.append("=== 进化分析完整报告 ===")
        report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 基础统计
        report.append("\n1. 基础统计")
        report.append(f"总代数: {stats['total_generations']}")
        report.append(f"平均变异率: {stats['avg_mutation_rate']:.2%}")
        report.append(f"平均继承率: {stats['avg_inheritance_rate']:.2%}")
        
        # 变异分析
        report.append("\n2. 变异分析")
        report.append(f"自然变异总数: {stats['total_mutations']}")
        report.append(f"保障触发总数: {stats['total_safe_triggers']}")
        report.append(f"每代平均变异次数: {stats['total_mutations'] / stats['total_generations']:.2f}")
        
        # 属性分析
        report.append("\n3. 属性分析")
        report.append(f"最大属性数: {stats['max_attributes']:.1f}")
        
        # 概念多样性分析
        report.append("\n4. 概念多样性分析")
        diversity_trend = stats['concept_diversity_trend']
        report.append(f"初始多样性: {diversity_trend[0]:.2f}")
        report.append(f"最终多样性: {diversity_trend[-1]:.2f}")
        report.append(f"多样性变化率: {(diversity_trend[-1] - diversity_trend[0]) / len(diversity_trend):.2%}/代")
        
        # 属性生命周期分析
        report.append("\n5. 属性生命周期分析")
        long_lived = [attr for attr, stats in self.analytics.attribute_stats.items() 
                     if stats['last_seen'] - stats['first_seen'] >= 3]
        report.append(f"长寿属性数量: {len(long_lived)}")
        report.append(f"属性总类型数: {len(self.analytics.attribute_stats)}")
        
        return "\n".join(report)
    
    def _generate_summary_report(self):
        """生成摘要报告"""
        stats = self.analytics.get_statistics()
        
        summary = {
            'total_generations': stats['total_generations'],
            'mutation_efficiency': stats['total_mutations'] / stats['total_generations'],
            'inheritance_stability': stats['avg_inheritance_rate'],
            'concept_diversity_growth': (stats['concept_diversity_trend'][-1] - 
                                       stats['concept_diversity_trend'][0]) / len(stats['concept_diversity_trend'])
        }
        
        return json.dumps(summary, indent=2)
    
    def _generate_metrics_report(self):
        """生成指标报告"""
        history = self.analytics.history
        return history.describe().to_string()
    
    def _save_report(self, report, report_type):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_path / f'evolution_report_{report_type}_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Evolution report saved to {report_file}")
