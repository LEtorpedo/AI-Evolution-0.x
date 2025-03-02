import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import yaml
from datetime import datetime
import matplotlib.font_manager as fm
import os

logger = logging.getLogger(__name__)

class EvolutionVisualizer:
    """进化过程可视化器：生成各类演化趋势图"""
    
    def __init__(self, analytics, config_path="config.yaml"):
        """初始化可视化器
        
        Args:
            analytics: EvolutionAnalytics实例
            config_path: 配置文件路径
        """
        self.analytics = analytics
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['monitor']
        
        # 初始化存储路径
        self.base_path = Path(self.config['storage']['base_path'])
        self.vis_path = self.base_path / self.config['storage']['visualizations']
        self.vis_path.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """配置绘图样式"""
        # 使用内置样式
        try:
            plt.style.use('ggplot')
        except:
            plt.style.use('default')
        
        # 设置支持中文的字体
        self._setup_chinese_font()
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
    
    def _setup_chinese_font(self):
        """设置支持中文的字体"""
        # 尝试使用系统中可能存在的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'Noto Sans CJK SC']
        
        # 检查是否有可用的中文字体
        font_found = False
        for font_name in chinese_fonts:
            font_path = None
            try:
                # 尝试查找字体
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if font_path and not font_path.endswith('DejaVuSans.ttf'):
                    plt.rcParams['font.family'] = font_name
                    logger.info(f"Using Chinese font: {font_name}")
                    font_found = True
                    break
            except:
                continue
        
        if not font_found:
            # 如果没有找到中文字体，使用英文标签
            logger.warning("No Chinese font found, using English labels instead")
            self.use_english_labels = True
        else:
            self.use_english_labels = False
    
    def plot_evolution_trends(self):
        """绘制演化趋势综合图"""
        if len(self.analytics.history) == 0:
            logger.warning("No data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 变异率和继承率趋势
        ax1 = axes[0, 0]
        self._plot_rates(ax1)
        
        # 属性数量趋势
        ax2 = axes[0, 1]
        self._plot_attributes(ax2)
        
        # 概念多样性趋势
        ax3 = axes[1, 0]
        self._plot_diversity(ax3)
        
        # 变异类型分布
        ax4 = axes[1, 1]
        self._plot_mutation_types(ax4)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.vis_path / f'evolution_trends_{timestamp}.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'])
        plt.close()
        
        logger.info(f"Evolution trends plot saved to {save_path}")
    
    def _plot_rates(self, ax):
        """绘制变异率和继承率趋势"""
        generations = self.analytics.history['generation']
        ax.plot(generations, self.analytics.history['mutation_rate'], 
                label='Mutation Rate' if self.use_english_labels else '变异率', marker='o')
        ax.plot(generations, self.analytics.history['inheritance_rate'], 
                label='Inheritance Rate' if self.use_english_labels else '继承率', marker='^')
        ax.set_title('Mutation and Inheritance Rates' if self.use_english_labels else '变异率和继承率趋势')
        ax.set_xlabel('Generation' if self.use_english_labels else '代数')
        ax.set_ylabel('Rate' if self.use_english_labels else '比率')
        ax.grid(True)
        ax.legend()
    
    def _plot_attributes(self, ax):
        """绘制属性数量趋势"""
        generations = self.analytics.history['generation']
        ax.plot(generations, self.analytics.history['avg_attributes'], 
                label='Avg Attributes' if self.use_english_labels else '平均属性数', color='green')
        ax.fill_between(generations, 
                       self.analytics.history['avg_attributes'] - self.analytics.history['avg_attributes'].std(),
                       self.analytics.history['avg_attributes'] + self.analytics.history['avg_attributes'].std(),
                       alpha=0.2, color='green')
        ax.set_title('Attribute Count Trend' if self.use_english_labels else '属性数量趋势')
        ax.set_xlabel('Generation' if self.use_english_labels else '代数')
        ax.set_ylabel('Attribute Count' if self.use_english_labels else '属性数量')
        ax.grid(True)
    
    def _plot_diversity(self, ax):
        """绘制概念多样性趋势"""
        generations = self.analytics.history['generation']
        ax.plot(generations, self.analytics.history['concept_diversity'],
                label='Concept Diversity' if self.use_english_labels else '概念多样性', color='purple')
        ax.set_title('Concept Diversity Trend' if self.use_english_labels else '概念多样性趋势')
        ax.set_xlabel('Generation' if self.use_english_labels else '代数')
        ax.set_ylabel('Diversity Index' if self.use_english_labels else '多样性指数')
        ax.grid(True)
    
    def _plot_mutation_types(self, ax):
        """绘制变异类型分布"""
        natural = self.analytics.history['natural_mutations'].sum()
        safe = self.analytics.history['safe_triggers'].sum()
        
        types = ['Natural Mutations', 'Safe Triggers'] if self.use_english_labels else ['自然变异', '保障触发']
        counts = [natural, safe]
        
        ax.bar(types, counts, color=['blue', 'red'])
        ax.set_title('Mutation Type Distribution' if self.use_english_labels else '变异类型分布')
        ax.set_ylabel('Count' if self.use_english_labels else '次数')
        
        # 添加数值标签
        for i, count in enumerate(counts):
            ax.text(i, count, str(count), ha='center', va='bottom')
