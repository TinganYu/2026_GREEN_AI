"""
Pareto 前沿視覺化

建立多目標優化結果的互動式視覺化。
"""

import os
import logging
from typing import Dict, List, Any
import optuna
import optuna.visualization as optuna_vis

logger = logging.getLogger("ParetoVisualizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class ParetoVisualizer:
    """視覺化 Pareto 前沿和權衡"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化視覺化器

        Args:
            config: 配置字典
        """
        self.config = config
        self.viz_config = config['output']['visualization']

        logger.info("Pareto 視覺化器已初始化")

    def create_visualizations(self, pareto_trials: List[Dict[str, Any]],
                            satisfying_trials: List[Dict[str, Any]],
                            output_dir: str,
                            study: optuna.Study = None) -> Dict[str, str]:
        """
        建立所有視覺化

        Args:
            pareto_trials: Pareto 最優試驗
            satisfying_trials: 滿足目標的試驗
            output_dir: 保存圖表的目錄
            study: Optuna Study 物件（用於進階視覺化，可選）

        Returns:
            {plot_type: file_path} 字典
        """
        if not self.viz_config['enabled']:
            logger.info("視覺化已停用")
            return {}

        logger.info("正在建立視覺化...")

        plot_files = {}

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots

            # 3D Pareto 前沿
            if '3d_pareto' in self.viz_config['plot_types']:
                fig = self._plot_3d_pareto(pareto_trials, satisfying_trials, go)
                path = os.path.join(output_dir, 'pareto_3d.html')
                fig.write_html(path)
                plot_files['3d_pareto'] = path
                logger.info(f"3D Pareto 圖表已保存至：{path}")

            # 平行座標圖
            if 'parallel_coordinates' in self.viz_config['plot_types']:
                fig = self._plot_parallel_coordinates(pareto_trials, px)
                path = os.path.join(output_dir, 'parallel_coordinates.html')
                fig.write_html(path)
                plot_files['parallel_coordinates'] = path
                logger.info(f"平行座標圖已保存至：{path}")

            # 權衡矩陣
            if 'trade_off_matrix' in self.viz_config['plot_types']:
                fig = self._plot_trade_off_matrix(pareto_trials, go, make_subplots)
                path = os.path.join(output_dir, 'trade_off_matrix.html')
                fig.write_html(path)
                plot_files['trade_off_matrix'] = path
                logger.info(f"權衡矩陣已保存至：{path}")

            # ===== 新增：優化過程視覺化 =====
            # 只有當提供 Study 物件時才生成（Optuna-based optimizers）
            if study is not None:
                logger.info("建立優化過程視覺化（使用 Optuna Study）...")

                # 優化歷史
                if 'optimization_history' in self.viz_config['plot_types']:
                    fig = self._plot_optimization_history(study)
                    path = os.path.join(output_dir, 'optimization_history.html')
                    fig.write_html(path)
                    plot_files['optimization_history'] = path
                    logger.info(f"優化歷史圖已保存至：{path}")

                # 超參數重要性
                if 'param_importances' in self.viz_config['plot_types']:
                    fig = self._plot_param_importances(study)
                    path = os.path.join(output_dir, 'param_importances.html')
                    fig.write_html(path)
                    plot_files['param_importances'] = path
                    logger.info(f"超參數重要性圖已保存至：{path}")

                # 試驗時間線與狀態分佈
                if 'trial_timeline' in self.viz_config['plot_types']:
                    fig = self._plot_trial_timeline(study)
                    path = os.path.join(output_dir, 'trial_timeline.html')
                    fig.write_html(path)
                    plot_files['trial_timeline'] = path
                    logger.info(f"試驗時間線圖已保存至：{path}")

                # 超體積收斂
                if 'hypervolume_history' in self.viz_config['plot_types']:
                    fig = self._plot_hypervolume_history(study)
                    path = os.path.join(output_dir, 'hypervolume_history.html')
                    fig.write_html(path)
                    plot_files['hypervolume_history'] = path
                    logger.info(f"超體積收斂圖已保存至：{path}")

            else:
                logger.info("未提供 Study 物件，跳過優化過程視覺化")

        except ImportError:
            logger.warning("Plotly 未安裝。請使用以下指令安裝：pip install plotly")
            return {}
        except Exception as e:
            logger.error(f"建立視覺化失敗：{e}")
            return {}

        return plot_files

    def _plot_3d_pareto(self, pareto_trials: List[Dict], satisfying_trials: List[Dict], go) -> Any:
        """建立 Pareto 前沿的 3D 散點圖"""

        # 提取資料（轉換為百分比）
        acc_changes = [t['objectives']['accuracy_change'] * 100 for t in pareto_trials]
        gpu_changes = [t['objectives']['gpu_peak_change'] * 100 for t in pareto_trials]
        latency_changes = [t['objectives']['latency_change'] * 100 for t in pareto_trials]

        # 標記滿足目標的解決方案
        colors = ['green' if t in satisfying_trials else 'blue' for t in pareto_trials]
        methods = [t['config']['method'] for t in pareto_trials]

        # 建立散點圖
        fig = go.Figure(data=[go.Scatter3d(
            x=acc_changes,
            y=gpu_changes,
            z=latency_changes,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.8
            ),
            text=[f"Method: {m}" for m in methods],
            hovertemplate='<b>%{text}</b><br>' +
                         'Accuracy Change: %{x:+.2f}%<br>' +
                         'GPU Peak Change: %{y:+.2f}%<br>' +
                         'Latency Change: %{z:+.2f}%<extra></extra>'
        )])

        fig.update_layout(
            title='3D Pareto Frontier',
            scene=dict(
                xaxis_title='準確率變化 (%) [正=上升↑, 負=下降↓]',
                yaxis_title='GPU 峰值變化 (%) [負=減少↓, 正=增加↑]',
                zaxis_title='延遲變化 (%) [負=減少↓, 正=增加↑]'
            ),
            width=900,
            height=700
        )

        return fig

    def _plot_parallel_coordinates(self, pareto_trials: List[Dict], px) -> Any:
        """建立平行座標圖"""

        # 準備資料
        data = []
        for i, t in enumerate(pareto_trials):
            data.append({
                'trial': i,
                'method': t['config']['method'],
                'accuracy_change': t['objectives']['accuracy_change'] * 100,
                # 反轉 GPU 和延遲軸（負值表示降低，現在顯示在上方）
                'gpu_peak_reduction': -t['objectives']['gpu_peak_change'] * 100,
                'latency_reduction': -t['objectives']['latency_change'] * 100,
                'satisfies_targets': 1 if t['satisfies_targets'] else 0
            })

        # 建立圖表
        fig = px.parallel_coordinates(
            data,
            dimensions=['accuracy_change', 'gpu_peak_reduction', 'latency_reduction'],
            color='satisfies_targets',
            labels={
                'accuracy_change': '準確率變化 (%)<br>[↑正=改善]',
                'gpu_peak_reduction': 'GPU 峰值節省 (%)<br>[↑正=節省]',
                'latency_reduction': '延遲減少 (%)<br>[↑正=加速]'
            },
            title='Pareto Solutions - Parallel Coordinates<br><sub>註：GPU和延遲軸已反轉，向上表示改善</sub>',
            color_continuous_scale=px.colors.diverging.Tealrose
        )

        return fig

    def _plot_trade_off_matrix(self, pareto_trials: List[Dict], go, make_subplots) -> Any:
        """建立 2x2 權衡矩陣"""

        # 提取資料
        acc_changes = [t['objectives']['accuracy_change'] * 100 for t in pareto_trials]
        gpu_changes = [t['objectives']['gpu_peak_change'] * 100 for t in pareto_trials]
        latency_changes = [t['objectives']['latency_change'] * 100 for t in pareto_trials]
        methods = [t['config']['method'] for t in pareto_trials]

        # 建立子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy vs GPU', 'Accuracy vs Latency',
                          'GPU vs Latency', 'Method Distribution')
        )

        # Accuracy vs GPU
        fig.add_trace(
            go.Scatter(x=acc_changes, y=gpu_changes, mode='markers',
                      marker=dict(size=8), text=methods,
                      name='', showlegend=False),
            row=1, col=1
        )
        fig.update_xaxes(title_text="準確率變化 (%)", row=1, col=1)
        fig.update_yaxes(title_text="GPU 峰值變化 (%)", row=1, col=1)

        # Accuracy vs Latency
        fig.add_trace(
            go.Scatter(x=acc_changes, y=latency_changes, mode='markers',
                      marker=dict(size=8), text=methods,
                      name='', showlegend=False),
            row=1, col=2
        )
        fig.update_xaxes(title_text="準確率變化 (%)", row=1, col=2)
        fig.update_yaxes(title_text="延遲變化 (%)", row=1, col=2)

        # GPU vs Latency
        fig.add_trace(
            go.Scatter(x=gpu_changes, y=latency_changes, mode='markers',
                      marker=dict(size=8), text=methods,
                      name='', showlegend=False),
            row=2, col=1
        )
        fig.update_xaxes(title_text="GPU 峰值變化 (%)", row=2, col=1)
        fig.update_yaxes(title_text="延遲變化 (%)", row=2, col=1)

        # Method distribution
        method_counts = {}
        for m in methods:
            method_counts[m] = method_counts.get(m, 0) + 1

        fig.add_trace(
            go.Bar(x=list(method_counts.keys()), y=list(method_counts.values()),
                  name='', showlegend=False),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Method", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        fig.update_layout(height=800, title_text="Trade-off Analysis Matrix")

        return fig

    def _plot_optimization_history(self, study: optuna.Study) -> Any:
        """
        建立優化歷史圖，顯示目標值隨試驗的變化

        Uses Optuna's plot_optimization_history() to show how each objective
        improves (or degrades) across trials. This helps verify the optimization
        algorithm is exploring the search space effectively.

        Args:
            study: Optuna Study 物件

        Returns:
            Plotly figure object
        """
        try:
            # Optuna's built-in function for multi-objective optimization
            # Shows all 3 objectives: accuracy_change, gpu_peak_change, latency_change
            fig = optuna_vis.plot_optimization_history(study)

            # Customize title and layout
            fig.update_layout(
                title='Optimization History - Objective Values Across Trials',
                xaxis_title='Trial Number',
                yaxis_title='Objective Value',
                height=600,
                width=1000,
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.warning(f"Failed to create optimization history plot: {e}")
            # Return empty figure if visualization fails
            import plotly.graph_objects as go
            return go.Figure()

    def _plot_param_importances(self, study: optuna.Study) -> Any:
        """
        建立超參數重要性圖，分析哪些量化參數影響最大

        Uses Optuna's plot_param_importances() to show which hyperparameters
        (method, bits, group_size, etc.) have the most impact on objectives.
        Importance is calculated using fANOVA (functional ANOVA).

        Args:
            study: Optuna Study 物件

        Returns:
            Plotly figure object
        """
        try:
            # Optuna automatically selects target objective in multi-objective studies
            # For multi-objective, it analyzes importance for all objectives
            fig = optuna_vis.plot_param_importances(study)

            fig.update_layout(
                title='Hyperparameter Importance - Impact on Objectives',
                height=600,
                width=1000
            )

            return fig

        except Exception as e:
            logger.warning(f"Failed to create parameter importance plot: {e}")
            logger.info("Parameter importance requires multiple completed trials")
            import plotly.graph_objects as go
            return go.Figure()

    def _plot_trial_timeline(self, study: optuna.Study) -> Any:
        """
        建立試驗時間線，顯示完成/剪枝/失敗試驗的分佈

        Creates a combined visualization:
        - Left: Timeline showing when each trial ran (using Optuna's plot_timeline)
        - Right: Bar chart showing distribution of trial states

        Args:
            study: Optuna Study 物件

        Returns:
            Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from collections import Counter

            # Collect trial states
            trial_states = [trial.state.name for trial in study.trials]
            state_counts = Counter(trial_states)

            # Create figure with 2 subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Trial Timeline', 'State Distribution'),
                specs=[[{"type": "scatter"}, {"type": "bar"}]],
                column_widths=[0.6, 0.4]
            )

            # Left: Timeline (use Optuna's built-in)
            try:
                timeline_fig = optuna_vis.plot_timeline(study)
                for trace in timeline_fig.data:
                    fig.add_trace(trace, row=1, col=1)
            except:
                logger.warning("Could not create timeline subplot")

            # Right: State distribution bar chart
            colors = {'COMPLETE': 'green', 'PRUNED': 'orange', 'FAIL': 'red'}
            fig.add_trace(
                go.Bar(
                    x=list(state_counts.keys()),
                    y=list(state_counts.values()),
                    text=list(state_counts.values()),
                    textposition='auto',
                    marker=dict(
                        color=[colors.get(state, 'gray') for state in state_counts.keys()]
                    )
                ),
                row=1, col=2
            )

            fig.update_layout(
                title='Trial States - Timeline and Distribution',
                height=500,
                width=1200,
                showlegend=True
            )

            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="State", row=1, col=2)
            fig.update_yaxes(title_text="Trial", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)

            return fig

        except Exception as e:
            logger.warning(f"Failed to create trial timeline plot: {e}")
            import plotly.graph_objects as go
            return go.Figure()

    def _plot_hypervolume_history(self, study: optuna.Study) -> Any:
        """
        建立超體積收斂圖，追蹤 Pareto 前沿品質改進

        Hypervolume indicator measures the quality of the Pareto frontier.
        Higher hypervolume = better coverage of objective space.
        Shows improvement over trials.

        Args:
            study: Optuna Study 物件

        Returns:
            Plotly figure object
        """
        try:
            # Optuna has built-in hypervolume history calculation
            # Requires reference point (defined in config: objectives[*].reference_point)
            fig = optuna_vis.plot_hypervolume_history(study)

            fig.update_layout(
                title='Hypervolume Convergence - Pareto Frontier Quality Over Time',
                xaxis_title='Trial Number',
                yaxis_title='Hypervolume Indicator',
                height=600,
                width=1000
            )

            return fig

        except Exception as e:
            logger.warning(f"Failed to create hypervolume history plot: {e}")
            logger.info("Hypervolume calculation requires trials with finite objective values")
            logger.info("Trials with infinity values (failed/pruned) are excluded automatically")
            import plotly.graph_objects as go
            return go.Figure()
