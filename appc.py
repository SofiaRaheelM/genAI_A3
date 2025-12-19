"""
CIFAR-10 Model Performance Dashboard
Compare training results, validation metrics, and performance across models
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CIFAR-10 Model Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .model-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .comparison-table {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px 10px 0px 0px;
        padding: 0 2rem;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADER
# ============================================================================

class PerformanceAnalyzer:
    """Load and analyze model performance data"""
    
    def __init__(self, base_dir="CIFAR10_Models"):
        self.base_dir = Path(base_dir)
        self.model_data = {}
        self.comparison_data = {}
        
        # Model colors
        self.model_colors = {
            'ViT': '#FF6B6B',
            'Hybrid CNN-MLP': '#4ECDC4',
            'ResNet': '#FFD166'
        }
        
        # Model icons
        self.model_icons = {
            'ViT': '🔄',
            'Hybrid CNN-MLP': '⚡',
            'ResNet': '🏆'
        }
    
    def load_all_model_data(self):
        """Load data from all models"""
        model_dirs = {
            'ViT': self.base_dir / 'ViT',
            'Hybrid CNN-MLP': self.base_dir / 'Hybrid_CNN_MLP',
            'ResNet': self.base_dir / 'ResNet'
        }
        
        for model_name, model_dir in model_dirs.items():
            if model_dir.exists():
                self.model_data[model_name] = self._load_model_data(model_dir, model_name)
        
        # Create comparison data
        self._create_comparison_data()
        
        return len(self.model_data)
    
    def _load_model_data(self, model_dir, model_name):
        """Load data for a specific model"""
        data = {
            'name': model_name,
            'color': self.model_colors.get(model_name, '#667eea'),
            'icon': self.model_icons.get(model_name, '🤖')
        }
        
        # Load training log
        log_path = model_dir / 'logs' / 'training_log.json'
        if log_path.exists():
            with open(log_path, 'r') as f:
                data['training_log'] = json.load(f)
        
        # Load test metrics
        metrics_path = model_dir / 'logs' / 'test_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                data['test_metrics'] = json.load(f)
        
        # Load classification report
        report_path = model_dir / 'logs' / 'classification_report.json'
        if report_path.exists():
            with open(report_path, 'r') as f:
                data['classification_report'] = json.load(f)
        
        # Try to find model files
        model_files = list(model_dir.glob('**/*.pth'))
        if model_files:
            data['model_files'] = [f.name for f in model_files[:3]]
            data['model_size_mb'] = sum(f.stat().st_size for f in model_files[:3]) / (1024 * 1024)
        
        # Try to find visualizations
        viz_files = list(model_dir.glob('**/*.png'))
        if viz_files:
            data['viz_files'] = [f.name for f in viz_files]
        
        return data
    
    def _create_comparison_data(self):
        """Create comparison metrics across models"""
        comparison = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'training_time': [],
            'best_val_acc': [],
            'final_train_acc': [],
            'final_val_acc': [],
            'model_size': [],
            'inference_speed': []
        }
        
        model_names = []
        
        for model_name, data in self.model_data.items():
            model_names.append(model_name)
            
            # Test metrics
            if 'test_metrics' in data:
                metrics = data['test_metrics']
                comparison['accuracy'].append(metrics.get('accuracy', 0))
                comparison['precision'].append(metrics.get('precision_weighted', 0))
                comparison['recall'].append(metrics.get('recall_weighted', 0))
                comparison['f1'].append(metrics.get('f1_weighted', 0))
                comparison['inference_speed'].append(metrics.get('fps', 0))
            else:
                comparison['accuracy'].append(0)
                comparison['precision'].append(0)
                comparison['recall'].append(0)
                comparison['f1'].append(0)
                comparison['inference_speed'].append(0)
            
            # Training metrics
            if 'training_log' in data:
                log = data['training_log']
                comparison['training_time'].append(log.get('training_time', 0))
                comparison['best_val_acc'].append(log.get('best_val_acc', 0))
                
                history = log.get('history_summary', {})
                comparison['final_train_acc'].append(history.get('final_train_acc', 0))
                comparison['final_val_acc'].append(history.get('final_val_acc', 0))
            else:
                comparison['training_time'].append(0)
                comparison['best_val_acc'].append(0)
                comparison['final_train_acc'].append(0)
                comparison['final_val_acc'].append(0)
            
            # Model size
            comparison['model_size'].append(data.get('model_size_mb', 0))
        
        # Create DataFrame
        self.comparison_df = pd.DataFrame(comparison, index=model_names)
        
        # Calculate rankings
        self._calculate_rankings()
    
    def _calculate_rankings(self):
        """Calculate rankings for each metric"""
        rankings = {}
        
        # Higher is better metrics
        better_metrics = ['accuracy', 'precision', 'recall', 'f1', 'best_val_acc', 
                         'final_train_acc', 'final_val_acc', 'inference_speed']
        
        # Lower is better metrics
        worse_metrics = ['training_time', 'model_size']
        
        for metric in better_metrics:
            if metric in self.comparison_df.columns:
                rankings[metric] = self.comparison_df[metric].rank(ascending=False, method='min')
        
        for metric in worse_metrics:
            if metric in self.comparison_df.columns:
                rankings[metric] = self.comparison_df[metric].rank(ascending=True, method='min')
        
        self.rankings_df = pd.DataFrame(rankings, index=self.comparison_df.index)
        
        # Calculate overall score (lower rank is better)
        if not self.rankings_df.empty:
            self.comparison_df['overall_rank'] = self.rankings_df.mean(axis=1).rank(method='min')
            self.comparison_df = self.comparison_df.sort_values('overall_rank')
    
    def get_model_summary(self, model_name):
        """Get summary for a specific model"""
        if model_name not in self.model_data:
            return None
        
        data = self.model_data[model_name]
        summary = {
            'name': model_name,
            'color': data['color'],
            'icon': data['icon']
        }
        
        # Basic info
        summary['has_training_log'] = 'training_log' in data
        summary['has_test_metrics'] = 'test_metrics' in data
        summary['has_classification_report'] = 'classification_report' in data
        summary['num_model_files'] = len(data.get('model_files', []))
        summary['num_viz_files'] = len(data.get('viz_files', []))
        
        # Test metrics
        if 'test_metrics' in data:
            metrics = data['test_metrics']
            summary['accuracy'] = metrics.get('accuracy', 'N/A')
            summary['f1_score'] = metrics.get('f1_weighted', 'N/A')
            summary['inference_speed'] = metrics.get('fps', 'N/A')
        
        # Training log
        if 'training_log' in data:
            log = data['training_log']
            summary['best_val_acc'] = log.get('best_val_acc', 'N/A')
            summary['training_time'] = log.get('training_time', 'N/A')
        
        return summary

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_performance_radar_chart(analyzer):
    """Create radar chart comparing models"""
    if analyzer.comparison_df.empty:
        return None
    
    # Select metrics for radar
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'inference_speed']
    available_metrics = [m for m in metrics if m in analyzer.comparison_df.columns]
    
    if len(available_metrics) < 3:
        return None
    
    # Normalize values (0-1)
    normalized_df = analyzer.comparison_df[available_metrics].copy()
    for metric in available_metrics:
        max_val = normalized_df[metric].max()
        if max_val > 0:
            normalized_df[metric] = normalized_df[metric] / max_val
    
    # Create radar chart
    fig = go.Figure()
    
    for model_name in normalized_df.index:
        values = normalized_df.loc[model_name].tolist()
        values = values + [values[0]]  # Close the loop
        
        angles = [n / len(available_metrics) * 2 * np.pi for n in range(len(available_metrics))]
        angles = angles + [angles[0]]  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            name=model_name,
            fill='toself',
            line_color=analyzer.model_colors.get(model_name, '#667eea')
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Performance Comparison Radar Chart',
        font=dict(size=12)
    )
    
    return fig

def create_training_history_chart(analyzer):
    """Create training history comparison chart"""
    if not analyzer.model_data:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss',
                       'Training Accuracy', 'Validation Accuracy')
    )
    
    for model_name, data in analyzer.model_data.items():
        if 'training_log' in data and 'history' in data['training_log']:
            history = data['training_log']['history']
            
            if 'train_loss' in history and len(history['train_loss']) > 0:
                # Training Loss
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['train_loss']) + 1)),
                        y=history['train_loss'],
                        name=f'{model_name} Train Loss',
                        line=dict(color=analyzer.model_colors[model_name], dash='solid'),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            if 'val_loss' in history and len(history['val_loss']) > 0:
                # Validation Loss
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['val_loss']) + 1)),
                        y=history['val_loss'],
                        name=f'{model_name} Val Loss',
                        line=dict(color=analyzer.model_colors[model_name], dash='dot'),
                        showlegend=True
                    ),
                    row=1, col=2
                )
            
            if 'train_acc' in history and len(history['train_acc']) > 0:
                # Training Accuracy
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['train_acc']) + 1)),
                        y=history['train_acc'],
                        name=f'{model_name} Train Acc',
                        line=dict(color=analyzer.model_colors[model_name], dash='solid'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            if 'val_acc' in history and len(history['val_acc']) > 0:
                # Validation Accuracy
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['val_acc']) + 1)),
                        y=history['val_acc'],
                        name=f'{model_name} Val Acc',
                        line=dict(color=analyzer.model_colors[model_name], dash='dot'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Training History Comparison",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
    
    return fig

def create_metric_comparison_bar_chart(analyzer):
    """Create bar chart comparing key metrics"""
    if analyzer.comparison_df.empty:
        return None
    
    # Select key metrics
    metrics_to_plot = ['accuracy', 'f1', 'inference_speed']
    available_metrics = [m for m in metrics_to_plot if m in analyzer.comparison_df.columns]
    
    if not available_metrics:
        return None
    
    # Create grouped bar chart
    fig = go.Figure()
    
    for idx, metric in enumerate(available_metrics):
        for model_name in analyzer.comparison_df.index:
            fig.add_trace(go.Bar(
                name=model_name,
                x=[metric],
                y=[analyzer.comparison_df.loc[model_name, metric]],
                marker_color=analyzer.model_colors.get(model_name, '#667eea'),
                showlegend=(idx == 0),  # Only show legend for first metric
                text=f"{analyzer.comparison_df.loc[model_name, metric]:.3f}",
                textposition='auto'
            ))
    
    fig.update_layout(
        barmode='group',
        title='Key Metrics Comparison',
        yaxis_title='Score',
        xaxis_title='Metric',
        height=400
    )
    
    return fig

def create_class_performance_chart(analyzer):
    """Create class-wise performance comparison"""
    all_reports = {}
    
    for model_name, data in analyzer.model_data.items():
        if 'classification_report' in data:
            report = data['classification_report']
            # Extract per-class metrics
            class_scores = {}
            for class_name in analyzer.class_names:
                if class_name in report:
                    class_scores[class_name] = report[class_name]['f1-score']
            all_reports[model_name] = class_scores
    
    if not all_reports:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_reports)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale='Viridis',
        text=df.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Class-wise F1-Score Comparison',
        xaxis_title='Model',
        yaxis_title='Class',
        height=500
    )
    
    return fig

def create_comprehensive_scoreboard(analyzer):
    """Create comprehensive scoreboard"""
    if analyzer.comparison_df.empty:
        return None
    
    # Create scoreboard DataFrame
    scoreboard = analyzer.comparison_df.copy()
    
    # Add rankings
    if not analyzer.rankings_df.empty:
        for metric in analyzer.rankings_df.columns:
            scoreboard[f'{metric}_rank'] = analyzer.rankings_df[metric]
    
    # Format values
    formatted_df = pd.DataFrame()
    for model_name in scoreboard.index:
        row = {}
        for col in scoreboard.columns:
            if 'rank' in col:
                row[col] = f"#{int(scoreboard.loc[model_name, col])}"
            elif col == 'accuracy':
                row[col] = f"{scoreboard.loc[model_name, col]:.3f}"
            elif col == 'f1':
                row[col] = f"{scoreboard.loc[model_name, col]:.3f}"
            elif col == 'training_time':
                row[col] = f"{scoreboard.loc[model_name, col]:.1f}s"
            elif col == 'model_size':
                row[col] = f"{scoreboard.loc[model_name, col]:.1f}MB"
            elif col == 'inference_speed':
                row[col] = f"{scoreboard.loc[model_name, col]:.1f}FPS"
            else:
                row[col] = scoreboard.loc[model_name, col]
        formatted_df = pd.concat([formatted_df, pd.DataFrame([row], index=[model_name])])
    
    return formatted_df

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PerformanceAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # App header
    st.markdown("""
    <div class="main-header">
        <h1>📊 CIFAR-10 Model Performance Dashboard</h1>
        <p>Comprehensive comparison of Vision Transformer, Hybrid CNN-MLP, and ResNet</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model directory
        model_dir = st.text_input("Model Directory", "CIFAR10_Models")
        
        # Load data button
        if st.button("📂 Load Performance Data", type="primary"):
            with st.spinner("Loading model data..."):
                num_loaded = analyzer.load_all_model_data()
                if num_loaded > 0:
                    st.success(f"✅ Loaded data from {num_loaded} model(s)")
                else:
                    st.error("❌ No model data found")
        
        # Model selection for detailed view
        if analyzer.model_data:
            st.divider()
            st.markdown("### 🔍 Model Details")
            
            selected_model = st.selectbox(
                "Select model for details:",
                list(analyzer.model_data.keys())
            )
            
            if selected_model:
                summary = analyzer.get_model_summary(selected_model)
                if summary:
                    st.markdown(f"#### {summary['icon']} {summary['name']}")
                    st.markdown(f"**Color:** `{summary['color']}`")
                    st.markdown(f"**Model Files:** {summary['num_model_files']}")
                    st.markdown(f"**Visualizations:** {summary['num_viz_files']}")
                    
                    if 'accuracy' in summary:
                        st.metric("Test Accuracy", f"{summary['accuracy']:.3f}")
                    if 'best_val_acc' in summary:
                        st.metric("Best Val Acc", f"{summary['best_val_acc']:.2f}%")
        
        # Analysis options
        st.divider()
        st.markdown("### 📈 Analysis Options")
        
        show_rankings = st.checkbox("Show Rankings", value=True)
        normalize_charts = st.checkbox("Normalize Charts", value=True)
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Training Comparison", "🎯 Performance", "🏆 Leaderboard"])
    
    with tab1:
        # Overview tab
        st.markdown("## 📊 Model Overview")
        
        if analyzer.model_data:
            # Display model cards
            cols = st.columns(len(analyzer.model_data))
            
            for idx, (model_name, data) in enumerate(analyzer.model_data.items()):
                with cols[idx]:
                    color = data['color']
                    icon = data['icon']
                    
                    st.markdown(f"""
                    <div class="model-section" style="border-left-color: {color};">
                        <h2>{icon} {model_name}</h2>
                        <div style="text-align: center; font-size: 2rem; margin: 1rem 0;">
                            {icon}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    if 'test_metrics' in data:
                        metrics = data['test_metrics']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                        with col2:
                            st.metric("F1-Score", f"{metrics.get('f1_weighted', 0):.3f}")
                    
                    if 'training_log' in data:
                        log = data['training_log']
                        st.metric("Best Val Acc", f"{log.get('best_val_acc', 0):.2f}%")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            
            if not analyzer.comparison_df.empty:
                # Create summary metrics
                summary_metrics = analyzer.comparison_df[['accuracy', 'f1', 'inference_speed']].copy()
                summary_metrics['model'] = summary_metrics.index
                
                # Melt for plotting
                melted_df = summary_metrics.melt(id_vars=['model'], 
                                                value_vars=['accuracy', 'f1', 'inference_speed'],
                                                var_name='metric', 
                                                value_name='score')
                
                # Create grouped bar chart
                fig = px.bar(melted_df, 
                           x='model', 
                           y='score', 
                           color='metric',
                           barmode='group',
                           title='Performance Metrics Comparison',
                           color_discrete_map={
                               'accuracy': '#FF6B6B',
                               'f1': '#4ECDC4',
                               'inference_speed': '#FFD166'
                           })
                
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👈 Load model data using the sidebar button")
    
    with tab2:
        # Training Comparison tab
        st.markdown("## 📈 Training History Comparison")
        
        if analyzer.model_data:
            # Training history chart
            history_chart = create_training_history_chart(analyzer)
            if history_chart:
                st.plotly_chart(history_chart, use_container_width=True)
            else:
                st.warning("Training history data not available")
            
            # Epoch statistics
            st.markdown("### ⏱️ Training Statistics")
            
            stats_data = []
            for model_name, data in analyzer.model_data.items():
                if 'training_log' in data and 'history_summary' in data['training_log']:
                    summary = data['training_log']['history_summary']
                    stats_data.append({
                        'Model': model_name,
                        'Epochs': summary.get('num_epochs_trained', 'N/A'),
                        'Final Train Loss': summary.get('final_train_loss', 'N/A'),
                        'Final Val Loss': summary.get('final_val_loss', 'N/A'),
                        'Final Train Acc': f"{summary.get('final_train_acc', 0):.1f}%" if summary.get('final_train_acc') else 'N/A',
                        'Final Val Acc': f"{summary.get('final_val_acc', 0):.1f}%" if summary.get('final_val_acc') else 'N/A'
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Convergence analysis
            st.markdown("### 📉 Convergence Analysis")
            
            convergence_data = []
            for model_name, data in analyzer.model_data.items():
                if 'training_log' in data and 'history' in data['training_log']:
                    history = data['training_log']['history']
                    if 'train_acc' in history and 'val_acc' in history:
                        train_acc = history['train_acc']
                        val_acc = history['val_acc']
                        
                        if len(train_acc) > 0 and len(val_acc) > 0:
                            # Calculate convergence metrics
                            convergence_score = min(val_acc) / max(train_acc) if max(train_acc) > 0 else 0
                            overfit_gap = max(train_acc) - max(val_acc)
                            
                            convergence_data.append({
                                'Model': model_name,
                                'Max Train Acc': f"{max(train_acc):.1f}%",
                                'Max Val Acc': f"{max(val_acc):.1f}%",
                                'Overfitting Gap': f"{overfit_gap:.1f}%",
                                'Convergence Score': f"{convergence_score:.3f}"
                            })
            
            if convergence_data:
                convergence_df = pd.DataFrame(convergence_data)
                st.dataframe(convergence_df, use_container_width=True, hide_index=True)
                
                # Interpretation
                st.markdown("#### 📝 Interpretation")
                st.info("""
                - **Overfitting Gap**: Difference between training and validation accuracy (lower is better)
                - **Convergence Score**: Ratio of min validation to max training accuracy (closer to 1 is better)
                """)
        
        else:
            st.info("👈 Load model data to see training comparisons")
    
    with tab3:
        # Performance tab
        st.markdown("## 🎯 Model Performance Analysis")
        
        if analyzer.model_data:
            # Performance radar chart
            radar_chart = create_performance_radar_chart(analyzer)
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
            
            # Metric comparison bar chart
            bar_chart = create_metric_comparison_bar_chart(analyzer)
            if bar_chart:
                st.plotly_chart(bar_chart, use_container_width=True)
            
            # Class performance heatmap
            class_chart = create_class_performance_chart(analyzer)
            if class_chart:
                st.plotly_chart(class_chart, use_container_width=True)
            
            # Detailed metrics table
            st.markdown("### 📋 Detailed Performance Metrics")
            
            if not analyzer.comparison_df.empty:
                # Format for display
                display_df = analyzer.comparison_df.copy()
                
                # Format percentages and add units
                for col in display_df.columns:
                    if col in ['accuracy', 'precision', 'recall', 'f1']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
                    elif col == 'inference_speed':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f} FPS")
                    elif col == 'training_time':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f} s")
                    elif col == 'model_size':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f} MB")
                
                st.dataframe(display_df, use_container_width=True)
            
            # Strengths and weaknesses analysis
            st.markdown("### 💡 Strengths & Weaknesses Analysis")
            
            if not analyzer.comparison_df.empty and not analyzer.rankings_df.empty:
                analysis_data = []
                
                for model_name in analyzer.comparison_df.index:
                    # Find best and worst metrics
                    best_metrics = analyzer.rankings_df.loc[model_name].nsmallest(2).index.tolist()
                    worst_metrics = analyzer.rankings_df.loc[model_name].nlargest(2).index.tolist()
                    
                    analysis_data.append({
                        'Model': model_name,
                        'Strengths': ', '.join(best_metrics),
                        'Weaknesses': ', '.join(worst_metrics),
                        'Best Metric': best_metrics[0] if best_metrics else 'N/A',
                        'Worst Metric': worst_metrics[0] if worst_metrics else 'N/A'
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        
        else:
            st.info("👈 Load model data to see performance analysis")
    
    with tab4:
        # Leaderboard tab
        st.markdown("## 🏆 Model Leaderboard")
        
        if analyzer.model_data and not analyzer.comparison_df.empty:
            # Create scoreboard
            scoreboard = create_comprehensive_scoreboard(analyzer)
            
            if scoreboard is not None:
                st.markdown("### 🥇 Overall Rankings")
                
                # Display ranking with colors
                for rank, (model_name, row) in enumerate(scoreboard.iterrows(), 1):
                    color = analyzer.model_colors.get(model_name, '#667eea')
                    icon = analyzer.model_icons.get(model_name, '🤖')
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="background: rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)};
                                    padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 5px solid {color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h3 style="margin: 0;">#{rank} {icon} {model_name}</h3>
                                </div>
                                <div style="font-size: 1.5rem; font-weight: bold;">
                                    Overall Rank: {int(row.get('overall_rank', rank))}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display metrics in columns
                        metric_cols = st.columns(4)
                        metrics_to_show = ['accuracy', 'f1', 'inference_speed', 'model_size']
                        
                        for idx, metric in enumerate(metrics_to_show):
                            if metric in row.index:
                                with metric_cols[idx]:
                                    st.metric(
                                        metric.title(),
                                        row[metric],
                                        f"Rank: {row.get(f'{metric}_rank', 'N/A')}"
                                    )
                
                # Detailed rankings table
                st.markdown("### 📊 Detailed Rankings")
                st.dataframe(scoreboard, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if 'overall_rank' in analyzer.comparison_df.columns:
                    best_model = analyzer.comparison_df['overall_rank'].idxmin()
                    
                    recommendations = {
                        'ViT': "Best for research and when interpretability of attention is important.",
                        'Hybrid CNN-MLP': "Best balance of performance and efficiency. Good for deployment.",
                        'ResNet': "Best overall accuracy. Recommended for production use when accuracy is critical."
                    }
                    
                    st.success(f"**Recommended Model: {best_model}**")
                    st.info(recommendations.get(best_model, "Consider your specific requirements."))
                    
                    # Use case table
                    use_cases = pd.DataFrame({
                        'Use Case': ['Maximum Accuracy', 'Fast Training', 'Efficient Inference', 'Research/Education'],
                        'Recommended Model': ['ResNet', 'ResNet (Transfer Learning)', 'Hybrid CNN-MLP', 'All Three'],
                        'Reason': ['Pretrained advantage', 'Transfer learning', 'Simpler architecture', 'Compare approaches']
                    })
                    
                    st.dataframe(use_cases, use_container_width=True, hide_index=True)
            
            # Performance trends
            st.markdown("### 📈 Performance Trends")
            
            if not analyzer.comparison_df.empty:
                # Create trend analysis
                trend_data = []
                
                for model_name in analyzer.comparison_df.index:
                    trend_data.append({
                        'Model': model_name,
                        'Accuracy Rank': analyzer.rankings_df.loc[model_name, 'accuracy'] if 'accuracy' in analyzer.rankings_df.columns else 0,
                        'Speed Rank': analyzer.rankings_df.loc[model_name, 'inference_speed'] if 'inference_speed' in analyzer.rankings_df.columns else 0,
                        'Efficiency Rank': analyzer.rankings_df.loc[model_name, 'model_size'] if 'model_size' in analyzer.rankings_df.columns else 0
                    })
                
                trend_df = pd.DataFrame(trend_data)
                
                # Create radar chart for trends
                fig = go.Figure()
                
                for idx, row in trend_df.iterrows():
                    values = [row['Accuracy Rank'], row['Speed Rank'], row['Efficiency Rank']]
                    values = values + [values[0]]  # Close loop
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=['Accuracy', 'Speed', 'Efficiency', 'Accuracy'],
                        name=row['Model'],
                        fill='toself',
                        line_color=analyzer.model_colors.get(row['Model'], '#667eea')
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, len(analyzer.model_data)])),
                    title='Trade-off Analysis (Lower Rank is Better)',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👈 Load model data to see leaderboard")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    # Set default parameters
    if 'class_names' not in st.session_state:
        st.session_state.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Run the app
    main()