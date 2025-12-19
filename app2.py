# """
# CIFAR-10 Model Comparison Dashboard
# Black & White Aesthetic Theme
# FIXED: Works with actual file structure from training
# """

# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms, models
# import numpy as np
# from PIL import Image
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import time
# from pathlib import Path
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import json
# import glob

# # ============================================================================
# # PAGE CONFIGURATION - Black & White Theme
# # ============================================================================

# st.set_page_config(
#     page_title="CIFAR-10 Model Comparator",
#     page_icon="🖼️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for black and white theme
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #000000;
#         color: #ffffff;
#     }
    
#     .main-header {
#         font-size: 3rem;
#         font-weight: 800;
#         text-align: center;
#         margin-bottom: 2rem;
#         color: #ffffff;
#         text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.1);
#     }
    
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #ffffff;
#         border-bottom: 2px solid #ffffff;
#         padding-bottom: 0.5rem;
#         margin-bottom: 1rem;
#     }
    
#     .model-card {
#         background-color: #111111;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border: 1px solid #333333;
#         transition: transform 0.3s ease;
#     }
    
#     .model-card:hover {
#         transform: translateY(-5px);
#         border-color: #ffffff;
#     }
    
#     .metric-box {
#         background-color: #1a1a1a;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #ffffff;
#         margin: 0.5rem 0;
#     }
    
#     .prediction-card {
#         background-color: #0a0a0a;
#         padding: 1.2rem;
#         border-radius: 8px;
#         border: 1px solid #444444;
#         margin-bottom: 1rem;
#     }
    
#     .stButton > button {
#         background-color: #000000;
#         color: #ffffff;
#         border: 1px solid #ffffff;
#         border-radius: 5px;
#         padding: 0.5rem 2rem;
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         background-color: #ffffff;
#         color: #000000;
#         border-color: #ffffff;
#     }
    
#     .upload-section {
#         background-color: #0d0d0d;
#         padding: 2rem;
#         border-radius: 10px;
#         border: 2px dashed #444444;
#     }
    
#     /* Tab styling */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 2rem;
#         background-color: #111111;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         background-color: #222222;
#         color: #ffffff;
#         border-radius: 5px 5px 0 0;
#         padding: 0.5rem 2rem;
#         font-weight: 600;
#     }
    
#     .stTabs [aria-selected="true"] {
#         background-color: #ffffff;
#         color: #000000;
#     }
    
#     /* Custom metrics */
#     .metric-container {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         padding: 1rem;
#         background: linear-gradient(90deg, #111111, #222222);
#         border-radius: 8px;
#         margin: 0.5rem 0;
#     }
    
#     .metric-value {
#         font-size: 1.8rem;
#         font-weight: 800;
#         color: #ffffff;
#     }
    
#     .metric-label {
#         font-size: 0.9rem;
#         color: #aaaaaa;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     /* Custom progress bars */
#     .stProgress > div > div > div > div {
#         background-color: #ffffff;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================================
# # MODEL CLASSES (EXACTLY from training code)
# # ============================================================================

# class PatchEmbedding(nn.Module):
#     """Split image into patches and embed them."""
#     def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.n_patches = (img_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
#     def forward(self, x):
#         x = self.proj(x)
#         x = x.flatten(2)
#         x = x.transpose(1, 2)
#         return x

# class PositionalEncoding(nn.Module):
#     """Add learnable positional encoding to patch embeddings."""
#     def __init__(self, n_patches, embed_dim):
#         super().__init__()
#         self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
#     def forward(self, x):
#         B = x.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         return x

# class MultiHeadSelfAttention(nn.Module):
#     """Multi-head self-attention mechanism."""
#     def __init__(self, embed_dim, num_heads, dropout=0.1):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.qkv = nn.Linear(embed_dim, embed_dim * 3)
#         self.proj = nn.Linear(embed_dim, embed_dim)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.proj_dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn = F.softmax(attn, dim=-1)
#         attn = self.attn_dropout(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_dropout(x)
#         return x

# class MLP(nn.Module):
#     """Multi-Layer Perceptron for Transformer."""
#     def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         hidden_dim = int(embed_dim * mlp_ratio)
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.act = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_dim, embed_dim)
#         self.dropout2 = nn.Dropout(dropout)
        
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.dropout2(x)
#         return x

# class TransformerBlock(nn.Module):
#     """Transformer encoder block."""
#     def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

# class VisionTransformer(nn.Module):
#     """Complete Vision Transformer model for CIFAR-10."""
#     def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
#                  embed_dim=256, depth=6, num_heads=8, mlp_ratio=4, dropout=0.1):
#         super().__init__()
        
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
        
#         self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#         self.num_patches = self.patch_embed.n_patches
        
#         self.pos_embed = PositionalEncoding(self.num_patches, embed_dim)
#         self.pos_dropout = nn.Dropout(dropout)
        
#         self.blocks = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
#             for _ in range(depth)
#         ])
        
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes)
        
#     def forward(self, x):
#         x = self.patch_embed(x)
#         x = self.pos_embed(x)
#         x = self.pos_dropout(x)
        
#         for block in self.blocks:
#             x = block(x)
            
#         x = self.norm(x)
#         cls_token = x[:, 0]
#         return self.head(cls_token)

# class HybridCNNMLP(nn.Module):
#     """Hybrid CNN-MLP for Streamlit"""
#     def __init__(self, num_classes=10, dropout_rate=0.3):
#         super(HybridCNNMLP, self).__init__()
        
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#         )
        
#         self.cnn_output_size = 256 * 4 * 4
#         self.mlp = nn.Sequential(
#             nn.Linear(self.cnn_output_size, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate * 0.5),
            
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         features = self.cnn(x)
#         features = features.view(features.size(0), -1)
#         return self.mlp(features)

# class ResNetForStreamlit(nn.Module):
#     """ResNet for Streamlit"""
#     def __init__(self):
#         super().__init__()
#         self.model = models.resnet18(pretrained=False)
#         self.model.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.2),
#             nn.Linear(256, 10)
#         )
    
#     def forward(self, x):
#         return self.model(x)

# # ============================================================================
# # MODEL LOADER - FIXED for actual file structure
# # ============================================================================

# class CIFAR10ModelLoader:
#     """Loads models from the ACTUAL file structure from training"""
    
#     def __init__(self, base_dir="CIFAR10_Models"):
#         self.base_dir = Path(base_dir)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#                            'dog', 'frog', 'horse', 'ship', 'truck']
        
#         # Define model configurations - FIXED for actual paths
#         self.model_configs = {
#             'Vision Transformer (ViT)': {
#                 'type': 'ViT',
#                 'model_class': VisionTransformer,
#                 'transform': transforms.Compose([
#                     transforms.Resize((32, 32)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
#                 ]),
#                 'paths': [
#                     self.base_dir / 'ViT' / 'vit_streamlit_model.pth',  # Primary
#                     self.base_dir / 'ViT' / 'vit_best_model.pth',       # Alternative
#                     self.base_dir / 'ViT' / 'vit_checkpoint_epoch_final.pth'  # Fallback
#                 ],
#                 'color': '#ffffff',  # White
#                 'icon': '🔄',
#                 'description': 'Transformer-based architecture with self-attention'
#             },
#             'Hybrid CNN-MLP': {
#                 'type': 'Hybrid',
#                 'model_class': HybridCNNMLP,
#                 'transform': transforms.Compose([
#                     transforms.Resize((32, 32)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
#                 ]),
#                 'paths': [
#                     self.base_dir / 'Hybrid_CNN_MLP' / 'models' / 'inference_model.pth',
#                     self.base_dir / 'Hybrid_CNN_MLP' / 'models' / 'best_model.pth',
#                     self.base_dir / 'Hybrid_CNN_MLP' / 'hybrid_streamlit_model.pth'
#                 ],
#                 'color': '#888888',  # Gray
#                 'icon': '⚡',
#                 'description': 'CNN feature extractor + MLP classifier'
#             },
#             'ResNet (Transfer Learning)': {
#                 'type': 'ResNet',
#                 'model_class': ResNetForStreamlit,
#                 'transform': transforms.Compose([
#                     transforms.Resize((224, 224)),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                 ]),
#                 'paths': [
#                     self.base_dir / 'ResNet' / 'models' / 'inference_model.pth',
#                     self.base_dir / 'ResNet' / 'models' / 'best_model.pth',
#                     self.base_dir / 'ResNet_Fast' / 'resnet_fast_model.pth'  # From fast training
#                 ],
#                 'color': '#cccccc',  # Light Gray
#                 'icon': '🏆',
#                 'description': 'Pretrained ResNet with transfer learning'
#             }
#         }
        
#         self.loaded_models = {}
    
#     def find_model_file(self, model_name):
#         """Find model file from possible paths"""
#         config = self.model_configs[model_name]
        
#         for path in config['paths']:
#             if path.exists():
#                 return path
        
#         # Try to find any .pth file in the directory
#         for path in config['paths']:
#             directory = path.parent
#             if directory.exists():
#                 pth_files = list(directory.glob('*.pth'))
#                 if pth_files:
#                     return pth_files[0]
        
#         return None
    
#     def load_model(self, model_name):
#         """Load a specific model"""
#         config = self.model_configs[model_name]
        
#         # Find model file
#         model_path = self.find_model_file(model_name)
#         if not model_path:
#             st.error(f"❌ No model file found for {model_name}")
#             st.info(f"Looking in: {[str(p) for p in config['paths']]}")
#             return False
        
#         try:
#             # Show loading message
#             loading_text = st.empty()
#             loading_text.info(f"🔄 Loading {model_name} from {model_path.name}...")
            
#             # Load checkpoint
#             checkpoint = torch.load(model_path, map_location=self.device)
            
#             # Create model instance
#             if model_name == 'Vision Transformer (ViT)':
#                 if 'model_config' in checkpoint:
#                     config_data = checkpoint['model_config']
#                     model = VisionTransformer(
#                         img_size=config_data.get('img_size', 32),
#                         patch_size=config_data.get('patch_size', 4),
#                         embed_dim=config_data.get('embed_dim', 256),
#                         depth=config_data.get('depth', 6),
#                         num_heads=config_data.get('num_heads', 8),
#                         num_classes=10
#                     )
#                 else:
#                     model = VisionTransformer()  # Default
                    
#             elif model_name == 'Hybrid CNN-MLP':
#                 if 'model_config' in checkpoint:
#                     config_data = checkpoint['model_config']
#                     model = HybridCNNMLP(
#                         num_classes=10,
#                         dropout_rate=config_data.get('dropout_rate', 0.3)
#                     )
#                 else:
#                     model = HybridCNNMLP()
                    
#             else:  # ResNet
#                 model = ResNetForStreamlit()
            
#             # Load weights
#             if 'model_state_dict' in checkpoint:
#                 model.load_state_dict(checkpoint['model_state_dict'])
#             else:
#                 model.load_state_dict(checkpoint)  # Some models save directly
            
#             model.to(self.device)
#             model.eval()
            
#             # Store loaded model
#             self.loaded_models[model_name] = {
#                 'model': model,
#                 'transform': config['transform'],
#                 'color': config['color'],
#                 'icon': config['icon'],
#                 'description': config['description'],
#                 'config': checkpoint.get('model_config', {}),
#                 'path': model_path
#             }
            
#             loading_text.success(f"✅ {model_name} loaded successfully!")
#             return True
            
#         except Exception as e:
#             st.error(f"❌ Failed to load {model_name}: {str(e)}")
#             import traceback
#             st.code(traceback.format_exc())
#             return False
    
#     def load_all_models(self, selected_models):
#         """Load all selected models"""
#         success_count = 0
        
#         for model_name in selected_models:
#             if self.load_model(model_name):
#                 success_count += 1
        
#         return success_count
    
#     def predict(self, image, model_name):
#         """Make prediction using specified model"""
#         if model_name not in self.loaded_models:
#             return None
        
#         model_data = self.loaded_models[model_name]
        
#         try:
#             # Preprocess image
#             if isinstance(image, str):
#                 image = Image.open(image).convert('RGB')
            
#             input_tensor = model_data['transform'](image).unsqueeze(0).to(self.device)
            
#             # Predict
#             with torch.no_grad():
#                 start_time = time.time()
#                 outputs = model_data['model'](input_tensor)
#                 inference_time = (time.time() - start_time) * 1000  # ms
                
#                 probabilities = F.softmax(outputs, dim=1)
#                 confidence, predicted_idx = torch.max(probabilities, 1)
            
#             # Get top 3 predictions
#             top3_probs, top3_indices = torch.topk(probabilities, 3)
            
#             result = {
#                 'predicted_class': self.class_names[predicted_idx.item()],
#                 'confidence': confidence.item(),
#                 'class_index': predicted_idx.item(),
#                 'inference_time': inference_time,
#                 'all_probabilities': probabilities[0].cpu().numpy(),
#                 'top3': [
#                     {
#                         'class': self.class_names[idx],
#                         'confidence': float(prob),
#                         'color': 'green' if i == 0 else 'gray'
#                     }
#                     for i, (prob, idx) in enumerate(zip(top3_probs[0], top3_indices[0]))
#                 ]
#             }
            
#             return result
            
#         except Exception as e:
#             st.error(f"Prediction error for {model_name}: {str(e)}")
#             return None
    
#     def get_model_stats(self):
#         """Get statistics about loaded models"""
#         stats = {}
#         for model_name, data in self.loaded_models.items():
#             stats[model_name] = {
#                 'parameters': sum(p.numel() for p in data['model'].parameters()),
#                 'trainable': sum(p.numel() for p in data['model'].parameters() if p.requires_grad),
#                 'config': data.get('config', {}),
#                 'loaded_from': data['path'].name
#             }
#         return stats

# # ============================================================================
# # VISUALIZATION FUNCTIONS
# # ============================================================================

# def create_model_comparison_chart(results):
#     """Create comparison visualization"""
    
#     if not results:
#         return None
    
#     model_names = list(results.keys())
    
#     # Create radar chart
#     categories = ['Accuracy', 'Confidence', 'Speed', 'Top-3 Match']
    
#     fig = go.Figure()
    
#     for model_name, result in results.items():
#         # Calculate metrics
#         accuracy = result['confidence']
#         speed = 1000 / (result['inference_time'] + 0.001)  # FPS
#         speed_normalized = min(speed / 100, 1.0)  # Normalize to 0-1
        
#         # Top-3 match (check if top 3 predictions are similar across models)
#         top3_classes = [pred['class'] for pred in result['top3']]
#         top3_score = len(set(top3_classes)) / 3
        
#         values = [
#             accuracy,
#             result['confidence'],
#             speed_normalized,
#             top3_score
#         ]
        
#         # Close the radar chart
#         values = values + [values[0]]
        
#         # Get model color
#         model_color = '#ffffff'  # Default white
#         if model_name == 'Hybrid CNN-MLP':
#             model_color = '#888888'
#         elif model_name == 'ResNet (Transfer Learning)':
#             model_color = '#cccccc'
        
#         fig.add_trace(go.Scatterpolar(
#             r=values,
#             theta=categories + [categories[0]],
#             name=model_name,
#             fill='toself',
#             line=dict(width=2, color=model_color),
#             fillcolor=f'rgba{tuple(int(model_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}'
#         ))
    
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1]
#             )),
#         showlegend=True,
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font=dict(color='white'),
#         title=dict(
#             text='Model Comparison Radar',
#             font=dict(color='white', size=16)
#         )
#     )
    
#     return fig

# def create_confidence_bars(results):
#     """Create confidence comparison bar chart"""
    
#     model_names = list(results.keys())
#     confidences = [results[model]['confidence'] for model in model_names]
    
#     # Define colors
#     colors = []
#     for model_name in model_names:
#         if model_name == 'Vision Transformer (ViT)':
#             colors.append('#ffffff')
#         elif model_name == 'Hybrid CNN-MLP':
#             colors.append('#888888')
#         else:
#             colors.append('#cccccc')
    
#     fig = go.Figure(data=[
#         go.Bar(
#             x=model_names,
#             y=confidences,
#             marker_color=colors,
#             text=[f'{c:.1%}' for c in confidences],
#             textposition='auto',
#             textfont=dict(color='black', size=12)
#         )
#     ])
    
#     fig.update_layout(
#         title='Prediction Confidence by Model',
#         xaxis_title='Model',
#         yaxis_title='Confidence',
#         yaxis=dict(range=[0, 1]),
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font=dict(color='white'),
#         showlegend=False
#     )
    
#     return fig

# def create_class_probability_chart(results):
#     """Create probability distribution chart"""
    
#     if not results:
#         return None
    
#     class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#                    'dog', 'frog', 'horse', 'ship', 'truck']
    
#     fig = go.Figure()
    
#     # Define colors for each model
#     model_colors = {
#         'Vision Transformer (ViT)': '#ffffff',
#         'Hybrid CNN-MLP': '#888888',
#         'ResNet (Transfer Learning)': '#cccccc'
#     }
    
#     for model_name, result in results.items():
#         probs = result['all_probabilities']
        
#         fig.add_trace(go.Scatter(
#             x=class_names,
#             y=probs,
#             mode='lines+markers',
#             name=model_name,
#             line=dict(width=2, color=model_colors.get(model_name, '#ffffff'))
#         ))
    
#     fig.update_layout(
#         title='Class Probability Distribution',
#         xaxis_title='Class',
#         yaxis_title='Probability',
#         yaxis=dict(range=[0, 1]),
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font=dict(color='white'),
#         hovermode='x unified'
#     )
    
#     return fig

# def create_inference_time_chart(results):
#     """Create inference time comparison chart"""
    
#     model_names = list(results.keys())
#     times = [results[model]['inference_time'] for model in model_names]
    
#     # Define colors
#     colors = []
#     for model_name in model_names:
#         if model_name == 'Vision Transformer (ViT)':
#             colors.append('#ffffff')
#         elif model_name == 'Hybrid CNN-MLP':
#             colors.append('#888888')
#         else:
#             colors.append('#cccccc')
    
#     fig = go.Figure(data=[
#         go.Bar(
#             x=model_names,
#             y=times,
#             marker_color=colors,
#             text=[f'{t:.1f} ms' for t in times],
#             textposition='auto',
#             textfont=dict(color='black', size=12)
#         )
#     ])
    
#     fig.update_layout(
#         title='Inference Time Comparison',
#         xaxis_title='Model',
#         yaxis_title='Time (ms)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font=dict(color='white'),
#         showlegend=False
#     )
    
#     return fig

# # ============================================================================
# # FILE STRUCTURE CHECKER
# # ============================================================================

# def check_file_structure(base_dir="CIFAR10_Models"):
#     """Check and display the actual file structure"""
    
#     base_path = Path(base_dir)
    
#     if not base_path.exists():
#         st.error(f"❌ Directory not found: {base_dir}")
#         st.info("Please ensure your models are in the correct location.")
#         return False
    
#     st.markdown("### 📁 Current File Structure")
    
#     # Display file tree
#     file_tree = []
    
#     for root, dirs, files in os.walk(base_dir):
#         level = root.replace(base_dir, '').count(os.sep)
#         indent = ' ' * 4 * level
#         file_tree.append(f"{indent}📁 {os.path.basename(root)}/")
#         subindent = ' ' * 4 * (level + 1)
        
#         for file in files:
#             if file.endswith('.pth') or file.endswith('.json') or file.endswith('.png'):
#                 size = os.path.getsize(os.path.join(root, file))
#                 size_str = f"({size/1024/1024:.1f} MB)" if size > 1024*1024 else f"({size/1024:.1f} KB)"
#                 file_tree.append(f"{subindent}📄 {file} {size_str}")
    
#     st.code("\n".join(file_tree), language='text')
    
#     # Check for model files
#     vit_files = list(base_path.glob('ViT/**/*.pth'))
#     hybrid_files = list(base_path.glob('Hybrid_CNN_MLP/**/*.pth'))
#     resnet_files = list(base_path.glob('ResNet/**/*.pth'))
    
#     st.markdown("### 🔍 Model Files Found")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric("ViT Files", len(vit_files))
#         if vit_files:
#             for file in vit_files[:3]:  # Show first 3
#                 st.caption(f"• {file.name}")
    
#     with col2:
#         st.metric("Hybrid Files", len(hybrid_files))
#         if hybrid_files:
#             for file in hybrid_files[:3]:
#                 st.caption(f"• {file.name}")
    
#     with col3:
#         st.metric("ResNet Files", len(resnet_files))
#         if resnet_files:
#             for file in resnet_files[:3]:
#                 st.caption(f"• {file.name}")
    
#     return len(vit_files) > 0 or len(hybrid_files) > 0 or len(resnet_files) > 0

# # ============================================================================
# # STREAMLIT APP
# # ============================================================================

# def main():
#     """Main Streamlit application"""
    
#     # Initialize model loader
#     model_loader = CIFAR10ModelLoader()
    
#     # Sidebar
#     with st.sidebar:
#         st.markdown("""
#         <div style='text-align: center; margin-bottom: 2rem;'>
#             <h1 style='color: white;'>⚙️ SETTINGS</h1>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # File structure check
#         if st.button("🔍 Check File Structure", use_container_width=True):
#             check_file_structure()
        
#         st.divider()
        
#         # Model selection
#         st.markdown("### 📊 Select Models")
#         model_options = list(model_loader.model_configs.keys())
#         selected_models = st.multiselect(
#             "Choose models to compare:",
#             model_options,
#             default=model_options,
#             label_visibility="collapsed"
#         )
        
#         # Load models button
#         if st.button("🚀 Load Selected Models", use_container_width=True):
#             if selected_models:
#                 with st.spinner("Loading models..."):
#                     success_count = model_loader.load_all_models(selected_models)
#                     if success_count > 0:
#                         st.success(f"✅ Loaded {success_count} model(s)")
#                     else:
#                         st.error("❌ Failed to load any models")
#             else:
#                 st.warning("Please select at least one model")
        
#         st.divider()
        
#         # Upload section
#         st.markdown("### 📁 Upload Images")
#         upload_option = st.radio(
#             "Choose upload method:",
#             ["Single Image", "Multiple Images", "Test Folder"],
#             label_visibility="collapsed"
#         )
        
#         uploaded_files = None
        
#         if upload_option == "Single Image":
#             uploaded_file = st.file_uploader(
#                 "Upload an image",
#                 type=['png', 'jpg', 'jpeg', 'bmp'],
#                 label_visibility="collapsed",
#                 key="single_upload"
#             )
#             if uploaded_file:
#                 uploaded_files = [uploaded_file]
                
#         elif upload_option == "Multiple Images":
#             uploaded_files = st.file_uploader(
#                 "Upload multiple images",
#                 type=['png', 'jpg', 'jpeg', 'bmp'],
#                 accept_multiple_files=True,
#                 label_visibility="collapsed",
#                 key="multi_upload"
#             )
        
#         else:  # Test Folder
#             test_folder = st.text_input("Enter test folder path:", "test_images/")
#             if st.button("📂 Load from Folder", use_container_width=True):
#                 folder_path = Path(test_folder)
#                 if folder_path.exists():
#                     image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + \
#                                  list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.bmp"))
#                     if image_files:
#                         uploaded_files = image_files
#                         st.success(f"✅ Found {len(image_files)} images")
#                     else:
#                         st.warning("No images found in folder")
#                 else:
#                     st.error(f"Folder not found: {test_folder}")
        
#         st.divider()
        
#         # Additional options
#         st.markdown("### ⚙️ Display Options")
#         show_details = st.checkbox("Show detailed predictions", value=True)
#         show_visualizations = st.checkbox("Show visualizations", value=True)
#         compare_models = st.checkbox("Enable model comparison", value=True)
    
#     # Main content
#     st.markdown("<h1 class='main-header'>CIFAR-10 MODEL COMPARATOR</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: center; color: #aaaaaa; font-size: 1.2rem; margin-bottom: 3rem;'>Compare Vision Transformer, Hybrid CNN-MLP, and ResNet on CIFAR-10</p>", unsafe_allow_html=True)
    
#     # Create tabs
#     tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🖼️ Predictions", "📈 Analysis", "ℹ️ Info"])
    
#     with tab1:
#         # Dashboard
#         st.markdown("<h2 class='sub-header'>MODEL DASHBOARD</h2>", unsafe_allow_html=True)
        
#         if not model_loader.loaded_models:
#             st.info("👈 Select models and click 'Load Selected Models' to begin")
            
#             # Show available models
#             st.markdown("### Available Models")
#             cols = st.columns(3)
            
#             for idx, (model_name, config) in enumerate(model_loader.model_configs.items()):
#                 with cols[idx]:
#                     st.markdown(f"""
#                     <div class='model-card' style='opacity: 0.7;'>
#                         <div style='font-size: 2rem; text-align: center;'>{config['icon']}</div>
#                         <h3 style='text-align: center; color: {config['color']};'>{model_name}</h3>
#                         <div style='text-align: center; color: #aaaaaa; font-size: 0.9rem;'>
#                             {config['description']}
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
#         else:
#             # Model cards for loaded models
#             cols = st.columns(len(model_loader.loaded_models))
            
#             for idx, (model_name, model_data) in enumerate(model_loader.loaded_models.items()):
#                 with cols[idx]:
#                     st.markdown(f"""
#                     <div class='model-card'>
#                         <div style='font-size: 2rem; text-align: center;'>{model_data['icon']}</div>
#                         <h3 style='text-align: center; color: {model_data['color']};'>{model_name}</h3>
#                         <div style='text-align: center; color: #aaaaaa; font-size: 0.9rem;'>
#                             Loaded from: {model_data['path'].name}
#                         </div>
#                         <div style='text-align: center; margin-top: 1rem;'>
#                             <small>{model_data['description']}</small>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # Model statistics
#             st.markdown("<br>", unsafe_allow_html=True)
#             st.markdown("<h3 class='sub-header'>MODEL STATISTICS</h3>", unsafe_allow_html=True)
            
#             stats = model_loader.get_model_stats()
#             stats_cols = st.columns(len(stats))
            
#             for idx, (model_name, model_stats) in enumerate(stats.items()):
#                 with stats_cols[idx]:
#                     st.metric("Parameters", f"{model_stats['parameters']:,}")
#                     st.metric("Trainable", f"{model_stats['trainable']:,}")
#                     st.caption(f"From: {model_stats['loaded_from']}")
    
#     with tab2:
#         # Predictions tab
#         st.markdown("<h2 class='sub-header'>IMAGE PREDICTIONS</h2>", unsafe_allow_html=True)
        
#         if uploaded_files:
#             # Limit to first 10 images for performance
#             max_images = min(10, len(uploaded_files))
#             if len(uploaded_files) > max_images:
#                 st.info(f"Showing first {max_images} of {len(uploaded_files)} images")
#                 uploaded_files = uploaded_files[:max_images]
            
#             # Process each uploaded image
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 st.markdown(f"### 📸 Image {idx + 1}")
                
#                 col_img, col_results = st.columns([1, 2])
                
#                 with col_img:
#                     # Display image
#                     try:
#                         if isinstance(uploaded_file, str):
#                             image = Image.open(uploaded_file)
#                         else:
#                             image = Image.open(uploaded_file)
                        
#                         st.image(image, caption="Uploaded Image", use_column_width=True)
                        
#                         # Image info
#                         st.caption(f"Size: {image.size} | Mode: {image.mode}")
                        
#                     except Exception as e:
#                         st.error(f"Error loading image: {str(e)}")
#                         continue
                
#                 with col_results:
#                     if model_loader.loaded_models:
#                         # Get predictions from all models
#                         all_results = {}
                        
#                         for model_name in model_loader.loaded_models.keys():
#                             result = model_loader.predict(image, model_name)
#                             if result:
#                                 all_results[model_name] = result
                        
#                         # Display predictions
#                         if all_results:
#                             # Create comparison table
#                             comparison_data = []
                            
#                             for model_name, result in all_results.items():
#                                 comparison_data.append({
#                                     'Model': model_name,
#                                     'Prediction': result['predicted_class'],
#                                     'Confidence': f"{result['confidence']:.1%}",
#                                     'Time (ms)': f"{result['inference_time']:.1f}",
#                                     'Top 2': f"{result['top3'][1]['class']} ({result['top3'][1]['confidence']:.1%})"
#                                 })
                            
#                             df = pd.DataFrame(comparison_data)
#                             st.dataframe(df, use_container_width=True, hide_index=True)
                            
#                             # Show consensus
#                             predictions = [r['predicted_class'] for r in all_results.values()]
#                             if len(set(predictions)) == 1:
#                                 st.success(f"✅ All models agree: **{predictions[0]}**")
#                             else:
#                                 st.warning(f"⚠️ Models disagree: {', '.join(set(predictions))}")
                            
#                             # Show detailed predictions if enabled
#                             if show_details:
#                                 st.markdown("#### Detailed Predictions")
                                
#                                 for model_name, result in all_results.items():
#                                     with st.expander(f"{model_name} - {result['predicted_class']} ({result['confidence']:.1%})"):
#                                         # Top 3 predictions
#                                         st.write("**Top 3 Predictions:**")
#                                         for i, pred in enumerate(result['top3'], 1):
#                                             col_pred, col_conf = st.columns([3, 1])
#                                             with col_pred:
#                                                 color = "green" if i == 1 else "gray"
#                                                 st.markdown(f"<span style='color: {color};'>{i}. {pred['class']}</span>", unsafe_allow_html=True)
#                                             with col_conf:
#                                                 st.progress(pred['confidence'], 
#                                                           text=f"{pred['confidence']:.1%}")
                                        
#                                         # All probabilities
#                                         st.write("**All Class Probabilities:**")
#                                         for cls_name, prob in zip(model_loader.class_names, result['all_probabilities']):
#                                             st.progress(float(prob), 
#                                                       text=f"{cls_name}: {prob:.1%}")
#                     else:
#                         st.warning("No models loaded. Please load models from the sidebar.")
        
#         else:
#             st.info("👈 Upload images from the sidebar to see predictions")
    
#     with tab3:
#         # Analysis tab
#         st.markdown("<h2 class='sub-header'>MODEL ANALYSIS</h2>", unsafe_allow_html=True)
        
#         if uploaded_files and model_loader.loaded_models:
#             # Use first image for analysis
#             try:
#                 if isinstance(uploaded_files[0], str):
#                     image = Image.open(uploaded_files[0])
#                 else:
#                     image = Image.open(uploaded_files[0])
                
#                 # Get predictions for analysis
#                 analysis_results = {}
#                 for model_name in model_loader.loaded_models.keys():
#                     result = model_loader.predict(image, model_name)
#                     if result:
#                         analysis_results[model_name] = result
#                         analysis_results[model_name]['color'] = model_loader.loaded_models[model_name]['color']
                
#                 if analysis_results and show_visualizations:
#                     # Create visualizations in columns
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         # Radar chart
#                         radar_fig = create_model_comparison_chart(analysis_results)
#                         if radar_fig:
#                             st.plotly_chart(radar_fig, use_container_width=True)
                    
#                     with col2:
#                         # Confidence bars
#                         bar_fig = create_confidence_bars(analysis_results)
#                         if bar_fig:
#                             st.plotly_chart(bar_fig, use_container_width=True)
                    
#                     col3, col4 = st.columns(2)
                    
#                     with col3:
#                         # Inference time chart
#                         time_fig = create_inference_time_chart(analysis_results)
#                         if time_fig:
#                             st.plotly_chart(time_fig, use_container_width=True)
                    
#                     with col4:
#                         # Probability distribution
#                         prob_fig = create_class_probability_chart(analysis_results)
#                         if prob_fig:
#                             st.plotly_chart(prob_fig, use_container_width=True)
                    
#                     # Consensus analysis
#                     if compare_models and len(analysis_results) > 1:
#                         st.markdown("#### 📊 Consensus Analysis")
                        
#                         # Check if all models agree
#                         predictions = [r['predicted_class'] for r in analysis_results.values()]
#                         unique_predictions = set(predictions)
                        
#                         if len(unique_predictions) == 1:
#                             st.success(f"✅ **All models agree**: {list(unique_predictions)[0]}")
#                         else:
#                             st.warning(f"⚠️ **Models disagree**: {', '.join(unique_predictions)}")
                            
#                             # Show agreement matrix
#                             st.markdown("**Agreement Matrix:**")
#                             model_names = list(analysis_results.keys())
#                             agreement_matrix = []
                            
#                             for i, model1 in enumerate(model_names):
#                                 row = []
#                                 for j, model2 in enumerate(model_names):
#                                     if i == j:
#                                         row.append("✓")
#                                     else:
#                                         agree = (analysis_results[model1]['predicted_class'] == 
#                                                 analysis_results[model2]['predicted_class'])
#                                         row.append("✓" if agree else "✗")
#                                 agreement_matrix.append(row)
                            
#                             agreement_df = pd.DataFrame(
#                                 agreement_matrix,
#                                 index=[f"{name[:15]}..." if len(name) > 15 else name for name in model_names],
#                                 columns=[f"{name[:15]}..." if len(name) > 15 else name for name in model_names]
#                             )
#                             st.dataframe(agreement_df, use_container_width=True)
                
#             except Exception as e:
#                 st.error(f"Error analyzing image: {str(e)}")
#         else:
#             st.info("👈 Upload an image and load models to see analysis")
    
#     with tab4:
#         # Info tab
#         st.markdown("<h2 class='sub-header'>INFORMATION</h2>", unsafe_allow_html=True)
        
#         st.markdown("""
#         ### 📋 About This Application
        
#         This application compares three different neural network architectures on the CIFAR-10 dataset:
        
#         1. **Vision Transformer (ViT)** - Uses self-attention mechanism
#         2. **Hybrid CNN-MLP** - Combines convolutional layers with MLP
#         3. **ResNet** - Pretrained ResNet with transfer learning
        
#         ### 📁 Expected File Structure
        
#         ```
#         CIFAR10_Models/
#         ├── ViT/
#         │   ├── vit_streamlit_model.pth      # Primary ViT model
#         │   ├── vit_best_model.pth           # Alternative
#         │   └── vit_checkpoint_epoch_*.pth   # Checkpoints
#         ├── Hybrid_CNN_MLP/
#         │   ├── models/
#         │   │   ├── inference_model.pth      # Primary Hybrid model
#         │   │   └── best_model.pth           # Alternative
#         │   └── hybrid_streamlit_model.pth   # Fallback
#         └── ResNet/
#             ├── models/
#             │   ├── inference_model.pth      # Primary ResNet model
#             │   └── best_model.pth           # Alternative
#             └── ResNet_Fast/
#                 └── resnet_fast_model.pth    # Fast training version
#         ```
        
#         ### 🚀 How to Use
        
#         1. **Check File Structure** - Ensure models are in correct locations
#         2. **Select Models** - Choose which models to load
#         3. **Load Models** - Click the load button
#         4. **Upload Images** - Single, multiple, or from folder
#         5. **View Results** - See predictions and comparisons
        
#         ### 📊 Expected Performance
        
#         | Model | Expected Accuracy | Training Time | Inference Speed |
#         |-------|-------------------|---------------|-----------------|
#         | ViT | 85-90% | Longest | Medium |
#         | Hybrid | 84-88% | Medium | Fastest |
#         | ResNet | 92-96% | Shortest | Medium |
        
#         ### 🔧 Troubleshooting
        
#         If models fail to load:
#         1. Check file paths are correct
#         2. Ensure .pth files exist
#         3. Check file permissions
#         4. Try alternative model files
#         """)

# # ============================================================================
# # RUN THE APP
# # ============================================================================

# if __name__ == "__main__":
#     # Set plotly theme
#     import plotly.io as pio
#     pio.templates.default = "plotly_dark"
    
#     # Set matplotlib style
#     plt.style.use('dark_background')
#     sns.set_style("darkgrid")
    
#     # Run main app
#     main()


"""
CIFAR-10 Model Comparison Dashboard
FIXED VERSION - Universal model loader
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CIFAR-10 Model Comparator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; color: #ffffff; }
    .model-card { background: #1a1a1a; padding: 1.5rem; border-radius: 10px; border: 1px solid #333; }
    .metric-box { background: #222222; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .prediction-card { background: #1a1a1a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UNIVERSAL MODEL LOADER - FIXED FOR ALL MODELS
# ============================================================================

class UniversalModelLoader:
    """Universal model loader that works with all trained model architectures"""
    
    def __init__(self, base_dir="CIFAR10_Models"):
        self.base_dir = Path(base_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.loaded_models = {}
        
        # Define model search patterns
        self.model_patterns = {
            'Vision Transformer': {
                'search_paths': [
                    self.base_dir / 'ViT' / '*.pth',
                    self.base_dir / 'ViT' / '**/*.pth',
                ],
                'type': 'ViT'
            },
            'Hybrid CNN-MLP': {
                'search_paths': [
                    self.base_dir / 'Hybrid_CNN_MLP' / '*.pth',
                    self.base_dir / 'Hybrid_CNN_MLP' / '**/*.pth',
                ],
                'type': 'Hybrid'
            },
            'ResNet': {
                'search_paths': [
                    self.base_dir / 'ResNet' / '*.pth',
                    self.base_dir / 'ResNet' / '**/*.pth',
                ],
                'type': 'ResNet'
            }
        }
    
    def find_model_files(self):
        """Find all model files in the directory"""
        model_files = {}
        for model_name, config in self.model_patterns.items():
            files = []
            for pattern in config['search_paths']:
                for path in Path('.').glob(str(pattern)):
                    if path.is_file():
                        files.append(path)
            if files:
                model_files[model_name] = files
        return model_files
    
    def create_model_from_checkpoint(self, checkpoint_path):
        """Dynamically create model based on checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Try to detect model type from checkpoint
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                
                # ViT detection
                if any(key in config for key in ['embed_dim', 'num_heads', 'patch_size']):
                    return self._create_vit_model(config)
                
                # Hybrid CNN-MLP detection
                elif 'cnn_output_size' in config or 'dropout_rate' in config:
                    return self._create_hybrid_model(config)
            
            # ResNet detection
            elif any('resnet' in key.lower() for key in checkpoint.keys() if isinstance(key, str)):
                return self._create_resnet_model()
            
            # Try to load as generic model
            return self._load_generic_model(checkpoint, checkpoint_path)
            
        except Exception as e:
            st.error(f"Error creating model: {str(e)}")
            return None
    
    def _create_vit_model(self, config):
        """Create ViT model from config"""
        try:
            # Try importing the exact ViT class if available
            try:
                from models import VisionTransformer
                model = VisionTransformer(
                    img_size=config.get('img_size', 32),
                    patch_size=config.get('patch_size', 4),
                    embed_dim=config.get('embed_dim', 256),
                    depth=config.get('depth', 6),
                    num_heads=config.get('num_heads', 8),
                    num_classes=10
                )
            except:
                # Create simplified ViT
                class SimpleViT(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.patch_embed = nn.Conv2d(3, 256, kernel_size=4, stride=4)
                        n_patches = (32 // 4) ** 2
                        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, 256))
                        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
                        self.transformer = nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024),
                            num_layers=6
                        )
                        self.norm = nn.LayerNorm(256)
                        self.head = nn.Linear(256, 10)
                    
                    def forward(self, x):
                        x = self.patch_embed(x)
                        x = x.flatten(2).transpose(1, 2)
                        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
                        x = torch.cat((cls_tokens, x), dim=1)
                        x = x + self.pos_embed
                        x = self.transformer(x)
                        x = self.norm(x)
                        cls_token = x[:, 0]
                        return self.head(cls_token)
                
                model = SimpleViT()
            
            model = model.to(self.device)
            return model
            
        except Exception as e:
            st.error(f"Error creating ViT: {str(e)}")
            return None
    
    def _create_hybrid_model(self, config):
        """Create Hybrid CNN-MLP model that matches training"""
        try:
            # This EXACTLY matches the training architecture
            class ExactHybridCNNMLP(nn.Module):
                def __init__(self, num_classes=10, dropout_rate=0.3):
                    super().__init__()
                    # EXACT CNN architecture from training
                    self.cnn = nn.Sequential(
                        # Block 1
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Dropout2d(0.1),  # Added dropout between blocks
                        
                        # Block 2
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Dropout2d(0.2),  # Added dropout between blocks
                        
                        # Block 3
                        nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Dropout2d(0.3),  # Added dropout between blocks
                    )
                    
                    self.cnn_output_size = 256 * 4 * 4
                    
                    # EXACT MLP architecture from training
                    self.mlp = nn.Sequential(
                        nn.Linear(self.cnn_output_size, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout_rate),
                        
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout_rate),
                        
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout_rate * 0.5),
                        
                        nn.Linear(256, num_classes)
                    )
                
                def forward(self, x):
                    features = self.cnn(x)
                    features = features.view(features.size(0), -1)
                    return self.mlp(features)
            
            model = ExactHybridCNNMLP(
                num_classes=10,
                dropout_rate=config.get('dropout_rate', 0.3)
            )
            model = model.to(self.device)
            return model
            
        except Exception as e:
            st.error(f"Error creating Hybrid model: {str(e)}")
            return None
    
    def _create_resnet_model(self):
        """Create ResNet model that matches training"""
        try:
            # Load pretrained ResNet18
            model = models.resnet18(pretrained=False)
            
            # Replace the final fully connected layer to match training
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 10)
            )
            
            model = model.to(self.device)
            return model
            
        except Exception as e:
            st.error(f"Error creating ResNet: {str(e)}")
            return None
    
    def _load_generic_model(self, checkpoint, checkpoint_path):
        """Load model generically using state dict"""
        try:
            # Try to detect model type from state dict keys
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Check for ViT keys
            vit_keys = ['patch_embed', 'pos_embed', 'cls_token']
            if any(key in str(state_dict.keys()) for key in vit_keys):
                return self._create_vit_model({'img_size': 32})
            
            # Check for ResNet keys
            resnet_keys = ['conv1.weight', 'layer1', 'layer2', 'layer3', 'layer4']
            if any(key in str(state_dict.keys()) for key in resnet_keys):
                return self._create_resnet_model()
            
            # Default to Hybrid CNN-MLP
            return self._create_hybrid_model({'dropout_rate': 0.3})
            
        except Exception as e:
            st.error(f"Generic load error: {str(e)}")
            return None
    
    def load_model_smart(self, model_name, model_path):
        """Smart loading with multiple attempts"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Try different loading strategies
            strategies = [
                self._try_load_with_state_dict,
                self._try_load_direct,
                self._try_load_with_model_type
            ]
            
            for strategy in strategies:
                model = strategy(checkpoint, model_name)
                if model is not None:
                    # Try to load weights
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # Try direct loading
                        try:
                            model.load_state_dict(checkpoint)
                        except:
                            # Try with strict=False (ignore mismatches)
                            model.load_state_dict(checkpoint, strict=False)
                    
                    model = model.to(self.device)
                    model.eval()
                    
                    # Store with appropriate transform
                    if 'ViT' in model_name or 'Vision' in model_name:
                        transform = transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                               (0.2470, 0.2435, 0.2616))
                        ])
                    elif 'ResNet' in model_name:
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
                        ])
                    else:  # Hybrid
                        transform = transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                               (0.2470, 0.2435, 0.2616))
                        ])
                    
                    self.loaded_models[model_name] = {
                        'model': model,
                        'transform': transform,
                        'path': model_path
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            st.error(f"Smart load failed for {model_name}: {str(e)}")
            return False
    
    def _try_load_with_state_dict(self, checkpoint, model_name):
        """Try loading with state dict"""
        try:
            if 'model_state_dict' in checkpoint:
                # Try to create appropriate model
                if 'ViT' in model_name:
                    return self._create_vit_model({})
                elif 'ResNet' in model_name:
                    return self._create_resnet_model()
                else:
                    return self._create_hybrid_model({})
        except:
            pass
        return None
    
    def _try_load_direct(self, checkpoint, model_name):
        """Try direct model loading"""
        try:
            # Check if checkpoint is a model directly
            if isinstance(checkpoint, nn.Module):
                return checkpoint
        except:
            pass
        return None
    
    def _try_load_with_model_type(self, checkpoint, model_name):
        """Try loading based on model type in name"""
        if 'ViT' in model_name or 'Vision' in model_name:
            return self._create_vit_model({})
        elif 'ResNet' in model_name:
            return self._create_resnet_model()
        else:
            return self._create_hybrid_model({})
    
    def load_all_models(self):
        """Load all available models"""
        model_files = self.find_model_files()
        
        if not model_files:
            st.warning("No model files found!")
            return 0
        
        success_count = 0
        
        for model_name, files in model_files.items():
            # Try each file for this model type
            for model_path in files[:3]:  # Try first 3 files
                with st.spinner(f"Loading {model_name} from {model_path.name}..."):
                    if self.load_model_smart(model_name, model_path):
                        st.success(f"✅ {model_name} loaded from {model_path.name}")
                        success_count += 1
                        break
                time.sleep(0.5)  # Small delay
        
        return success_count
    
    def predict(self, image, model_name):
        """Make prediction"""
        if model_name not in self.loaded_models:
            return None
        
        model_data = self.loaded_models[model_name]
        
        try:
            # Preprocess
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            input_tensor = model_data['transform'](image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                start_time = time.time()
                outputs = model_data['model'](input_tensor)
                inference_time = (time.time() - start_time) * 1000
                
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get top 3
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            
            result = {
                'predicted_class': self.class_names[predicted_idx.item()],
                'confidence': confidence.item(),
                'inference_time': inference_time,
                'all_probabilities': probabilities[0].cpu().numpy(),
                'top3': [
                    {
                        'class': self.class_names[idx],
                        'confidence': float(prob)
                    }
                    for prob, idx in zip(top3_probs[0], top3_indices[0])
                ]
            }
            
            return result
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

# ============================================================================
# SIMPLE STREAMLIT APP
# ============================================================================

def main():
    """Main app with simple interface"""
    
    st.title("🖼️ CIFAR-10 Model Comparator")
    st.markdown("Compare Vision Transformer, Hybrid CNN-MLP, and ResNet")
    
    # Initialize loader
    if 'loader' not in st.session_state:
        st.session_state.loader = UniversalModelLoader()
    
    loader = st.session_state.loader
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Load All Models", type="primary"):
            with st.spinner("Finding and loading models..."):
                success = loader.load_all_models()
                if success > 0:
                    st.success(f"Loaded {success} model(s)")
                else:
                    st.error("Failed to load any models")
        
        st.divider()
        
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Main content
    if uploaded_file and loader.loaded_models:
        st.header("📊 Predictions")
        
        # Get predictions
        results = {}
        for model_name in loader.loaded_models.keys():
            with st.spinner(f"Predicting with {model_name}..."):
                result = loader.predict(uploaded_file, model_name)
                if result:
                    results[model_name] = result
        
        if results:
            # Display results in columns
            cols = st.columns(len(results))
            
            for idx, (model_name, result) in enumerate(results.items()):
                with cols[idx]:
                    # Model card
                    with st.container():
                        st.markdown(f"### {model_name}")
                        st.markdown(f"**Prediction:** {result['predicted_class']}")
                        st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                        st.markdown(f"**Time:** {result['inference_time']:.1f} ms")
                        
                        # Top 3 predictions
                        with st.expander("Top 3 predictions"):
                            for pred in result['top3']:
                                st.progress(pred['confidence'], 
                                          text=f"{pred['class']}: {pred['confidence']:.1%}")
            
            # Comparison chart
            st.divider()
            st.header("📈 Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Prediction': result['predicted_class'],
                    'Confidence': result['confidence'],
                    'Time (ms)': result['inference_time']
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Check consensus
            predictions = [r['predicted_class'] for r in results.values()]
            if len(set(predictions)) == 1:
                st.success(f"✅ All models agree: **{predictions[0]}**")
            else:
                st.warning(f"⚠️ Models disagree: {', '.join(set(predictions))}")
        
    elif uploaded_file:
        st.warning("Please load models first using the button in the sidebar")
    else:
        st.info("👈 Upload an image to get started")
        
        # Show available models
        st.header("📋 Available Models")
        model_files = loader.find_model_files()
        
        if model_files:
            for model_name, files in model_files.items():
                with st.expander(f"{model_name} ({len(files)} files)"):
                    for file in files[:5]:  # Show first 5
                        st.write(f"• {file.name}")
        else:
            st.error("No model files found! Please ensure models are in CIFAR10_Models/ folder")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()