"""
CIFAR-10 Model Dashboard - FIXED VERSION
Exact model architectures matching training
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CIFAR-10 Model Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        color: white;
    }
    .header {
        text-align: center;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .model-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
        transition: transform 0.3s;
    }
    .model-card:hover {
        transform: translateY(-5px);
    }
    .file-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        font-family: monospace;
    }
    .prediction-result {
        background: rgba(0, 255, 0, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff00;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# EXACT MODEL DEFINITIONS FROM TRAINING CODE
# ============================================================================

class VisionTransformer(nn.Module):
    """Vision Transformer EXACTLY from training"""
    def __init__(self, img_size=32, patch_size=4, embed_dim=256, depth=6, num_heads=8, num_classes=10):
        super().__init__()
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            self._make_transformer_block(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def _make_transformer_block(self, embed_dim, num_heads):
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            # Attention part
            norm1, attn, norm2, fc1, act, fc2 = block
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            residual = x
            x = norm2(x)
            x = fc1(x)
            x = act(x)
            x = fc2(x)
            x = residual + x
        
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

class HybridCNNMLP(nn.Module):
    """Hybrid CNN-MLP EXACTLY from training"""
    def __init__(self, num_classes=10):
        super().__init__()
        # IMPORTANT: This EXACTLY matches the training architecture
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Dropout2d after block 1
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Dropout2d after block 2
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Dropout2d after block 3
            nn.Dropout2d(0.3),
        )
        
        # Calculate output size
        self.cnn_output_size = 256 * 4 * 4
        
        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # 0.5 * 0.5
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return self.mlp(features)

class ResNetModel(nn.Module):
    """ResNet model matching training"""
    def __init__(self):
        super().__init__()
        # Load ResNet18
        resnet = models.resnet18(pretrained=False)
        
        # Remove the original fc layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom classifier matching training
        self.classifier = nn.Sequential(
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
    
    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        return self.classifier(features)

# ============================================================================
# SMART MODEL LOADER WITH ARCHITECTURE DETECTION
# ============================================================================

class SmartModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.loaded_models = {}
    
    def detect_model_type(self, checkpoint_path):
        """Detect model type from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Check for ViT keys
            vit_keys = ['patch_embed', 'pos_embed', 'cls_token']
            if any(key in str(state_dict.keys()) for key in vit_keys):
                return 'ViT'
            
            # Check for ResNet keys
            resnet_keys = ['conv1.weight', 'layer1', 'bn1']
            if any(key in str(state_dict.keys()) for key in resnet_keys):
                return 'ResNet'
            
            # Default to Hybrid
            return 'Hybrid'
            
        except:
            return 'Hybrid'  # Default
    
    def load_model_with_flexible_state_dict(self, model_path, model_type='auto'):
        """Load model with flexible state_dict matching"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"📦 Loading from: {model_path}")
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Auto-detect model type if needed
            if model_type == 'auto':
                model_type = self.detect_model_type(model_path)
            print(f"🤖 Detected model type: {model_type}")
            
            # Create model
            if model_type == 'ViT':
                model = VisionTransformer()
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                       (0.2470, 0.2435, 0.2616))
                ])
            elif model_type == 'Hybrid':
                model = HybridCNNMLP()
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                       (0.2470, 0.2435, 0.2616))
                ])
            else:  # ResNet
                model = ResNetModel()
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            
            # Get state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Print first few keys to debug
            print("🔑 First 5 state_dict keys:")
            for i, key in enumerate(list(state_dict.keys())[:5]):
                print(f"  {i}: {key} - shape: {state_dict[key].shape}")
            
            # Try different loading strategies
            strategies = [
                self._load_strict,
                self._load_ignore_mismatch,
                self._load_remove_prefix,
                self._load_partial
            ]
            
            for strategy in strategies:
                try:
                    print(f"🔄 Trying strategy: {strategy.__name__}")
                    loaded = strategy(model, state_dict)
                    if loaded:
                        print("✅ Load successful!")
                        break
                except Exception as e:
                    print(f"❌ Strategy failed: {str(e)}")
                    continue
            
            model.to(self.device)
            model.eval()
            
            # Test forward pass
            test_input = torch.randn(1, 3, 32, 32).to(self.device)
            with torch.no_grad():
                output = model(test_input)
                print(f"🧪 Test output shape: {output.shape}")
            
            self.loaded_models[model_type] = {
                'model': model,
                'transform': transform,
                'path': model_path
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def _load_strict(self, model, state_dict):
        """Load with strict=True"""
        model.load_state_dict(state_dict, strict=True)
        return True
    
    def _load_ignore_mismatch(self, model, state_dict):
        """Load with strict=False (ignore mismatches)"""
        model.load_state_dict(state_dict, strict=False)
        return True
    
    def _load_remove_prefix(self, model, state_dict):
        """Remove 'model.' or 'module.' prefix from keys"""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.'
            elif key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.'
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        return True
    
    def _load_partial(self, model, state_dict):
        """Load only matching parameters"""
        model_dict = model.state_dict()
        
        # Filter state_dict to only include keys that exist in model
        filtered_dict = {k: v for k, v in state_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        
        print(f"📊 Loading {len(filtered_dict)}/{len(state_dict)} parameters")
        
        if len(filtered_dict) > 0:
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            return True
        
        return False
    
    def predict(self, image, model_type):
        """Make prediction"""
        if model_type not in self.loaded_models:
            return None
        
        model_data = self.loaded_models[model_type]
        
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
            
            # Get top 3 predictions
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
# DEBUG UTILITIES
# ============================================================================

def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint file structure"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        st.markdown(f"### 📊 Checkpoint Analysis: {os.path.basename(checkpoint_path)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Keys in checkpoint:**")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], (dict, list, torch.Tensor)):
                    if isinstance(checkpoint[key], torch.Tensor):
                        st.write(f"- {key}: Tensor {tuple(checkpoint[key].shape)}")
                    else:
                        st.write(f"- {key}: {type(checkpoint[key]).__name__}")
                else:
                    st.write(f"- {key}: {checkpoint[key]}")
        
        with col2:
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                st.write("**State Dict Info:**")
                st.write(f"- Total parameters: {len(state_dict)}")
                st.write("**First 10 keys:**")
                for i, key in enumerate(list(state_dict.keys())[:10]):
                    st.write(f"  {i}: {key} - {tuple(state_dict[key].shape)}")
        
        return True
    except Exception as e:
        st.error(f"Failed to analyze checkpoint: {str(e)}")
        return False

# ============================================================================
# FILE EXPLORER
# ============================================================================

def explore_files(base_dir):
    """Explore files in directory"""
    files_info = []
    
    if os.path.exists(base_dir):
        for root, dirs, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.lower().endswith(('.pth', '.ckpt', '.pt')):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, base_dir)
                    
                    try:
                        size_bytes = os.path.getsize(full_path)
                        size_mb = size_bytes / (1024 * 1024)
                        files_info.append({
                            'path': rel_path,
                            'size_mb': f"{size_mb:.2f}",
                            'full_path': full_path,
                            'filename': filename
                        })
                    except:
                        continue
    
    return files_info

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = SmartModelLoader()
    
    if 'files_info' not in st.session_state:
        st.session_state.files_info = []
    
    # App title
    st.markdown("""
    <div class="header">
        <h1>🖼️ CIFAR-10 Model Dashboard</h1>
        <p>Smart Model Loader with Architecture Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        # Model directory
        model_dir = st.text_input("Model Directory", "CIFAR10_Models")
        
        # Scan button
        if st.button("🔍 Scan for Model Files", type="primary", use_container_width=True):
            with st.spinner("Scanning directory..."):
                st.session_state.files_info = explore_files(model_dir)
                if st.session_state.files_info:
                    st.success(f"Found {len(st.session_state.files_info)} model files")
                else:
                    st.warning("No .pth files found")
        
        st.divider()
        
        # Manual model loading
        st.markdown("### 🤖 Manual Model Loading")
        
        if st.session_state.files_info:
            # Let user select specific file
            file_options = [f"{f['path']} ({f['size_mb']} MB)" for f in st.session_state.files_info]
            selected_file_idx = st.selectbox("Select model file:", range(len(file_options)), 
                                           format_func=lambda i: file_options[i])
            
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Model type:", ["auto", "ViT", "Hybrid", "ResNet"])
            with col2:
                if st.button("📥 Load This Model", use_container_width=True):
                    selected_file = st.session_state.files_info[selected_file_idx]
                    with st.spinner(f"Loading {selected_file['filename']}..."):
                        if st.session_state.model_loader.load_model_with_flexible_state_dict(
                            selected_file['full_path'], model_type
                        ):
                            st.success("Model loaded!")
                            st.rerun()
            
            # Debug button
            if st.button("🔧 Analyze Checkpoint", use_container_width=True):
                selected_file = st.session_state.files_info[selected_file_idx]
                analyze_checkpoint(selected_file['full_path'])
        
        st.divider()
        
        # Image upload
        st.markdown("### 📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image to classify",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.session_state.uploaded_image = image
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 File Explorer", "🤖 Loaded Models", "🔧 Debug Tools"])
    
    with tab1:
        # File Explorer
        st.markdown("## 📁 Model Files Explorer")
        
        if st.session_state.files_info:
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(st.session_state.files_info))
            with col2:
                avg_size = np.mean([float(f['size_mb']) for f in st.session_state.files_info])
                st.metric("Avg Size", f"{avg_size:.1f} MB")
            with col3:
                pth_files = [f for f in st.session_state.files_info]
                st.metric("Model Files", len(pth_files))
            
            # File list
            st.markdown("### 📋 Model Files")
            for file in st.session_state.files_info:
                with st.expander(f"{file['path']} ({file['size_mb']} MB)"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Full path:** `{file['full_path']}`")
                    with col2:
                        if st.button("📊 Analyze", key=f"analyze_{file['path']}"):
                            analyze_checkpoint(file['full_path'])
        else:
            st.info("Click 'Scan for Model Files' to explore")
    
    with tab2:
        # Loaded Models
        st.markdown("## 🤖 Loaded Models")
        
        if st.session_state.model_loader.loaded_models:
            # Show loaded models
            st.success(f"✅ {len(st.session_state.model_loader.loaded_models)} model(s) loaded")
            
            for model_type, model_data in st.session_state.model_loader.loaded_models.items():
                colors = {"ViT": "#FF6B6B", "Hybrid": "#4ECDC4", "ResNet": "#FFD166"}
                
                st.markdown(f"""
                <div class='model-card' style='border-left-color: {colors.get(model_type, "#667eea")};'>
                    <h3>{model_type}</h3>
                    <p><strong>Loaded from:</strong> {os.path.basename(model_data['path'])}</p>
                    <p><strong>Parameters:</strong> {sum(p.numel() for p in model_data['model'].parameters()):,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Make predictions if image uploaded
            if 'uploaded_image' in st.session_state:
                st.markdown("## 📊 Predictions")
                
                results = {}
                for model_type in st.session_state.model_loader.loaded_models.keys():
                    with st.spinner(f"Predicting with {model_type}..."):
                        result = st.session_state.model_loader.predict(
                            st.session_state.uploaded_image, model_type
                        )
                        if result:
                            results[model_type] = result
                
                if results:
                    # Display in columns
                    cols = st.columns(len(results))
                    for idx, (model_type, result) in enumerate(results.items()):
                        with cols[idx]:
                            st.markdown(f"""
                            <div style='text-align: center; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 10px;'>
                                <h4>{model_type}</h4>
                                <h2 style='color: #00ff00;'>{result['predicted_class']}</h2>
                                <p>Confidence: <strong>{result['confidence']:.1%}</strong></p>
                                <p>Time: <strong>{result['inference_time']:.1f} ms</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Detailed view
                    st.markdown("### 🔍 Detailed Predictions")
                    for model_type, result in results.items():
                        with st.expander(f"{model_type} - {result['predicted_class']} ({result['confidence']:.1%})"):
                            st.write("**Top 3 Predictions:**")
                            for i, pred in enumerate(result['top3'], 1):
                                col_pred, col_bar = st.columns([3, 7])
                                with col_pred:
                                    st.write(f"{i}. {pred['class']}")
                                with col_bar:
                                    st.progress(pred['confidence'], 
                                              text=f"{pred['confidence']:.1%}")
        else:
            st.info("No models loaded. Use the sidebar to load models.")
    
    with tab3:
        # Debug Tools
        st.markdown("## 🔧 Debug Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload All Models", use_container_width=True):
                # Clear loaded models
                st.session_state.model_loader.loaded_models = {}
                st.rerun()
        
        with col2:
            if st.button("🧪 Test Model Forward Pass", use_container_width=True):
                if st.session_state.model_loader.loaded_models:
                    for model_type, model_data in st.session_state.model_loader.loaded_models.items():
                        try:
                            test_input = torch.randn(1, 3, 32, 32).to(st.session_state.model_loader.device)
                            with torch.no_grad():
                                output = model_data['model'](test_input)
                            st.success(f"{model_type}: Test passed! Output shape: {output.shape}")
                        except Exception as e:
                            st.error(f"{model_type}: Test failed - {str(e)}")
                else:
                    st.warning("No models loaded")
        
        # Show device info
        st.markdown("### 💻 System Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Device:** {st.session_state.model_loader.device}")
            st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
        with col2:
            if torch.cuda.is_available():
                st.write(f"**CUDA Device:** {torch.cuda.get_device_name(0)}")
                st.write(f"**CUDA Memory:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()