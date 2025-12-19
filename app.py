# app.py — FULL STREAMLIT VERSION (Gradio → Streamlit)

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy

import streamlit as st
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# ============================================================
# SETUP & CONFIG
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path("umc005-corpus")
ARTIFACT_DIR = Path("artifacts")
SPM_DIR = ARTIFACT_DIR / "spm"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MAX_TOKENS = 160
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

whitespace_re = re.compile(r"\s+")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def normalize_whitespace(text: str) -> str:
    return whitespace_re.sub(" ", text.strip())

def clean_en(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("\u200c", "")
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\[\]\-/ ]", " ", text)
    return normalize_whitespace(text.lower())

def encode_sentence(proc: spm.SentencePieceProcessor, text: str, max_len: int) -> List[int]:
    tokens = proc.encode(text, out_type=int)
    tokens = tokens[: max_len - 2]
    return [BOS_ID] + tokens + [EOS_ID]

def ids_to_sentence(token_ids: List[int], processor: spm.SentencePieceProcessor) -> str:
    filtered = []
    for idx in token_ids:
        if idx == EOS_ID:
            break
        if idx in {PAD_ID, BOS_ID}:
            continue
        filtered.append(idx)
    if not filtered:
        return ""
    return processor.decode(filtered)

# ============================================================
# MODEL ARCHITECTURES (same as training)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.linear2.out_features)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, mask, src_key_padding_mask)
        return self.norm(output)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, cross_attn = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt, cross_attn

class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.linear2.out_features)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        cross_attn_map = None
        for mod in self.layers:
            output, cross_attn = mod(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if cross_attn is not None:
                cross_attn_map = cross_attn
        return self.norm(output), cross_attn_map

class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_ID)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.register_buffer("causal_mask", torch.triu(torch.ones(MAX_TOKENS, MAX_TOKENS), diagonal=1).bool())

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        return self.causal_mask[:size, :size]

    def encode(self, src, src_pad_mask):
        src = self.src_embed(src) * self.scale
        src = self.pos_encoder(self.dropout(src))
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(self, tgt, memory, tgt_pad_mask, src_pad_mask):
        tgt = self.tgt_embed(tgt) * self.scale
        tgt = self.pos_encoder(self.dropout(tgt))
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output, cross_attn = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return output, cross_attn

    def greedy_decode(self, src, src_pad_mask, max_len=MAX_TOKENS):
        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_pad_mask)
            ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
            attn_maps = []
            for _ in range(max_len - 1):
                dec_out, cross_attn = self.decode(ys, memory, ys == PAD_ID, src_pad_mask)
                logits = self.generator(dec_out[:, -1:, :])
                next_token = logits.argmax(dim=-1)
                ys = torch.cat([ys, next_token], dim=1)
                if cross_attn is not None:
                    attn_maps.append(cross_attn.mean(dim=1)[:, -1, :])
            return ys, None


# ------------ LSTM model (unchanged) ------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, dec_dim, bias=False)
        self.dec_proj = nn.Linear(dec_dim, dec_dim, bias=False)
        self.energy = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        scores = self.energy(torch.tanh(self.enc_proj(encoder_outputs) + self.dec_proj(decoder_state).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(mask, -1e4)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)
        self.decoder = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim,
                               num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attention = BahdanauAttention(hidden_dim * 2, hidden_dim)
        self.generator = nn.Linear(hidden_dim + hidden_dim * 2, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _bridge(self, states):
        states = states.view(self.num_layers, 2, states.size(1), self.hidden_dim).sum(1)
        return states

    def greedy_decode(self, src, src_pad, src_len, max_len=MAX_TOKENS):
        self.eval()
        with torch.no_grad():
            emb = self.src_embed(src)
            packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
            enc_outputs, (hidden, cell) = self.encoder(packed)
            enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=True)
            mask = src_pad[:, : enc_outputs.size(1)]
            hidden = self._bridge(hidden)
            cell = self._bridge(cell)
            context = torch.zeros(src.size(0), enc_outputs.size(2), device=src.device)

            ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
            for _ in range(max_len - 1):
                emb = self.tgt_embed(ys[:, -1]).unsqueeze(1)
                dec_input = torch.cat([emb.squeeze(1), context], dim=-1).unsqueeze(1)
                dec_out, (hidden, cell) = self.decoder(dec_input, (hidden, cell))
                dec_state = dec_out.squeeze(1)
                context, _ = self.attention(hidden[-1], enc_outputs, mask)
                logits = self.generator(torch.cat([dec_state, context], dim=-1))
                next_tok = logits.argmax(-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)

            return ys


# ============================================================
# LOAD TOKENIZERS
# ============================================================
eng_spm_path = SPM_DIR / "eng_bpe.model"
urd_spm_path = SPM_DIR / "urd_bpe.model"
assert eng_spm_path.exists()
assert urd_spm_path.exists()

sp_en = spm.SentencePieceProcessor()
sp_ur = spm.SentencePieceProcessor()
sp_en.load(str(eng_spm_path))
sp_ur.load(str(urd_spm_path))

# ============================================================
# LOAD MODELS
# ============================================================
TRANSFORMER_CFG = {
    "d_model": 384,
    "nhead": 6,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "dim_feedforward": 1536,
    "dropout": 0.15,
}
LSTM_CFG = {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.2}

transformer_model = TransformerSeq2Seq(sp_en.vocab_size(), sp_ur.vocab_size(), **TRANSFORMER_CFG).to(DEVICE)
lstm_model = Seq2SeqLSTM(sp_en.vocab_size(), sp_ur.vocab_size(), **LSTM_CFG).to(DEVICE)

transformer_model.load_state_dict(torch.load(CHECKPOINT_DIR / "transformer.pt", map_location=DEVICE))
lstm_model.load_state_dict(torch.load(CHECKPOINT_DIR / "lstm.pt", map_location=DEVICE))
transformer_model.eval()
lstm_model.eval()

# ============================================================
# TRANSLATION FUNCTION
# ============================================================
def interactive_translate(model_name: str, text: str):
    cleaned = clean_en(text)
    if not cleaned:
        return ""

    src = torch.tensor([encode_sentence(sp_en, cleaned, MAX_TOKENS)], device=DEVICE)
    src_pad = (src == PAD_ID)
    src_len = (src != PAD_ID).sum(1)

    if model_name == "Transformer":
        decoded, _ = transformer_model.greedy_decode(src, src_pad)
    else:
        decoded = lstm_model.greedy_decode(src, src_pad, src_len)

    translation = ids_to_sentence(decoded[0].tolist(), sp_ur)
    return translation

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="English → Urdu Neural Translator", page_icon="🌐", layout="wide")

# Custom CSS for yellow-orange gradient theme with white background
st.markdown("""
<style>
    * {
        background-color: white !important;
    }
    
    .main {
        background-color: white;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(255, 165, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255, 165, 0, 0.4);
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
    }
    
    .stTextArea {
        border-radius: 12px;
    }
    
    .stTextArea textarea {
        background-color: white !important;
        border: 2px solid #FFD700 !important;
        border-radius: 8px;
        font-size: 15px;
    }
    
    h1 {
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    h2, h3 {
        color: #FF8C00;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    p {
        color: #000 !important;
        font-size: 15px;
    }
    
    .stMarkdown {
        color: #000 !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #000 !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%);
        border: 2px solid #FFD700;
        border-radius: 12px;
        color: #000 !important;
        padding: 15px;
    }
    
    .stSuccess > div {
        color: #000 !important;
        font-weight: 500;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%);
        border: 2px solid #FFA500;
        border-radius: 12px;
        color: #000 !important;
        padding: 15px;
    }
    
    .stInfo > div {
        color: #000 !important;
        font-weight: 500;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%);
        border: 2px solid #FFA500;
        border-radius: 12px;
        color: #000 !important;
        padding: 15px;
    }
    
    .stWarning > div {
        color: #000 !important;
        font-weight: 500;
    }
    
    .model-card {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.08) 0%, rgba(255, 215, 0, 0.12) 100%);
        padding: 25px;
        border-radius: 16px;
        border: 2px solid #FFD700;
        margin: 15px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.1);
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 165, 0, 0.2);
        border-color: #FFA500;
    }
    
    .model-card b {
        color: #FF8C00;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🌐 English → Urdu Translator
### Powered by Deep Learning
Translate English text into Urdu using state-of-the-art transformer and LSTM models.
""")

st.markdown("---")

# Model Selection with Custom Buttons
st.markdown("### Select Your Model")

col_t, col_l = st.columns(2)

with col_t:
    if st.button("Transformer", key="transformer_btn", use_container_width=True):
        st.session_state.selected_model = "Transformer"

with col_l:
    if st.button("LSTM", key="lstm_btn", use_container_width=True):
        st.session_state.selected_model = "LSTM"

# Initialize selected model if not set
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Transformer"

# Display model information
col_model1, col_model2 = st.columns(2)

with col_model1:
    if st.session_state.selected_model == "Transformer":
        st.markdown("""
        <div class="model-card">
        <b>Transformer</b>
        <br><br>• 4-layer encoder-decoder
        <br>• Multi-head attention mechanism
        <br>• Better translation quality
        <br>• State-of-the-art performance
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="model-card" style="opacity: 0.6;">
        <b>Transformer</b>
        <br><br>• 4-layer encoder-decoder
        <br>• Multi-head attention mechanism
        <br>• Better translation quality
        <br>• State-of-the-art performance
        </div>
        """, unsafe_allow_html=True)

with col_model2:
    if st.session_state.selected_model == "LSTM":
        st.markdown("""
        <div class="model-card">
        <b>LSTM</b>
        <br><br>• Bidirectional LSTM encoder
        <br>• Bahdanau attention mechanism
        <br>• Faster inference speed
        <br>• Lower memory footprint
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="model-card" style="opacity: 0.6;">
        <b>LSTM</b>
        <br><br>• Bidirectional LSTM encoder
        <br>• Bahdanau attention mechanism
        <br>• Faster inference speed
        <br>• Lower memory footprint
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Translation Input
st.markdown("### Translate Text")
user_input = st.text_area(
    "Enter text:",
    placeholder="",
    height=120
)

if st.button("Translate", use_container_width=True):
    if user_input.strip():
        with st.spinner("Processing..."):
            translation = interactive_translate(st.session_state.selected_model, user_input)

        st.markdown("### Translation Result")
        st.success(translation)
    else:
        st.warning("Please enter some text to translate.")

st.markdown("---")

# Example Sentences
st.markdown("### Example Sentences")

col_ex1, col_ex2, col_ex3 = st.columns(3)

examples = [
    "Peace be upon you",
    "God is merciful and compassionate",
    "Education is the key to success"
]

with col_ex1:
    if st.button(examples[0], use_container_width=True, key="ex1"):
        st.session_state["temp"] = examples[0]

with col_ex2:
    if st.button(examples[1], use_container_width=True, key="ex2"):
        st.session_state["temp"] = examples[1]

with col_ex3:
    if st.button(examples[2], use_container_width=True, key="ex3"):
        st.session_state["temp"] = examples[2]

if "temp" in st.session_state:
    st.info(f"Selected: *{st.session_state['temp']}*")
