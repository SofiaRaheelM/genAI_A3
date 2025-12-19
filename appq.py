
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
# MODEL ARCHITECTURES
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
# STREAMLIT CHAT UI
# ============================================================

st.set_page_config(
    page_title="English → Urdu Translator",
    page_icon="🌐",
    layout="centered"
)

# Custom CSS for ChatGPT-like UI
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background-color: #ffffff;
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* English message bubble (left-aligned) */
    .user-message {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 215, 0, 0.1) 100%);
        padding: 16px 20px;
        border-radius: 18px;
        margin: 15px 0 10px 0;
        text-align: left;
        font-size: 16px;
        color: #000;
        border: 1px solid #FFD700;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(255, 165, 0, 0.1);
    }
    
    /* Urdu message bubble (right-aligned) */
    .bot-message {
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%);
        padding: 16px 20px;
        border-radius: 18px;
        margin: 10px 0 25px auto;
        text-align: right;
        direction: rtl;
        font-size: 18px;
        font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", serif;
        color: white;
        max-width: 85%;
        box-shadow: 0 2px 12px rgba(255, 165, 0, 0.3);
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 24px;
        border: 2px solid #FFD700;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9f9f9;
    }
    
    h2 {
        color: #FF8C00;
        text-align: center;
        font-weight: 700;
    }
    
    /* Scrollable chat container */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h2>🌐 English → Urdu Neural Translator</h2>", unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.radio(
    "Select Translation Model:",
    ["Transformer", "LSTM"],
    index=0,
    help="Transformer offers better quality, LSTM is faster"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use:**
1. Type English text in the input box
2. Press Enter to translate
3. Urdu translation appears below
4. Conversation history is preserved

**Models:**
- **Transformer**: 4-layer encoder-decoder with multi-head attention
- **LSTM**: Bidirectional LSTM with Bahdanau attention
""")

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    # English message (left-aligned)
    st.markdown(
        f"<div class='user-message'><strong>English:</strong><br>{msg['english']}</div>",
        unsafe_allow_html=True
    )
    # Urdu message (right-aligned)
    st.markdown(
        f"<div class='bot-message'>{msg['urdu']}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# Chat input at the bottom
user_input = st.chat_input("Type English text and press Enter...")

if user_input:
    # Show translation in progress
    with st.spinner("Translating..."):
        translation = interactive_translate(model_choice, user_input)
    
    # Add to message history
    st.session_state.messages.append({
        "english": user_input,
        "urdu": translation
    })
    
    # Rerun to display new message
    st.rerun()
