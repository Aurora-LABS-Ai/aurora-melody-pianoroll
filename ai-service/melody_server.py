#!/usr/bin/env python3
"""
Aurora Melody AI Server
========================

FastAPI server wrapping the Melody Generator transformer model.
Provides HTTP API for melody generation from Aurora Melody app.

Configuration (priority order):
    1. Command-line arguments (highest)
    2. Environment variables / .env file
    3. config.yaml
    4. Default values (lowest)

Usage:
    # With config file (recommended):
    python melody_server.py
    
    # With command-line args:
    python melody_server.py --model_path model.safetensors --config_path config.json --vocab_path vocab.pkl
    
    # With uvicorn directly:
    uvicorn melody_server:app --host 0.0.0.0 --port 8765

The server exposes:
    GET  /health          - Health check
    GET  /info            - Model info and available parameters
    POST /generate        - Generate melody (returns notes for piano roll)
    POST /generate/midi   - Generate melody (returns raw MIDI bytes)
"""

import os
import sys
import json
import base64
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from io import BytesIO

import torch
import torch.nn as nn
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Get script directory for relative config paths
SCRIPT_DIR = Path(__file__).parent.resolve()


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

class ServerConfig:
    """Server configuration loaded from multiple sources."""
    
    def __init__(self):
        self.model_path: Optional[str] = None
        self.config_path: Optional[str] = None
        self.vocab_path: Optional[str] = None
        self.host: str = "127.0.0.1"
        self.port: int = 8765
        self.device: str = "auto"
        self.log_level: str = "INFO"
    
    def load_yaml(self, yaml_path: Path) -> bool:
        """Load configuration from YAML file."""
        if not yaml_path.exists():
            return False
        
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                return False
            
            # Model paths
            model = data.get('model', {})
            if model.get('weights'):
                self.model_path = self._resolve_path(model['weights'])
            if model.get('config'):
                self.config_path = self._resolve_path(model['config'])
            if model.get('vocab'):
                self.vocab_path = self._resolve_path(model['vocab'])
            
            # Server settings
            server = data.get('server', {})
            if server.get('host'):
                self.host = server['host']
            if server.get('port'):
                self.port = int(server['port'])
            
            # Device settings
            device = data.get('device', {})
            if device.get('type'):
                self.device = device['type']
            
            # Logging
            logging_cfg = data.get('logging', {})
            if logging_cfg.get('level'):
                self.log_level = logging_cfg['level']
            
            return True
        
        except ImportError:
            print("Warning: PyYAML not installed. Install with: pip install pyyaml")
            return False
        except Exception as e:
            print(f"Warning: Failed to load {yaml_path}: {e}")
            return False
    
    def load_env(self) -> None:
        """Load configuration from environment variables and .env file."""
        # Try to load .env file
        env_path = SCRIPT_DIR / ".env"
        if env_path.exists():
            try:
                self._load_dotenv(env_path)
            except Exception as e:
                print(f"Warning: Failed to load .env: {e}")
        
        # Read from environment (overrides .env)
        if os.environ.get("MODEL_PATH"):
            self.model_path = self._resolve_path(os.environ["MODEL_PATH"])
        if os.environ.get("CONFIG_PATH"):
            self.config_path = self._resolve_path(os.environ["CONFIG_PATH"])
        if os.environ.get("VOCAB_PATH"):
            self.vocab_path = self._resolve_path(os.environ["VOCAB_PATH"])
        if os.environ.get("HOST"):
            self.host = os.environ["HOST"]
        if os.environ.get("PORT"):
            self.port = int(os.environ["PORT"])
        if os.environ.get("DEVICE"):
            self.device = os.environ["DEVICE"]
        if os.environ.get("LOG_LEVEL"):
            self.log_level = os.environ["LOG_LEVEL"]
    
    def _load_dotenv(self, path: Path) -> None:
        """Simple .env file parser (no external dependency)."""
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:  # Don't override existing env vars
                        os.environ[key] = value
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths to absolute paths."""
        p = Path(path)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        return str(p.resolve())
    
    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    def is_valid(self) -> bool:
        """Check if all required paths are set."""
        return all([self.model_path, self.config_path, self.vocab_path])
    
    def validate_paths(self) -> List[str]:
        """Validate that all paths exist. Returns list of errors."""
        errors = []
        if self.model_path and not Path(self.model_path).exists():
            errors.append(f"Model file not found: {self.model_path}")
        if self.config_path and not Path(self.config_path).exists():
            errors.append(f"Config file not found: {self.config_path}")
        if self.vocab_path and not Path(self.vocab_path).exists():
            errors.append(f"Vocab file not found: {self.vocab_path}")
        return errors


def load_config() -> ServerConfig:
    """Load configuration from all sources."""
    config = ServerConfig()
    
    # 1. Load from config.yaml (lowest priority)
    yaml_path = SCRIPT_DIR / "config.yaml"
    if yaml_path.exists():
        config.load_yaml(yaml_path)
    
    # 2. Load from environment / .env (overrides yaml)
    config.load_env()
    
    return config


# Global config instance
server_config = load_config()

# Configure logging based on config
logging.basicConfig(
    level=getattr(logging, server_config.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("melody_server")


# ============================================================================
# MODEL ARCHITECTURE (from melody_generator.py)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MelodyTransformer(nn.Module):
    """Transformer model for melody generation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_encoding = PositionalEncoding(config['d_model'], config['max_seq_len'])
        self.dropout = nn.Dropout(config['dropout'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['n_layers']
        )
        
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids) * np.sqrt(self.config['d_model'])
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, start_tokens: List[int], max_length: int = 300, 
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9, 
                repetition_penalty: float = 1.1, device='cpu'):
        """Generate melody tokens with advanced sampling."""
        self.eval()
        
        with torch.no_grad():
            generated = start_tokens.copy()
            input_ids = torch.tensor([generated], device=device)
            
            eos_token = self.config.get('eos_token', 1)
            
            for _ in range(max_length - len(start_tokens)):
                logits = self(input_ids)
                next_token_logits = logits[0, -1].clone()
                
                # Repetition penalty
                if repetition_penalty != 1.0 and len(generated) > 1:
                    for prev_token in set(generated):
                        if prev_token < len(next_token_logits):
                            if next_token_logits[prev_token] < 0:
                                next_token_logits[prev_token] *= repetition_penalty
                            else:
                                next_token_logits[prev_token] /= repetition_penalty
                
                # Temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-K filtering
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, indices, values)
                
                # Nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
                
                if next_token == eos_token:
                    break
        
        return generated


# ============================================================================
# TOKENIZER
# ============================================================================

class DataDrivenTokenizer:
    """Tokenizer wrapper for data-driven vocabulary."""
    
    def __init__(self, vocab_path: str):
        import pickle
        
        logger.info(f"Loading vocabulary from: {vocab_path}")
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.token_to_event = vocab_data['token_to_event']
        self.event_to_token = vocab_data['event_to_token']
        self.vocab_size = len(self.token_to_event)
        
        logger.info(f"Loaded vocabulary: {self.vocab_size:,} tokens")
    
    def decode_token(self, token_id: int) -> Optional[Dict]:
        return self.token_to_event.get(token_id, None)
    
    def encode_key(self, key: str, modulation: str = "tonic") -> Optional[int]:
        event_name = f"KEY_{key}_{modulation}"
        return self.event_to_token.get(event_name, None)
    
    def encode_rhythmic_feel(self, feel: str, intensity: int = 2) -> Optional[int]:
        event_name = f"RHYTHM_{feel}_{intensity}"
        return self.event_to_token.get(event_name, None)
    
    def encode_dynamic_shape(self, shape: str, intensity: int = 2) -> Optional[int]:
        event_name = f"DYNAMIC_{shape}_{intensity}"
        return self.event_to_token.get(event_name, None)
    
    def encode_contour(self, contour: str, size: int = 1) -> Optional[int]:
        event_name = f"CONTOUR_{contour}_{size}"
        return self.event_to_token.get(event_name, None)
    
    def encode_scale_degree(self, degree: str, quality: str = "stable") -> Optional[int]:
        event_name = f"DEGREE_{degree}_{quality}"
        return self.event_to_token.get(event_name, None)
    
    def detokenize_to_notes(self, tokens: List[int], tempo: int = 120) -> List[Dict]:
        """Convert tokens directly to note list for Aurora Melody piano roll."""
        notes = []
        current_beat = 0.0
        ticks_per_beat = 480  # Standard MIDI resolution
        ticks_per_step = ticks_per_beat / 16.0
        
        velocity_map = {
            "ppp": 16, "ppp+": 24, "pp": 32, "pp+": 40,
            "p": 48, "p+": 56, "mp": 64, "mp+": 72,
            "mf": 80, "mf+": 88, "f": 96, "f+": 104,
            "ff": 112, "ff+": 120, "fff": 124, "fff+": 127
        }
        
        articulation_modifiers = {
            'normal': 1.0, 'staccato': 0.5, 'legato': 1.1,
            'accent': 1.0, 'marcato': 0.7
        }
        
        for token in tokens:
            token_info = self.decode_token(token)
            if not token_info:
                continue
            
            token_type = token_info.get('type')
            
            if token_type == 'melody_note':
                pitch = token_info['pitch']
                duration = token_info['duration']
                velocity_name = token_info.get('velocity', 'mf')
                articulation = token_info.get('articulation', 'normal')
                
                velocity = velocity_map.get(velocity_name, 80)
                art_mod = articulation_modifiers.get(articulation, 1.0)
                
                # Convert duration from steps to beats
                duration_beats = (duration * art_mod) / 16.0
                
                notes.append({
                    "noteNumber": pitch,
                    "startBeat": current_beat,
                    "lengthBeats": max(0.0625, duration_beats),  # Minimum 1/16 beat
                    "velocity": velocity,
                    "channel": 1
                })
                
                current_beat += duration / 16.0
            
            elif token_type == 'rest':
                rest_duration = token_info['duration']
                current_beat += rest_duration / 16.0
        
        return notes
    
    def detokenize_to_midi(self, tokens: List[int], tempo: int = 120) -> bytes:
        """Convert tokens to MIDI bytes."""
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        track.append(MetaMessage('set_tempo', tempo=int(60000000 / tempo)))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4))
        
        cumulative_time = 0.0
        ticks_per_step = midi.ticks_per_beat / 16.0
        
        velocity_map = {
            "ppp": 8, "ppp+": 16, "pp": 24, "pp+": 32,
            "p": 40, "p+": 48, "mp": 56, "mp+": 64,
            "mf": 72, "mf+": 80, "f": 88, "f+": 96,
            "ff": 104, "ff+": 112, "fff": 120, "fff+": 127
        }
        
        for token in tokens:
            token_info = self.decode_token(token)
            if not token_info:
                continue
            
            token_type = token_info.get('type')
            
            if token_type == 'melody_note':
                pitch = token_info['pitch']
                duration = token_info['duration']
                velocity_name = token_info.get('velocity', 'mf')
                articulation = token_info.get('articulation', 'normal')
                
                velocity = velocity_map.get(velocity_name, 64)
                
                articulation_modifiers = {
                    'normal': 1.0, 'staccato': 0.5, 'legato': 1.2,
                    'accent': 1.0, 'marcato': 0.7
                }
                art_mod = articulation_modifiers.get(articulation, 1.0)
                
                duration_ticks = max(1, int(round(duration * ticks_per_step * art_mod)))
                
                track.append(Message('note_on', channel=0, note=pitch,
                                    velocity=velocity, time=int(round(cumulative_time))))
                track.append(Message('note_off', channel=0, note=pitch,
                                    velocity=0, time=duration_ticks))
                cumulative_time = 0.0
            
            elif token_type == 'rest':
                rest_duration = token_info['duration']
                cumulative_time += rest_duration * ticks_per_step
        
        bytes_io = BytesIO()
        midi.save(file=bytes_io)
        return bytes_io.getvalue()


# ============================================================================
# GLOBAL STATE
# ============================================================================

class ModelState:
    """Global model state container."""
    model: Optional[MelodyTransformer] = None
    config: Optional[Dict] = None
    tokenizer: Optional[DataDrivenTokenizer] = None
    device: str = "cpu"
    is_loaded: bool = False

model_state = ModelState()


# ============================================================================
# API MODELS
# ============================================================================

class GenerateRequest(BaseModel):
    """Request model for melody generation."""
    
    # Sampling parameters
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=1, le=200, description="Top-K sampling")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    max_length: int = Field(default=200, ge=50, le=500, description="Max generation length")
    
    # Musical parameters
    tempo: int = Field(default=120, ge=40, le=240, description="Output tempo BPM")
    
    # Key signature
    key: Optional[str] = Field(default=None, description="Key signature (e.g., C_MAJ, A_MIN)")
    key_modulation: str = Field(default="tonic", description="Key modulation type")
    
    # Rhythmic feel
    rhythmic_feel: Optional[str] = Field(default=None, description="Rhythmic feel")
    rhythmic_intensity: int = Field(default=2, ge=0, le=3, description="Rhythmic intensity")
    
    # Dynamic shape
    dynamic_shape: Optional[str] = Field(default=None, description="Dynamic shape")
    dynamic_intensity: int = Field(default=2, ge=0, le=4, description="Dynamic intensity")
    
    # Melodic contour
    contour: Optional[str] = Field(default=None, description="Starting melodic contour")
    contour_size: int = Field(default=1, ge=0, le=3, description="Contour interval size")
    
    # Scale degree
    scale_degree: Optional[str] = Field(default=None, description="Starting scale degree")
    scale_quality: str = Field(default="stable", description="Scale degree quality")
    
    # Random seed
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class NoteData(BaseModel):
    """Single note in Aurora Melody format."""
    noteNumber: int
    startBeat: float
    lengthBeats: float
    velocity: int
    channel: int = 1


class GenerateResponse(BaseModel):
    """Response model for melody generation."""
    status: str
    notes: List[NoteData] = []
    token_count: int = 0
    generation_time_ms: float = 0
    message: Optional[str] = None


class MidiResponse(BaseModel):
    """Response model for MIDI generation."""
    status: str
    midi_base64: str
    token_count: int = 0
    generation_time_ms: float = 0


class InfoResponse(BaseModel):
    """Model info response."""
    status: str
    model_loaded: bool
    device: str
    vocab_size: Optional[int] = None
    parameters: Dict[str, Any] = {}


# ============================================================================
# GENERATION LOGIC
# ============================================================================

def generate_melody_tokens(request: GenerateRequest) -> List[int]:
    """Generate melody tokens from request parameters."""
    
    if not model_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = model_state.model
    config = model_state.config
    tokenizer = model_state.tokenizer
    device = model_state.device
    
    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
    
    bos_token = config.get('bos_token', 0)
    start_tokens = [bos_token]
    
    # Build control tokens
    if request.key:
        token = tokenizer.encode_key(request.key, request.key_modulation)
        if token is not None:
            start_tokens.append(token)
    
    if request.rhythmic_feel:
        token = tokenizer.encode_rhythmic_feel(request.rhythmic_feel, request.rhythmic_intensity)
        if token is not None:
            start_tokens.append(token)
    
    if request.dynamic_shape:
        token = tokenizer.encode_dynamic_shape(request.dynamic_shape, request.dynamic_intensity)
        if token is not None:
            start_tokens.append(token)
    
    if request.contour:
        token = tokenizer.encode_contour(request.contour, request.contour_size)
        if token is not None:
            start_tokens.append(token)
    
    if request.scale_degree:
        token = tokenizer.encode_scale_degree(request.scale_degree, request.scale_quality)
        if token is not None:
            start_tokens.append(token)
    
    # Generate
    generated_tokens = model.generate(
        start_tokens=start_tokens,
        max_length=request.max_length,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        device=device
    )
    
    # Filter special tokens
    eos_token = config.get('eos_token', 1)
    pad_token = config.get('pad_token', 2)
    special_tokens = {bos_token, eos_token, pad_token}
    
    melody_tokens = [t for t in generated_tokens if t not in special_tokens]
    
    return melody_tokens


# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model loading."""
    global server_config
    
    # Reload config in case it was updated by CLI args
    server_config = load_config()
    
    if server_config.is_valid():
        # Validate paths exist
        errors = server_config.validate_paths()
        if errors:
            for err in errors:
                logger.error(err)
            logger.warning("Model files not found. Use /load endpoint to load model at runtime.")
        else:
            device = server_config.resolve_device()
            load_model_files(
                server_config.model_path,
                server_config.config_path,
                server_config.vocab_path,
                device
            )
    else:
        logger.warning("Model paths not configured.")
        logger.info("Configure via: config.yaml, .env file, environment variables, or CLI args")
        logger.info("Or use POST /load endpoint to load model at runtime.")
    
    yield
    
    # Cleanup
    logger.info("Shutting down server...")


app = FastAPI(
    title="Aurora Melody AI Server",
    description="AI-powered melody generation for Aurora Melody piano roll",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_files(model_path: str, config_path: str, vocab_path: str, device: str):
    """Load model, config, and vocabulary."""
    from safetensors.torch import load_file
    
    logger.info(f"Loading model on device: {device}")
    
    # Load config
    logger.info(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    logger.info(f"Creating model (vocab_size: {config['vocab_size']:,})")
    model = MelodyTransformer(config).to(device)
    
    # Load weights
    logger.info(f"Loading weights: {model_path}")
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load tokenizer
    tokenizer = DataDrivenTokenizer(vocab_path)
    
    # Store in global state
    model_state.model = model
    model_state.config = config
    model_state.tokenizer = tokenizer
    model_state.device = device
    model_state.is_loaded = True
    
    logger.info("Model loaded successfully!")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_state.is_loaded,
        "device": model_state.device
    }


@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get model info and available parameters."""
    params = {
        "keys": [
            "C_MAJ", "G_MAJ", "D_MAJ", "A_MAJ", "E_MAJ", "B_MAJ", "F#_MAJ",
            "F_MAJ", "Bb_MAJ", "Eb_MAJ", "Ab_MAJ", "Db_MAJ", "Gb_MAJ",
            "A_MIN", "E_MIN", "B_MIN", "F#_MIN", "C#_MIN", "G#_MIN",
            "D_MIN", "G_MIN", "C_MIN", "F_MIN", "Bb_MIN", "Eb_MIN"
        ],
        "key_modulations": ["tonic", "dominant", "subdominant", "relative", "parallel", "modulation"],
        "rhythmic_feels": ["straight", "swing", "shuffle", "latin", "funk"],
        "dynamic_shapes": ["none", "crescendo", "diminuendo", "swell"],
        "contours": ["step_up", "step_down", "leap_up", "leap_down", "repeat"],
        "scale_degrees": ["1", "2", "3", "4", "5", "6", "7"],
        "scale_qualities": ["stable", "unstable", "leading"]
    }
    
    return InfoResponse(
        status="ok",
        model_loaded=model_state.is_loaded,
        device=model_state.device,
        vocab_size=model_state.config.get('vocab_size') if model_state.config else None,
        parameters=params
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_notes(request: GenerateRequest):
    """Generate melody and return notes for Aurora Melody piano roll."""
    import time
    
    if not model_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Set environment variables or call /load endpoint.")
    
    start_time = time.time()
    
    try:
        # Generate tokens
        melody_tokens = generate_melody_tokens(request)
        
        # Convert to notes
        notes = model_state.tokenizer.detokenize_to_notes(melody_tokens, request.tempo)
        
        generation_time = (time.time() - start_time) * 1000
        
        logger.info(f"Generated {len(notes)} notes in {generation_time:.1f}ms")
        
        return GenerateResponse(
            status="success",
            notes=[NoteData(**n) for n in notes],
            token_count=len(melody_tokens),
            generation_time_ms=generation_time
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/midi", response_model=MidiResponse)
async def generate_midi(request: GenerateRequest):
    """Generate melody and return MIDI bytes (base64 encoded)."""
    import time
    
    if not model_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Generate tokens
        melody_tokens = generate_melody_tokens(request)
        
        # Convert to MIDI
        midi_bytes = model_state.tokenizer.detokenize_to_midi(melody_tokens, request.tempo)
        
        generation_time = (time.time() - start_time) * 1000
        
        logger.info(f"Generated MIDI ({len(midi_bytes)} bytes) in {generation_time:.1f}ms")
        
        return MidiResponse(
            status="success",
            midi_base64=base64.b64encode(midi_bytes).decode('utf-8'),
            token_count=len(melody_tokens),
            generation_time_ms=generation_time
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class LoadRequest(BaseModel):
    """Request to load model files."""
    model_path: str
    config_path: str
    vocab_path: str
    device: str = "cuda"


@app.post("/load")
async def load_model(request: LoadRequest):
    """Load model files at runtime."""
    try:
        load_model_files(
            request.model_path,
            request.config_path,
            request.vocab_path,
            request.device
        )
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Aurora Melody AI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration priority (highest to lowest):
  1. Command-line arguments
  2. Environment variables / .env file
  3. config.yaml
  4. Default values

Examples:
  # Start with config.yaml (edit config.yaml first):
  python melody_server.py
  
  # Start with command-line args:
  python melody_server.py --model_path model.safetensors --config_path config.json --vocab_path vocab.pkl
  
  # Start with environment variables:
  MODEL_PATH=./model.safetensors CONFIG_PATH=./config.json VOCAB_PATH=./vocab.pkl python melody_server.py
        """
    )
    parser.add_argument('--model_path', type=str, help='Model weights (.safetensors)')
    parser.add_argument('--config_path', type=str, help='Model config (.json)')
    parser.add_argument('--vocab_path', type=str, help='Vocabulary (.pkl)')
    parser.add_argument('--host', type=str, help='Server host (default: from config or 127.0.0.1)')
    parser.add_argument('--port', type=int, help='Server port (default: from config or 8765)')
    parser.add_argument('--device', type=str, help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--config', type=str, help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Load custom config file if specified
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            server_config.load_yaml(config_path)
        else:
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
    
    # Override with CLI args (highest priority)
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.config_path:
        os.environ["CONFIG_PATH"] = args.config_path
    if args.vocab_path:
        os.environ["VOCAB_PATH"] = args.vocab_path
    if args.device:
        os.environ["DEVICE"] = args.device
    
    # Determine host/port (CLI > env > config > default)
    host = args.host or os.environ.get("HOST") or server_config.host
    port = args.port or (int(os.environ.get("PORT")) if os.environ.get("PORT") else None) or server_config.port
    
    print("")
    print("=" * 50)
    print("  Aurora Melody AI Server")
    print("=" * 50)
    print(f"  Host:   {host}")
    print(f"  Port:   {port}")
    print(f"  Device: {server_config.resolve_device()}")
    print("")
    
    if server_config.is_valid():
        print(f"  Model:  {server_config.model_path}")
        print(f"  Config: {server_config.config_path}")
        print(f"  Vocab:  {server_config.vocab_path}")
    else:
        print("  Model paths not configured - will start without model")
        print("  Configure via config.yaml, .env, or CLI args")
    print("=" * 50)
    print("")
    
    import uvicorn
    uvicorn.run(
        "melody_server:app",
        host=host,
        port=port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

