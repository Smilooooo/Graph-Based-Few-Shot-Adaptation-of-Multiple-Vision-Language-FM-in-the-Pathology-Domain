"""
Model Registry for loading different CLIP-like foundation models.

Supports:
- OpenAI CLIP (original)
- open_clip models (BiomedCLIP, QuiltNet, etc.)
- HuggingFace transformers models (PLIP, etc.)
- Custom models (CONCH, etc.)

Usage:
    from trainers.model_registry import load_clip_model
    
    model, preprocess, tokenizer = load_clip_model(cfg)
"""

from matplotlib import text
import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable, Any

# Local CLIP module (original OpenAI)
from clip import clip


# =============================================================================
# Model Configuration Registry
# =============================================================================

MODEL_CONFIGS = {
    # OpenAI CLIP models (default)
    "RN50": {"framework": "openai_clip", "feature_dim": 1024},
    "RN101": {"framework": "openai_clip", "feature_dim": 512},
    "RN50x4": {"framework": "openai_clip", "feature_dim": 640},
    "RN50x16": {"framework": "openai_clip", "feature_dim": 768},
    "ViT-B/32": {"framework": "openai_clip", "feature_dim": 512},
    "ViT-B/16": {"framework": "openai_clip", "feature_dim": 512},
    
    # BiomedCLIP (open_clip with CustomTextCLIP)
    # Uses create_model_from_pretrained, context_length=256
    # Docs: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    "biomedclip": {
        "framework": "open_clip",
        "hub_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "feature_dim": 512,
        "context_length": 256,
    },
    
    # PLIP - Pathology Language and Image Pre-Training (HuggingFace transformers)
    # Uses CLIPModel and CLIPProcessor from transformers
    # Docs: https://huggingface.co/vinid/plip
    "plip": {
        "framework": "transformers",
        "hub_name": "vinid/plip",
        "feature_dim": 512,
        "context_length": 77,
    },
    
    # QuiltNet (open_clip)
    "quiltnet": {
        "framework": "open_clip",
        "hub_name": "hf-hub:wisdomik/QuiltNet-B-32",
        "feature_dim": 512,
        "context_length": 77,
    },
    
    # CONCH - Contrastive learning from Captions for Histopathology
    # Uses conch.open_clip_custom.create_model_from_pretrained
    # REQUIRES HuggingFace authentication (gated model)
    # Docs: https://github.com/mahmoodlab/CONCH
    "conch": {
        "framework": "conch",
        "model_name": "conch_ViT-B-16",  # Model architecture name
        "hub_name": "hf_hub:MahmoodLab/conch",  # HuggingFace path
        "feature_dim": 512,
        "context_length": 128,  # CONCH uses 127 + 1 padding = 128 total
    },
}


# =============================================================================
# Wrapper Classes for Unified Interface
# =============================================================================

class CLIPModelWrapper(nn.Module):
    """
    Unified wrapper that provides consistent interface across different CLIP implementations.
    
    Attributes:
        visual: Image encoder
        token_embedding: Token embedding layer (for text)
        transformer: Text transformer
        positional_embedding: Positional embeddings for text
        ln_final: Final layer norm for text
        text_projection: Text projection matrix
        logit_scale: Learnable temperature parameter
        dtype: Model dtype (float16 or float32)
    """
    
    def __init__(self, model, tokenizer, framework: str, feature_dim: int, context_length: int = 77):
        super().__init__()
        self.framework = framework
        self._feature_dim = feature_dim
        self._tokenizer = tokenizer
        self._model = model
        self._context_length = context_length
        
        if framework == "open_clip":
            self._setup_open_clip(model)
        elif framework == "transformers":
            self._setup_transformers(model)
        elif framework == "conch":
            self._setup_conch(model)
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
        
    def _setup_open_clip(self, model):
        """Setup for open_clip models (BiomedCLIP, QuiltNet)."""
        self.visual = model.visual
        self.logit_scale = model.logit_scale
        
        # Handle different open_clip architectures
        # Standard CLIP has these attributes, but CustomTextCLIP (BiomedCLIP) doesn't
        # BiomedCLIP uses a HuggingFace text encoder (PubMedBERT) internally
        self.token_embedding = getattr(model, 'token_embedding', None)
        self.transformer = getattr(model, 'transformer', None)
        self.positional_embedding = getattr(model, 'positional_embedding', None)
        self.ln_final = getattr(model, 'ln_final', None)
        self.text_projection = getattr(model, 'text_projection', None)
        
        # Get dtype from model parameters
        self.dtype = next(model.parameters()).dtype
        
    def _setup_transformers(self, model):
        """Setup for HuggingFace transformers models (PLIP)."""
        # transformers CLIPModel has different structure
        self.visual = TransformersVisualWrapper(model)
        # These need special handling for transformers
        self.positional_embedding = model.text_model.embeddings.position_embedding.weight
        self.ln_final = model.text_model.final_layer_norm
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        self.dtype = next(model.parameters()).dtype
        
    def _setup_conch(self, model):
        """Setup for CONCH model."""
        # CONCH follows open_clip interface but visual encoder returns tuple
        self._setup_open_clip(model)
        
        # Wrap visual encoder to handle tuple output
        # CONCH's CoCa architecture returns (pooled_features, sequence_features)
        # We only need the pooled features (first element)
        original_visual = self.visual
        
        class CONCHVisualWrapper(nn.Module):
            def __init__(self, visual_encoder):
                super().__init__()
                self.visual = visual_encoder
            
            def forward(self, x):
                result = self.visual(x)
                # Debug print to verify output structure
                print(f"CONCH visual output type: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")\
                # Debug print to check tensor shapes
                if isinstance(result, tuple):
                    print(f"  Pooled features shape: {result[0].shape}, Sequence features shape: {result[1].shape}")
                else:
                    print(f"  Visual output shape: {result.shape}")

                # CONCH returns tuple (pooled_features, sequence_features)
                if isinstance(result, tuple):
                    return result[0]  # Return only pooled features
                return result
        
        self.visual = CONCHVisualWrapper(original_visual)
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self._model.parameters()).device
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        return self.visual(image.type(self.dtype))
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to feature vectors."""
        if self.framework in ["open_clip", "conch"]:
            # BiomedCLIP uses CustomTextCLIP without token_embedding exposed

            return self._model.encode_text(text)
            
        elif self.framework == "transformers":
            # PLIP-specific: get_text_features with input_ids returns tensor directly
            features = self._model.get_text_features(input_ids=text)
            # Should return a tensor [batch, feature_dim]
            if isinstance(features, torch.Tensor):
                return features
            # Fallback for other transformers models
            elif hasattr(features, 'pooler_output'):
                return features.pooler_output
            else:
                return features.last_hidden_state[:, 0, :]
    def tokenize(self, texts, context_length: int = None):
        """Tokenize text using the appropriate tokenizer."""
        if context_length is None:
            context_length = self._context_length
        
        if self.framework == "transformers":
            return self._tokenizer(texts, padding=True, return_tensors="pt")["input_ids"]
        elif self.framework in ["open_clip", "conch"]:
            tokens = self._tokenizer(texts, context_length=context_length)
            
            # Debug: Check token range for CONCH
            if self.framework == "conch":
                print(f"  CONCH tokenization debug:")
                print(f"    Input texts (first): {texts[0] if isinstance(texts, list) else texts}")
                print(f"    Token shape: {tokens.shape}")
                print(f"    Token min: {tokens.min().item()}, max: {tokens.max().item()}")
                print(f"    Context length: {context_length}")
            
            return tokens
        else:
            raise ValueError(f"Unknown framework for tokenization: {self.framework}")


class TransformersVisualWrapper(nn.Module):
    """Wrapper to make transformers vision encoder match CLIP interface."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(x)
        pooled = outputs.pooler_output
        return self.visual_projection(pooled)


# =============================================================================
# Model Loading Functions
# =============================================================================


def _load_open_clip(hub_name: str, device: str = "cpu") -> Tuple[Any, Callable, Callable]:
    """Load open_clip model from HuggingFace Hub."""
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip is required for this model. "
            "Install with: pip install open_clip_torch"
        )
    
    # BiomedCLIP and similar models use create_model_from_pretrained
    # which returns (model, preprocess) - no transforms in the middle
    if hub_name.startswith("hf-hub:"):
        try:
            # Try create_model_from_pretrained first (for BiomedCLIP, etc.)
            model, preprocess = open_clip.create_model_from_pretrained(hub_name)
            tokenizer = open_clip.get_tokenizer(hub_name)
            print(f"  Loaded using create_model_from_pretrained")
        except Exception as e:
            # Fallback to create_model_and_transforms
            print(f"  Falling back to create_model_and_transforms: {e}")
            model, _, preprocess = open_clip.create_model_and_transforms(hub_name)
            tokenizer = open_clip.get_tokenizer(hub_name)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(hub_name)
        tokenizer = open_clip.get_tokenizer(hub_name)
    
    model = model.to(device)
    
    return model, preprocess, tokenizer


def _load_transformers(hub_name: str, device: str = "cpu") -> Tuple[Any, Callable, Callable]:
    """Load HuggingFace transformers CLIP model."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError(
            "transformers is required for this model. "
            "Install with: pip install transformers"
        )
    
    # Workaround for transformers bug with Dassl's logger (no isatty method)
    import sys
    original_stdout = sys.stdout
    
    # Create a wrapper that adds isatty method
    class StdoutWrapper:
        def __init__(self, wrapped):
            self._wrapped = wrapped
        def __getattr__(self, name):
            if name == 'isatty':
                return lambda: False
            return getattr(self._wrapped, name)
    
    sys.stdout = StdoutWrapper(sys.stdout)
    
    try:
        model = CLIPModel.from_pretrained(hub_name, use_safetensors=True)
        processor = CLIPProcessor.from_pretrained(hub_name)
    finally:
        sys.stdout = original_stdout
    
    model = model.to(device)
    
    # Create preprocess function from processor
    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    
    return model, preprocess, processor.tokenizer


def _load_conch(model_name: str, hub_name: str, device: str = "cpu") -> Tuple[Any, Callable, Callable]:
    """
    Load CONCH model.
    
    CONCH requires HuggingFace authentication since it's a gated model.
    Set HF_TOKEN environment variable or run `huggingface-cli login`.
    
    Args:
        model_name: Model architecture name (e.g., "conch_ViT-B-16")
        hub_name: HuggingFace hub path (e.g., "hf_hub:MahmoodLab/conch")
        device: Device to load model on
    """
    import os
    hf_token = os.environ.get("HF_TOKEN", None)
    
    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
        print(f"  Loading CONCH using conch package")
        print(f"  Model: {model_name}, Hub: {hub_name}")
        
        # CONCH API: create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=...)
        if hf_token:
            model, preprocess = create_model_from_pretrained(model_name, hub_name, hf_auth_token=hf_token)
        else:
            model, preprocess = create_model_from_pretrained(model_name, hub_name)
        
        model = model.to(device)
        
        # CONCH requires both tokenizer and tokenize function
        # Note: The tokenize function in CONCH uses batch_encode_plus which is deprecated
        # in newer transformers. We call the tokenizer directly instead.
        import torch.nn.functional as F
        tokenizer = get_tokenizer()
        print(f"  Using CONCH-specific tokenizer (direct call, bypassing batch_encode_plus)")
        
        # Create a wrapper that matches the expected interface
        # Replicates CONCH's tokenize logic: https://github.com/mahmoodlab/CONCH/blob/main/conch/open_clip_custom/custom_tokenizer.py
        def conch_tokenize(texts, context_length=None):
            # Call tokenizer directly (works with newer transformers)
            # CONCH model context length is 128, but last token is reserved for <cls>
            # so we use 127 and insert <pad> at the end as a temporary placeholder
            result = tokenizer(texts, 
                             max_length=127,
                             add_special_tokens=True, 
                             return_token_type_ids=False,
                             truncation=True,
                             padding='max_length',
                             return_tensors='pt')
            # Pad with one more token at the end (to make 128 total)
            tokens = F.pad(result['input_ids'], (0, 1), value=tokenizer.pad_token_id)
            return tokens
        
        return model, preprocess, conch_tokenize
        
    except ImportError as e:
        print(f"  CONCH package not found: {e}")
        print(f"  Install with: pip install conch-ai")
        print(f"  Or: pip install git+https://github.com/mahmoodlab/CONCH.git")
        raise ImportError(
            "CONCH model requires the conch package. "
            "Install with: pip install git+https://github.com/mahmoodlab/CONCH.git"
        )


def load_clip_model(cfg, device: str = "cpu", backbone_override: str = None) -> CLIPModelWrapper:
    """
    Load a CLIP-like model based on configuration.
    
    Args:
        cfg: Configuration object with MODEL.BACKBONE settings
        device: Device to load model on
        backbone_override: If provided, use this backbone name instead of cfg.MODEL.BACKBONE.NAME
            This is useful for multi-FM fusion where we need to load multiple models.
        
    Returns:
        CLIPModelWrapper: Unified model wrapper with consistent interface
        
    Config options:
        cfg.MODEL.BACKBONE.NAME: Model name (e.g., "RN50", "biomedclip", "plip")
        cfg.MODEL.BACKBONE.HUB_NAME: (Optional) Override HuggingFace hub name
        cfg.MODEL.BACKBONE.FRAMEWORK: (Optional) Override framework
        cfg.MODEL.BACKBONE.FEATURE_DIM: (Optional) Override feature dimension
    """
    backbone_name = backbone_override if backbone_override else cfg.MODEL.BACKBONE.NAME
    
    # Check if it's a known model
    if backbone_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[backbone_name].copy()
    else:
        # Try to use config values
        config = {
            "framework": getattr(cfg.MODEL.BACKBONE, "FRAMEWORK", "openai_clip"),
            "hub_name": getattr(cfg.MODEL.BACKBONE, "HUB_NAME", backbone_name),
            "feature_dim": getattr(cfg.MODEL.BACKBONE, "FEATURE_DIM", 512),
        }
    
    # Allow config overrides
    if hasattr(cfg.MODEL.BACKBONE, "FRAMEWORK") and cfg.MODEL.BACKBONE.FRAMEWORK:
        config["framework"] = cfg.MODEL.BACKBONE.FRAMEWORK
    if hasattr(cfg.MODEL.BACKBONE, "HUB_NAME") and cfg.MODEL.BACKBONE.HUB_NAME:
        config["hub_name"] = cfg.MODEL.BACKBONE.HUB_NAME
    if hasattr(cfg.MODEL.BACKBONE, "FEATURE_DIM") and cfg.MODEL.BACKBONE.FEATURE_DIM:
        config["feature_dim"] = cfg.MODEL.BACKBONE.FEATURE_DIM
    
    framework = config["framework"]
    feature_dim = config["feature_dim"]
    hub_name = config.get("hub_name", backbone_name)
    
    print(f"Loading model: {backbone_name}")
    print(f"  Framework: {framework}")
    print(f"  Feature dim: {feature_dim}")
    if hub_name != backbone_name:
        print(f"  Hub name: {hub_name}")
    
    # Load based on framework
    if framework == "open_clip":
        model, preprocess, tokenizer = _load_open_clip(hub_name, device)
    elif framework == "transformers":
        model, preprocess, tokenizer = _load_transformers(hub_name, device)
    elif framework == "conch":
        # CONCH requires model_name (architecture) and hub_name (HF path)
        model_name = config.get("model_name", "conch_ViT-B-16")
        model, preprocess, tokenizer = _load_conch(model_name, hub_name, device)
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    # Wrap in unified interface
    context_length = config.get("context_length", 77)
    wrapped = CLIPModelWrapper(model, tokenizer, framework, feature_dim, context_length)
    
    return wrapped


def get_feature_dim(cfg) -> int:
    """Get the feature dimension for a model configuration."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    if backbone_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[backbone_name]["feature_dim"]
    
    if hasattr(cfg.MODEL.BACKBONE, "FEATURE_DIM"):
        return cfg.MODEL.BACKBONE.FEATURE_DIM
    
    # Default
    return 512
