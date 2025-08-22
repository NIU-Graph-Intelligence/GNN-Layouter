"""
Unified model registry for all graph layout models.
This is the single source of truth for available models.
"""

# Import all models
from .simple_spring_gnn import SimpleSpringGNN
from .multiscale_spring_gnn import MultiScaleSpringGNN
from .antismoothing_spring_gnn import AntiSmoothingSpringGNN
from .force_gnn import ForceGNN


from .baseline import GCN, GAT, GIN, ChebNet


# Model registry - single source of truth
MODEL_REGISTRY = {
    'GCN': GCN,
    'GAT': GAT,
    'GIN': GIN,
    'ChebNet': ChebNet,
    # Spring layout models
    'SimpleSpringGNN': SimpleSpringGNN,
    'MultiScaleSpringGNN': MultiScaleSpringGNN,
    'AntiSmoothingSpringGNN': AntiSmoothingSpringGNN,
    'ForceGNN': ForceGNN,
}

def get_available_models():
    """Get list of all available model names"""
    return list(MODEL_REGISTRY.keys())

def get_model_class(model_name):
    """Get model class by name"""
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    return MODEL_REGISTRY[model_name]

def is_model_available(model_name):
    """Check if a model is available"""
    return model_name in MODEL_REGISTRY