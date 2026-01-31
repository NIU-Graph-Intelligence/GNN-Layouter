"""Graph neural network models for layout tasks."""

__all__ = []

# Try to import baseline models
try:
    from .baseline import GCN, GAT, GIN, ChebNet
    __all__.extend(['GCN', 'GAT', 'GIN', 'ChebNet'])
    print("Successfully imported baseline models")
except ImportError as e:
    print(f"Failed to import baseline models: {e}")

# Try to import spring layout models
try:
    from .simple_spring_gnn import SimpleSpringGNN
    __all__.append('SimpleSpringGNN')
    print("Successfully imported SimpleSpringGNN")
except ImportError as e:
    print(f"Failed to import SimpleSpringGNN: {e}")

try:
    from .multiscale_spring_gnn import MultiScaleSpringGNN
    __all__.append('MultiScaleSpringGNN')
    print("Successfully imported MultiScaleSpringGNN")
except ImportError as e:
    print(f"Failed to import MultiScaleSpringGNN: {e}")

try:
    from .antismoothing_spring_gnn import AntiSmoothingSpringGNN
    __all__.append('AntiSmoothingSpringGNN')
    print("Successfully imported AntiSmoothingSpringGNN")
except ImportError as e:
    print(f"Failed to import AntiSmoothingSpringGNN: {e}")

print(f"Available models: {__all__}")