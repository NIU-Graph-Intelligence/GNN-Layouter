"""
philayouter/executor

The equivariant executor: one learned step of a force-directed layout
update, X_{t+1} = Phi(G, X_t, tau_t), with every term of the FR update
mapped onto a dedicated component so the model is rotation/translation-
equivariant by construction.
"""

from .model import EquivariantExecutor
from .structural import StructuralEncoder

__all__ = ["EquivariantExecutor", "StructuralEncoder"]
