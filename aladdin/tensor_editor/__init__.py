"""Bounded direct tensor editing: the WSP edit engine and its safe operations."""
from . import ops
from .engine import EditRefused, TensorEditor

__all__ = ["TensorEditor", "EditRefused", "ops"]
