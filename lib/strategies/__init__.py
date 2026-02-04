"""EIMAS Strategies"""
def _safe_import(module_name, class_names):
    try:
        module = __import__(module_name, fromlist=class_names)
        return {name: getattr(module, name, None) for name in class_names}
    except (ImportError, AttributeError):
        return {name: None for name in class_names}

# Safe imports
try:
    from .etf import CustomETFBuilder
except ImportError:
    CustomETFBuilder = None

__all__ = ["CustomETFBuilder"]
