"""
Utility functions for standardized configuration handling.
Provides backwards compatibility with old config names.
"""

def get_latent_dim(cfg):
    """Get latent_dim with fallback to stoch_dim"""
    return cfg.training.model.get('latent_dim', cfg.training.model.get('stoch_dim', 128))

def get_hidden_dim(cfg):
    """Get hidden_dim with fallback to deter_dim"""
    return cfg.training.model.get('hidden_dim', cfg.training.model.get('deter_dim', 128))

def get_embed_dim(cfg):
    """Get embed_dim with fallback to r_dim"""
    return cfg.training.model.get('embed_dim', cfg.training.model.get('r_dim', 128))

def get_use_actions(cfg, default=True):
    """
    Get use_actions flag.
    Default True for model-free (backwards compatibility)
    Default False for model-based (new behavior)
    """
    return cfg.training.model.get('use_actions', default)

def get_aggregator(cfg):
    """Get aggregator type (default 'mean')"""
    return cfg.training.model.get('aggregator', 'mean')
