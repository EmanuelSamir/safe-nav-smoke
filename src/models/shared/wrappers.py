import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional
import abc

from src.models.model_free_utils import ObsSNP
from src.models.pinn_utils import ObsPINN
from src.models.snp_v2 import SnpV2, SnpV2State
from src.models.rnp import ScalarFieldRNP
from src.models.snp_v1 import ScalarFieldSNP

class SmokeModelWrapper(abc.ABC):
    """
    Unified interface for smoke forecasting models.
    """
    @abc.abstractmethod
    def predict_future(self, 
                       context_obs: Any, 
                       context_actions: torch.Tensor,
                       future_actions: Optional[torch.Tensor], 
                       query_timestamps: torch.Tensor,
                       query_locations: torch.Tensor) -> torch.Tensor:
        """
        Predict smoke values at specified future times and locations.
        
        Args:
            context_obs: Model-specific context data (e.g. sequence of obs, or bundle of points).
            context_actions: Actions taken during context (for sequential models).
            future_actions: Actions to be taken in future (for sequential models).
            query_timestamps: 1D Tensor of relative times to predict (T_future).
            query_locations: Spatial locations to predict (N, 2) or (T, N, 2).
        
        Returns:
            predictions: Tensor of shape (T_future, N_locations) containing smoke values.
        """
        pass

    @abc.abstractmethod
    def load_checkpoint(self, path: str, device: str):
        pass

class SNPWrapper(SmokeModelWrapper):
    """
    Wrapper for Sequential Models (SNP v1/v2, RNP).
    Assumes models have a 'forward_sequence' or step-by-step interface.
    """
    def __init__(self, model_class, params_class):
        self.model_class = model_class
        self.params_class = params_class
        self.model = None
        self.device = "cpu"

    def load_checkpoint(self, path: str, device: str):
        self.device = device
        # Assuming static load_from_checkpoint method exists
        if hasattr(self.model_class, 'load_from_checkpoint'):
             self.model = self.model_class.load_from_checkpoint(path, device)
        else:
             # Fallback manual load
             checkpoint = torch.load(path, map_location=device)
             params = checkpoint['hyper_parameters'] # or similar
             self.model = self.model_class(params).to(device)
             self.model.load_state_dict(checkpoint['state_dict']) # or 'model_state_dict'

    def predict_future(self, context_obs: List[ObsSNP], context_actions: torch.Tensor, 
                       future_actions: torch.Tensor, query_timestamps: torch.Tensor, 
                       query_locations: torch.Tensor) -> torch.Tensor:
        
        self.model.eval()
        
        # 1. Rollout Context in RNN
        # We need to initialize state and feed context one by one
        batch_size = 1 # Evaluation usually episode by episode
        state = self.model.init_state(batch_size, self.device) if hasattr(self.model, 'init_state') else None 
        # RNP/SNPv1 might use different state init
        if state is None:
             # Handle legacy/RNP state init if needed (usually zeros)
             # For now assume mostly SnpV2 compliant or adapt
             pass
             
        # ... logic to run context ...
        # This requires normalizing the interface of SNPv1/RNP to match SNPv2 or branching here.
        # Given the request "Modulariza los modelos... Prioriza legibilidad", 
        # I should probably have standardized the models first or put the logic here.
        
        # For SnpV2:
        if isinstance(self.model, SnpV2):
             # Feed context
             dones = torch.zeros(batch_size, 1).to(self.device)
             for t, (obs, action) in enumerate(zip(context_obs, context_actions)):
                  out = self.model(state, action.unsqueeze(0), dones, obs=obs)
                  state = out.state
             
             # Forecast
             preds = []
             # future_actions: (T_future, action_dim)
             for t in range(len(future_actions)):
                  action = future_actions[t].unsqueeze(0)
                  # Query msg construction
                  # query_locations might be (N, 2). Repeated for each t?
                  # Or query_locations is (T, N, 2)?
                  # Let's assume (N, 2) fixed grid for all T.
                  q_locs = query_locations.to(self.device) 
                  # Create dummy query obs with just xs/ys
                  query_obs = ObsSNP(xs=q_locs[:, 0].unsqueeze(0), ys=q_locs[:, 1].unsqueeze(0))
                  
                  out = self.model(state, action, dones, obs=None, query=query_obs)
                  state = out.state
                  
                  # out.decoded is Normal or Tensor? SnpV2 returns decoded distribution or None?
                  # SnpV2 returns 'decoded' which is Normal.
                  # We take the mean.
                  preds.append(out.decoded.loc.squeeze(0)) # (N,)
             
             return torch.stack(preds) # (T, N)
             
        raise NotImplementedError("Only SnpV2 fully supported in wrapper currently")


class PINNWrapper(SmokeModelWrapper):
    def __init__(self, model_class):
         self.model_class = model_class
         self.model = None
         self.device = "cpu"
         
    def load_checkpoint(self, path: str, device: str):
         self.device = device
         # Assuming 'hyper_parameters' dict saved
         checkpoint = torch.load(path, map_location=device)
         # We might need to inspect constructor args or save them in params object
         # For PINNs, arguments usually passed to __init__.
         # If not saved as params object, we rely on hardcoding or smart Dict unpacking if saved
         # Simplified for now: assume params dict match __init__
         params = checkpoint.get('hyper_parameters', {}) 
         # Some params might be objects, usually they are ints/strings.
         self.model = self.model_class(**params).to(device)
         self.model.load_state_dict(checkpoint['state_dict']) # Check naming 'model_state_dict' or 'state_dict'

    def predict_future(self, context_obs: ObsPINN, context_actions, future_actions, 
                       query_timestamps: torch.Tensor, query_locations: torch.Tensor) -> torch.Tensor:
         """
         PINNs don't use actions (usually).
         context_obs is the set of (x,y,t,v) points.
         query_timestamps: Relative times t > t_context_end.
         """
         self.model.eval()
         
         # Query Construction
         # query_locations: (N, 2)
         # query_timestamps: (T_future,)
         
         # We need to broadcast locations x timestamps
         # (T, N, 3) -> [x, y, t] for all combinations
         
         num_t = len(query_timestamps)
         num_loc = len(query_locations)
         
         # Expand time: (T, N)
         ts_grid = query_timestamps.unsqueeze(1).expand(num_t, num_loc)
         # Expand locs: (T, N, 2)
         locs_grid = query_locations.unsqueeze(0).expand(num_t, num_loc, 2)
         
         # Flat query: (T*N, 3)
         # But ObsPINN expects (B, M). Batch size here is 1 episode usually.
         # So we pass B=1.
         
         xs = locs_grid[..., 0].flatten().unsqueeze(0).to(self.device)
         ys = locs_grid[..., 1].flatten().unsqueeze(0).to(self.device)
         ts = ts_grid.flatten().unsqueeze(0).to(self.device)
         
         query = ObsPINN(xs=xs, ys=ys, ts=ts)
         context_obs = context_obs.to(self.device)
         
         # Forward
         # PINNs expect Batch dim.
         # context_obs should have B=1.
         with torch.no_grad():
              output = self.model(context_obs, query)
              
         # Output smoke_dist.loc: (1, T*N, 1)
         preds = output.smoke_dist.loc.view(num_t, num_loc)
         return preds

