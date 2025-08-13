import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_mean


class CartesianCoordinates:
    
    @staticmethod
    def normalize_coordinates(coords, center = False):
       
        if center:
            coords = coords - coords.mean(dim=0, keepdim=True)
        return F.normalize(coords, p=2, dim=1)

    @staticmethod
    def scale_coordinates(coords, radius):
        return coords * radius


class PolarCoordinates:
    
    @staticmethod
    def to_cartesian(angles, radius, constrain_radius, radius_range):
        
        # Scale angles from [-1, 1] to [-π, π]
        angles = angles * torch.pi
        
        # Optionally constrain radius
        if constrain_radius:
            min_r, max_r = radius_range
            radius = min_r + (max_r - min_r) * torch.sigmoid(radius)
        
        # Convert to Cartesian coordinates
        coords = torch.zeros(angles.size(0), 2, device=angles.device)
        coords[:, 0] = radius.squeeze() * torch.cos(angles.squeeze())  # x = r * cos(θ)
        coords[:, 1] = radius.squeeze() * torch.sin(angles.squeeze())  # y = r * sin(θ)
        
        return coords
    
    @staticmethod
    def from_cartesian(coords):       
        x, y = coords[:, 0], coords[:, 1]
        radius = torch.sqrt(x*x + y*y)
        angles = torch.atan2(y, x)
        return angles, radius


class CoordinateNormalizer:
    
    @staticmethod
    def normalize_with_radius(coords, radius, center = True):
        
        coords_norm = CartesianCoordinates.normalize_coordinates(coords, center=center)
        return CartesianCoordinates.scale_coordinates(coords_norm, radius)
    
    @staticmethod
    def normalize_with_constraints(coords, min_radius, max_radius):
        # Center and normalize
        coords = coords - coords.mean(dim=0, keepdim=True)
        coords_norm = F.normalize(coords, p=2, dim=1)
        
        # Calculate current radius
        radius = torch.sqrt(torch.sum(coords * coords, dim=1, keepdim=True))
        
        # Constrain radius
        radius_constrained = torch.clamp(radius, min_radius, max_radius)
        
        return coords_norm * radius_constrained


class LayoutProcessor:
    
    def __init__(self, use_polar = False):
        
        self.use_polar = use_polar
    
    def process_coordinates(self, coords_or_angles, radius, constrain_radius = True, center = True):
        
        if self.use_polar:
            if radius is None:
                raise ValueError("Radius required for polar coordinate processing")
            # Convert polar to Cartesian and normalize
            coords = PolarCoordinates.to_cartesian(coords_or_angles, radius, constrain_radius)
            return CartesianCoordinates.normalize_coordinates(coords, center=center)
        else:
            # Process Cartesian coordinates
            if radius is not None:
                return CoordinateNormalizer.normalize_with_radius(coords_or_angles, radius, center)
            else:
                return CartesianCoordinates.normalize_coordinates(coords_or_angles, center) 


class ForceDirectedProcessor(nn.Module):

    def __init__(self, in_feat: int):

        super().__init__()  # Add proper nn.Module initialization
        self.norm_before_rep = nn.LayerNorm(in_feat)
        self.norm_after_rep = nn.LayerNorm(in_feat)
        self.w = nn.Parameter(torch.ones(in_feat))
    
    def process_force_messages(self, x_input: torch.Tensor, neighbor_msg: torch.Tensor, 
                             col: torch.Tensor, size: int) -> tuple:

        # Aggregate neighbor messages with safety checks
        agg_neighbors = scatter_mean(neighbor_msg, col, dim=0, dim_size=size)
        agg_neighbors = self.norm_before_rep(agg_neighbors)
        
        # Calculate repulsion
        repulsion = (x_input - agg_neighbors) * self.w
        
        # Final processing
        fx = x_input + repulsion
        fx = self.norm_after_rep(fx)
        
        return fx, agg_neighbors

    @staticmethod
    def reshape_coordinates(coords: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # For force-directed layouts with variable graph sizes, return coordinates as-is
        # The batch tensor is used by PyG to track which nodes belong to which graph
        # We don't need to reshape since each node's coordinates are independent
        return coords 