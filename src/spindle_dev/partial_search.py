"""
Implementations for partial queries in Spindle.

Provides three complementary strategies to handle missing genes in target SPD matrices.
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def pad_query_identity(
    query_matrix: np.ndarray, 
    query_indices: List[int], 
    full_size: int
) -> np.ndarray:
    """
    1. The "Identity Padding" Trick 
    
    Pads a partial query matrix with zeros and 1s on the diagonal for missing genes.
    In log-space, ln(1) = 0, so the log of the padded matrix will have perfect zeros 
    in the missing columns/rows. This naturally penalizes the distance based on the 
    variance of the missing genes in the target matrix.
    
    Args:
        query_matrix: The partial covariance matrix provided by the user.
        query_indices: The indices corresponding to the queried genes in the full matrix.
        full_size: The total expected number of genes (size of the full target matrices).
        
    Returns:
        The padded full-size matrix.
    """
    if query_matrix.shape[0] != len(query_indices):
        raise ValueError("query_matrix size must match length of query_indices")
        
    padded_matrix = np.eye(full_size, dtype=query_matrix.dtype)
    ix = np.ix_(query_indices, query_indices)
    padded_matrix[ix] = query_matrix
    
    return padded_matrix


def impute_query_niche_mean(
    query_matrix: np.ndarray, 
    query_indices: List[int], 
    niche_mean_matrix: np.ndarray
) -> np.ndarray:
    """
    2. Niche-Mean Imputation 
    
    Instead of padding with identity, imputes missing rows and columns using the 
    mean matrix of the predicted Niche (cluster). This reduces
    the implicit distance penalty for missing genes, focusing the DAG search purely
    on the variance of the known valid genes.
    
    Args:
        query_matrix: The partial covariance matrix provided by the user.
        query_indices: The indices corresponding to the queried genes in the full matrix.
        niche_mean_matrix: The pre-calculated mean matrix for the predicted Niche (data.cluster_means).
        
    Returns:
        The full-size matrix with imputed missing dimensions.
    """
    if query_matrix.shape[0] != len(query_indices):
        raise ValueError("query_matrix size must match length of query_indices")
        
    imputed_matrix = niche_mean_matrix.copy()
    ix = np.ix_(query_indices, query_indices)
    imputed_matrix[ix] = query_matrix
    
    return imputed_matrix