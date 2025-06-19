"""
Watermark Addition Utilities

This module provides utilities for adding watermarks to audio signals.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import tqdm
import time
import logging

from . import metric_util

logger = logging.getLogger(__name__)

# The pattern bits can be any random sequence.
# But don't use all-zeros, all-ones, or any periodic sequence, which will seriously hurt decoding performance.
fix_pattern = np.array([
    1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
    1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
    1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0
], dtype=np.int32)


def add_watermark(
    bit_arr: np.ndarray,
    data: np.ndarray,
    num_point: int,
    shift_range: float,
    device: torch.device,
    model: torch.nn.Module,
    min_snr: float,
    max_snr: float,
    show_progress: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Add watermark to audio data with SNR constraints.
    
    Args:
        bit_arr: Watermark bit array
        data: Input audio data
        num_point: Number of points per chunk
        shift_range: Shift range for overlapping
        device: Torch device
        model: WavMark model
        min_snr: Minimum SNR threshold
        max_snr: Maximum SNR threshold
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (watermarked_data, info_dict)
    """
    start_time = time.time()
    
    # Calculate chunk parameters
    chunk_size = num_point + int(num_point * shift_range)
    num_segments = len(data) // chunk_size
    len_remain = len(data) - num_segments * chunk_size
    
    if num_segments == 0:
        logger.warning("Audio too short for watermarking, returning original")
        return data, {"time_cost": 0, "encoded_sections": 0, "skip_sections": 0}
    
    output_chunks = []
    encoded_sections = 0
    skip_sections = 0
    
    # Process segments
    iterator = range(num_segments)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Adding watermark")
    
    for i in iterator:
        start_point = i * chunk_size
        current_chunk = data[start_point:start_point + chunk_size].copy()
        
        # Split into cover area and shift area
        current_chunk_cover_area = current_chunk[:num_point]
        current_chunk_shift_area = current_chunk[num_point:]
        
        # Encode cover area with SNR check
        current_chunk_cover_area_wmd, state = encode_chunk_with_snr_check(
            i, current_chunk_cover_area, bit_arr, device, model, min_snr, max_snr
        )
        
        if state == "skip":
            skip_sections += 1
        else:
            encoded_sections += 1
        
        # Reconstruct chunk
        output = np.concatenate([current_chunk_cover_area_wmd, current_chunk_shift_area])
        assert output.shape == current_chunk.shape, f"Shape mismatch: {output.shape} vs {current_chunk.shape}"
        output_chunks.append(output)
    
    # Handle remaining data
    if len_remain > 0:
        output_chunks.append(data[-len_remain:])
    
    # Reconstruct full signal
    reconstructed_array = np.concatenate(output_chunks)
    time_cost = time.time() - start_time
    
    info = {
        "time_cost": time_cost,
        "encoded_sections": encoded_sections,
        "skip_sections": skip_sections,
        "total_sections": num_segments,
        "success_rate": encoded_sections / max(num_segments, 1)
    }
    
    logger.info(f"Watermarking completed: {encoded_sections}/{num_segments} sections encoded "
                f"({info['success_rate']:.1%} success rate) in {time_cost:.2f}s")
    
    return reconstructed_array, info


def encode_chunk_with_snr_check(
    idx_chunk: int,
    signal: np.ndarray,
    wm: np.ndarray,
    device: torch.device,
    model: torch.nn.Module,
    min_snr: float,
    max_snr: float
) -> Tuple[np.ndarray, Union[str, int]]:
    """
    Encode a chunk with SNR constraint checking.
    
    Args:
        idx_chunk: Chunk index
        signal: Input signal chunk
        wm: Watermark bits
        device: Torch device
        model: WavMark model
        min_snr: Minimum SNR threshold
        max_snr: Maximum SNR threshold
        
    Returns:
        Tuple of (encoded_signal, state_or_attempts)
    """
    signal_for_encode = signal.copy()
    encode_times = 0
    max_attempts = 10
    
    while encode_times < max_attempts:
        encode_times += 1
        signal_wmd = encode_chunk(signal_for_encode, wm, device, model)
        snr = metric_util.signal_noise_ratio(signal, signal_wmd)
        
        # Check if SNR is too low on first attempt
        if encode_times == 1 and snr < min_snr:
            logger.debug(f"Skip section {idx_chunk}: SNR too low ({snr:.1f} < {min_snr})")
            return signal, "skip"
        
        # Check if SNR is within acceptable range
        if snr <= max_snr:
            return signal_wmd, encode_times
        
        # SNR is too high, use current result as input for next iteration
        signal_for_encode = signal_wmd
    
    # Max attempts reached
    logger.debug(f"Section {idx_chunk}: Max attempts ({max_attempts}) reached, final SNR: {snr:.1f}")
    return signal_wmd, encode_times


def encode_chunk(
    chunk: np.ndarray,
    wm: np.ndarray,
    device: torch.device,
    model: torch.nn.Module
) -> np.ndarray:
    """
    Encode a single chunk with watermark.
    
    Args:
        chunk: Input audio chunk
        wm: Watermark bits
        device: Torch device
        model: WavMark model
        
    Returns:
        Encoded audio chunk
    """
    with torch.no_grad():
        try:
            signal = torch.FloatTensor(chunk).to(device).unsqueeze(0)
            message = torch.FloatTensor(wm).to(device).unsqueeze(0)
            
            signal_wmd_tensor = model.encode(signal, message)
            signal_wmd = signal_wmd_tensor.detach().cpu().numpy().squeeze()
            
            return signal_wmd
            
        except Exception as e:
            logger.error(f"Error encoding chunk: {e}")
            # Return original signal on error
            return chunk


def validate_watermark_bits(bit_arr: np.ndarray, expected_length: int = 32) -> bool:
    """
    Validate watermark bit array.
    
    Args:
        bit_arr: Bit array to validate
        expected_length: Expected length of bit array
        
    Returns:
        True if valid, False otherwise
    """
    if len(bit_arr) != expected_length:
        logger.error(f"Invalid watermark length: {len(bit_arr)}, expected {expected_length}")
        return False
    
    if not np.all(np.isin(bit_arr, [0, 1])):
        logger.error("Watermark contains non-binary values")
        return False
    
    return True
