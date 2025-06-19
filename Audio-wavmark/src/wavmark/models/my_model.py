"""
WavMark Model Implementation

This module contains the main WavMark model for audio watermarking.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np
from functools import lru_cache

from .hinet import Hinet


class Model(nn.Module):
    """
    WavMark model for audio watermarking.
    
    This model uses STFT/ISTFT for frequency domain processing and Hinet for
    the core watermarking operations.
    """
    
    def __init__(self, num_point: int, num_bit: int, n_fft: int, hop_length: int, num_layers: int):
        """
        Initialize WavMark model.
        
        Args:
            num_point: Number of audio points to process
            num_bit: Number of watermark bits
            n_fft: FFT window size
            hop_length: Hop length for STFT
            num_layers: Number of Hinet layers
        """
        super(Model, self).__init__()
        self.hinet = Hinet(num_layers=num_layers)
        self.watermark_fc = nn.Linear(num_bit, num_point)
        self.watermark_fc_back = nn.Linear(num_point, num_bit)
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Cache for window functions
        self._window_cache = {}
    
    def _get_window(self, device: torch.device) -> torch.Tensor:
        """Get or create Hann window for the specified device."""
        if device not in self._window_cache:
            self._window_cache[device] = torch.hann_window(self.n_fft, device=device)
        return self._window_cache[device]
    
    def stft(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            data: Input audio tensor [batch, samples]
            
        Returns:
            STFT tensor [batch, freq_bins, time_frames, 2]
        """
        window = self._get_window(data.device)
        
        # Use return_complex=True for modern PyTorch compatibility
        stft_complex = torch.stft(
            data, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=window, 
            return_complex=True
        )
        
        # Convert to real format [batch, freq_bins, time_frames, 2]
        return torch.view_as_real(stft_complex)
    
    def istft(self, signal_wmd_fft: torch.Tensor) -> torch.Tensor:
        """
        Compute Inverse Short-Time Fourier Transform.
        
        Args:
            signal_wmd_fft: STFT tensor [batch, freq_bins, time_frames, 2]
            
        Returns:
            Reconstructed audio tensor [batch, samples]
        """
        window = self._get_window(signal_wmd_fft.device)
        
        # Convert back to complex format
        stft_complex = torch.view_as_complex(signal_wmd_fft)
        
        return torch.istft(
            stft_complex, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=window, 
            return_complex=False
        )
    
    def encode(self, signal: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Encode watermark into audio signal.
        
        Args:
            signal: Input audio tensor [batch, samples]
            message: Watermark message tensor [batch, num_bits]
            
        Returns:
            Watermarked audio tensor [batch, samples]
        """
        signal_fft = self.stft(signal)
        
        # Expand message to audio length
        message_expand = self.watermark_fc(message)
        message_fft = self.stft(message_expand)
        
        # Apply encoding
        signal_wmd_fft, _ = self.enc_dec(signal_fft, message_fft, rev=False)
        
        # Convert back to time domain
        signal_wmd = self.istft(signal_wmd_fft)
        return signal_wmd
    
    def decode(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Decode watermark from audio signal.
        
        Args:
            signal: Audio tensor [batch, samples]
            
        Returns:
            Decoded message tensor [batch, num_bits]
        """
        signal_fft = self.stft(signal)
        watermark_fft = signal_fft
        
        # Apply decoding
        _, message_restored_fft = self.enc_dec(signal_fft, watermark_fft, rev=True)
        
        # Convert back to time domain and extract message
        message_restored_expanded = self.istft(message_restored_fft)
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        
        return message_restored_float
    
    def enc_dec(self, signal: torch.Tensor, watermark: torch.Tensor, rev: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core encoding/decoding operation using Hinet.
        
        Args:
            signal: Signal tensor [batch, freq_bins, time_frames, 2]
            watermark: Watermark tensor [batch, freq_bins, time_frames, 2]
            rev: If True, perform decoding; if False, perform encoding
            
        Returns:
            Tuple of (processed_signal, processed_watermark)
        """
        # Permute dimensions for Hinet: [batch, freq_bins, time_frames, 2] -> [batch, 2, time_frames, freq_bins]
        signal_permuted = signal.permute(0, 3, 2, 1)
        watermark_permuted = watermark.permute(0, 3, 2, 1)
        
        # Apply Hinet transformation
        signal2, watermark2 = self.hinet(signal_permuted, watermark_permuted, rev)
        
        # Permute back: [batch, 2, time_frames, freq_bins] -> [batch, freq_bins, time_frames, 2]
        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)
    
    def forward(self, signal: torch.Tensor, message: Optional[torch.Tensor] = None, mode: str = 'encode') -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            signal: Input audio tensor
            message: Watermark message tensor (required for encoding)
            mode: 'encode' or 'decode'
            
        Returns:
            Processed tensor
        """
        if mode == 'encode':
            if message is None:
                raise ValueError("Message is required for encoding mode")
            return self.encode(signal, message)
        elif mode == 'decode':
            return self.decode(signal)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'encode' or 'decode'")
