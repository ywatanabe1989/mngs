#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-31 (ywatanabe)"
# File: neural_network_layers.py

__file__ = "neural_network_layers.py"

"""
Neural Network Layer Examples for MNGS

This example demonstrates the custom neural network layers available in mngs.nn,
which are designed for signal processing and neuroscience applications.

Key features demonstrated:
1. Signal processing layers (filters, transforms)
2. Augmentation layers for training
3. Analysis layers (PSD, PAC, wavelets)
4. Integration with PyTorch models
"""

"""Imports"""
import argparse

"""Warnings"""
# mngs.pd.ignore_SettingWithCopyWarning()
# warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()


def demo_signal_processing_layers():
    """Demonstrate signal processing neural network layers."""
    print("\n=== Signal Processing Layers ===")
    
    # Generate example multi-channel signal
    n_channels = 8
    n_samples = 1000
    sample_rate = 250.0
    
    # Create signal with multiple frequency components
    t = np.linspace(0, n_samples/sample_rate, n_samples)
    signal = np.zeros((1, n_channels, n_samples))  # batch x channels x time
    
    for ch in range(n_channels):
        # Add different frequency components to each channel
        signal[0, ch, :] = (
            np.sin(2 * np.pi * 10 * t) +  # 10 Hz
            0.5 * np.sin(2 * np.pi * 25 * t) +  # 25 Hz
            0.3 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz (noise)
            0.1 * np.random.randn(n_samples)  # Random noise
        )
    
    signal_tensor = torch.FloatTensor(signal)
    
    # 1. Bandpass filter layer
    print("\n1. Bandpass Filter Layer")
    filter_layer = mngs.nn.Filters(
        n_chs=n_channels,
        samp_rate=sample_rate,
        low_hz=8.0,
        high_hz=30.0,
        learnable=False  # Can be made learnable
    )
    
    filtered_signal = filter_layer(signal_tensor)
    print(f"   Input shape: {signal_tensor.shape}")
    print(f"   Output shape: {filtered_signal.shape}")
    
    # 2. Hilbert transform layer
    print("\n2. Hilbert Transform Layer")
    hilbert_layer = mngs.nn.Hilbert(
        n_chs=n_channels,
        samp_rate=sample_rate,
        low_hz=8.0,
        high_hz=12.0,
        return_envelope=True
    )
    
    envelope = hilbert_layer(signal_tensor)
    print(f"   Envelope shape: {envelope.shape}")
    
    # 3. Power Spectral Density layer
    print("\n3. PSD Layer")
    psd_layer = mngs.nn.PSD(
        n_chs=n_channels,
        samp_rate=sample_rate,
        window_size=256,
        overlap=0.5
    )
    
    psd_features = psd_layer(signal_tensor)
    print(f"   PSD features shape: {psd_features.shape}")
    
    # 4. Wavelet transform layer
    print("\n4. Wavelet Transform Layer")
    wavelet_layer = mngs.nn.Wavelet(
        n_chs=n_channels,
        samp_rate=sample_rate,
        freqs=np.logspace(0, 1.7, 20),  # 1-50 Hz
        n_cycles=7
    )
    
    tfr = wavelet_layer(signal_tensor)
    print(f"   Time-frequency shape: {tfr.shape}")
    
    # Save example outputs
    mngs.io.save(signal[0].T, "raw_signal.npy")
    mngs.io.save(filtered_signal[0].detach().numpy().T, "filtered_signal.npy")
    mngs.io.save(envelope[0].detach().numpy().T, "envelope.npy")
    
    return signal_tensor, filtered_signal, envelope, psd_features, tfr


def demo_augmentation_layers():
    """Demonstrate data augmentation layers for neural networks."""
    print("\n=== Data Augmentation Layers ===")
    
    # Create example data
    batch_size = 4
    n_channels = 8
    n_samples = 500
    
    data = torch.randn(batch_size, n_channels, n_samples)
    
    # 1. Channel dropout
    print("\n1. Channel Dropout")
    channel_dropout = mngs.nn.DropoutChannels(p=0.2)
    channel_dropout.train()  # Enable dropout
    
    augmented = channel_dropout(data)
    dropped_channels = (augmented == 0).any(dim=2).sum().item()
    print(f"   Dropped {dropped_channels} channels out of {batch_size * n_channels}")
    
    # 2. Channel swapping
    print("\n2. Channel Swapping")
    swap_layer = mngs.nn.SwapChannels(p=0.5)
    swap_layer.train()
    
    swapped = swap_layer(data)
    print(f"   Swapped channels in batch (probabilistic)")
    
    # 3. Frequency gain changer
    print("\n3. Frequency Gain Changer")
    freq_gain = mngs.nn.FreqGainChanger(
        n_chs=n_channels,
        samp_rate=250.0,
        freq_bands=[(8, 12), (13, 30), (30, 50)],
        gain_range=(0.8, 1.2)
    )
    freq_gain.train()
    
    freq_augmented = freq_gain(data)
    print(f"   Applied frequency-specific gain changes")
    
    # 4. Channel gain changer
    print("\n4. Channel Gain Changer")
    channel_gain = mngs.nn.ChannelGainChanger(
        gain_range=(0.9, 1.1),
        p=0.5
    )
    channel_gain.train()
    
    gain_augmented = channel_gain(data)
    print(f"   Applied channel-specific gain changes")
    
    return data, augmented, swapped, freq_augmented, gain_augmented


def demo_analysis_layers():
    """Demonstrate neural network layers for signal analysis."""
    print("\n=== Signal Analysis Layers ===")
    
    # Generate example signal with PAC
    n_samples = 2000
    sample_rate = 250.0
    t = np.linspace(0, n_samples/sample_rate, n_samples)
    
    # Low frequency phase signal (10 Hz)
    phase_signal = np.sin(2 * np.pi * 10 * t)
    
    # High frequency amplitude signal (40 Hz) modulated by phase
    amp_signal = (1 + 0.5 * phase_signal) * np.sin(2 * np.pi * 40 * t)
    
    # Combine signals
    signal = torch.FloatTensor(phase_signal + 0.5 * amp_signal).unsqueeze(0).unsqueeze(0)
    
    # 1. Phase-Amplitude Coupling layer
    print("\n1. PAC Layer")
    pac_layer = mngs.nn.PAC(
        n_chs=1,
        samp_rate=sample_rate,
        phase_freqs=[(8, 12)],
        amp_freqs=[(30, 50)],
        method='tort'
    )
    
    pac_values = pac_layer(signal)
    print(f"   PAC values shape: {pac_values.shape}")
    print(f"   PAC strength: {pac_values.item():.3f}")
    
    # 2. Modulation Index layer
    print("\n2. Modulation Index Layer")
    mi_layer = mngs.nn.ModulationIndex(
        n_chs=1,
        samp_rate=sample_rate,
        phase_freq=(8, 12),
        amp_freq=(30, 50)
    )
    
    mi_values = mi_layer(signal)
    print(f"   MI values: {mi_values.item():.3f}")
    
    # 3. Spectrogram layer
    print("\n3. Spectrogram Layer")
    spectrogram_layer = mngs.nn.Spectrogram(
        n_chs=1,
        samp_rate=sample_rate,
        window_size=128,
        overlap=0.75
    )
    
    spec = spectrogram_layer(signal)
    print(f"   Spectrogram shape: {spec.shape}")
    
    return signal, pac_values, mi_values, spec


def demo_complete_model():
    """Demonstrate a complete neural network model using mngs layers."""
    print("\n=== Complete Neural Network Model ===")
    
    class SignalClassifier(nn.Module):
        """Example model combining multiple mngs layers."""
        
        def __init__(self, n_channels=8, n_classes=2):
            super().__init__()
            
            # Signal processing layers
            self.filter = mngs.nn.Filters(
                n_chs=n_channels,
                samp_rate=250.0,
                low_hz=8.0,
                high_hz=30.0,
                learnable=True  # Learnable filter parameters
            )
            
            # Feature extraction
            self.psd = mngs.nn.PSD(
                n_chs=n_channels,
                samp_rate=250.0,
                window_size=256,
                overlap=0.5
            )
            
            # Augmentation (only during training)
            self.dropout = mngs.nn.DropoutChannels(p=0.1)
            self.gain = mngs.nn.ChannelGainChanger(gain_range=(0.9, 1.1))
            
            # Get feature dimension
            dummy_input = torch.randn(1, n_channels, 1000)
            dummy_filtered = self.filter(dummy_input)
            dummy_features = self.psd(dummy_filtered)
            feature_dim = dummy_features.shape[1]
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes)
            )
            
        def forward(self, x):
            # Apply augmentation only during training
            if self.training:
                x = self.dropout(x)
                x = self.gain(x)
            
            # Signal processing
            x = self.filter(x)
            
            # Feature extraction
            features = self.psd(x)
            
            # Classification
            output = self.classifier(features)
            
            return output
    
    # Create model
    model = SignalClassifier(n_channels=8, n_classes=2)
    print(f"\nModel architecture created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    batch_size = 16
    n_channels = 8
    n_samples = 1000
    
    example_input = torch.randn(batch_size, n_channels, n_samples)
    
    # Training mode
    model.train()
    train_output = model(example_input)
    print(f"\nTraining mode output shape: {train_output.shape}")
    
    # Evaluation mode
    model.eval()
    with torch.no_grad():
        eval_output = model(example_input)
    print(f"Evaluation mode output shape: {eval_output.shape}")
    
    # Save model architecture
    model_info = {
        'architecture': str(model),
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'layer_info': {
            'filter': 'Learnable bandpass filter',
            'psd': 'Power spectral density features',
            'augmentation': 'Channel dropout and gain changes',
            'classifier': '3-layer MLP'
        }
    }
    mngs.io.save(model_info, "model_info.json")
    
    return model


def visualize_layer_outputs():
    """Create visualizations of layer outputs."""
    print("\n=== Creating Visualizations ===")
    
    # Setup plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate test signal
    n_samples = 1000
    sample_rate = 250.0
    t = np.linspace(0, n_samples/sample_rate, n_samples)
    
    # Multi-frequency signal
    signal = (
        np.sin(2 * np.pi * 10 * t) +
        0.5 * np.sin(2 * np.pi * 25 * t) +
        0.3 * np.sin(2 * np.pi * 40 * t)
    )
    signal_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
    
    # 1. Original vs Filtered
    ax = axes[0, 0]
    filter_layer = mngs.nn.Filters(n_chs=1, samp_rate=sample_rate, low_hz=8, high_hz=30)
    filtered = filter_layer(signal_tensor)[0, 0].detach().numpy()
    
    ax.plot(t[:250], signal[:250], 'b-', alpha=0.7, label='Original')
    ax.plot(t[:250], filtered[:250], 'r-', alpha=0.7, label='Filtered (8-30 Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Bandpass Filtering')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Envelope extraction
    ax = axes[0, 1]
    hilbert_layer = mngs.nn.Hilbert(n_chs=1, samp_rate=sample_rate, low_hz=8, high_hz=12)
    envelope = hilbert_layer(signal_tensor)[0, 0].detach().numpy()
    
    ax.plot(t[:500], filtered[:500], 'b-', alpha=0.5, label='Filtered signal')
    ax.plot(t[:500], envelope[:500], 'r-', linewidth=2, label='Envelope')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Hilbert Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PSD features
    ax = axes[1, 0]
    psd_layer = mngs.nn.PSD(n_chs=1, samp_rate=sample_rate, window_size=256)
    psd_features = psd_layer(signal_tensor)[0].detach().numpy()
    
    freqs = np.linspace(0, sample_rate/2, len(psd_features))
    ax.semilogy(freqs, psd_features, 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectral Density')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 60])
    
    # 4. Time-frequency representation
    ax = axes[1, 1]
    wavelet_layer = mngs.nn.Wavelet(
        n_chs=1,
        samp_rate=sample_rate,
        freqs=np.linspace(5, 50, 20),
        n_cycles=7
    )
    tfr = wavelet_layer(signal_tensor)[0, 0].detach().numpy()
    
    im = ax.imshow(
        np.abs(tfr),
        aspect='auto',
        origin='lower',
        extent=[0, n_samples/sample_rate, 5, 50],
        cmap='hot'
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Wavelet Transform')
    plt.colorbar(im, ax=ax, label='Power')
    
    plt.tight_layout()
    mngs.io.save(fig, "layer_outputs.png")
    plt.close()
    
    print("   Saved visualization to layer_outputs.png")


"""Functions & Classes"""
def main(args):
    """Run all neural network layer examples."""
    print("=" * 60)
    print("MNGS Neural Network Layer Examples")
    print("=" * 60)
    
    # Create output directory (handled by mngs.gen.start)
    mngs.io.save({"example": "nn_layers"}, "info.json")
    
    # Run demonstrations
    signal_outputs = demo_signal_processing_layers()
    augmentation_outputs = demo_augmentation_layers()
    analysis_outputs = demo_analysis_layers()
    model = demo_complete_model()
    visualize_layer_outputs()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary of MNGS Neural Network Layers:")
    print("=" * 60)
    print("\nSignal Processing Layers:")
    print("  - Filters: Learnable/fixed bandpass, lowpass, highpass")
    print("  - Hilbert: Envelope and phase extraction")
    print("  - PSD: Power spectral density features")
    print("  - Wavelet: Time-frequency decomposition")
    print("  - Spectrogram: Short-time Fourier transform")
    
    print("\nAugmentation Layers:")
    print("  - DropoutChannels: Channel-wise dropout")
    print("  - SwapChannels: Random channel permutation")
    print("  - FreqGainChanger: Frequency-band specific gain")
    print("  - ChannelGainChanger: Channel-specific gain")
    
    print("\nAnalysis Layers:")
    print("  - PAC: Phase-amplitude coupling")
    print("  - ModulationIndex: Cross-frequency coupling strength")
    print("  - Various feature extraction layers")
    
    print("\nKey Features:")
    print("  - PyTorch compatible")
    print("  - Differentiable signal processing")
    print("  - GPU acceleration support")
    print("  - Learnable parameters option")
    print("  - Batch processing")
    
    print(f"\nOutputs saved to: {CONFIG.SDIR}/")
    print("\nThese layers can be combined with standard PyTorch layers")
    print("to create powerful signal processing neural networks!")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="Neural network layer examples with mngs.nn")
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, np, torch, nn, mngs

    import sys
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()