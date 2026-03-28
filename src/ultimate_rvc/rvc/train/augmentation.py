"""
Audio augmentation transforms for training data.

Applies random augmentations to audio during training to improve
model generalization and robustness, especially useful for singing
voice conversion with limited data.
"""

from __future__ import annotations

import random

import torch


class TrainingAugmentor:
    """
    Applies random audio augmentations during training.

    Augmentations include:
    - Gaussian noise injection (SNR 20-40dB)
    - Volume perturbation (±3dB)
    - Random gain scaling

    Parameters
    ----------
    probability : float
        Probability of applying augmentation to each sample.
        Default is 0.5.
    noise_snr_range : tuple[float, float]
        Range of signal-to-noise ratio in dB for noise injection.
        Default is (20.0, 40.0).
    volume_range_db : tuple[float, float]
        Range of volume perturbation in dB.
        Default is (-3.0, 3.0).

    """

    def __init__(
        self,
        probability: float = 0.5,
        noise_snr_range: tuple[float, float] = (20.0, 40.0),
        volume_range_db: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        self.probability = probability
        self.noise_snr_range = noise_snr_range
        self.volume_range_db = volume_range_db

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation to a waveform tensor.

        Parameters
        ----------
        wav : torch.Tensor
            Input waveform tensor of shape (1, T).

        Returns
        -------
        torch.Tensor
            Augmented waveform tensor.

        """
        if random.random() > self.probability:
            return wav

        # Apply one random augmentation
        augmentation = random.choice(["noise", "volume"])

        if augmentation == "noise":
            wav = self._add_gaussian_noise(wav)
        elif augmentation == "volume":
            wav = self._perturb_volume(wav)

        return wav

    def _add_gaussian_noise(self, wav: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at a random SNR level."""
        snr_db = random.uniform(*self.noise_snr_range)
        signal_power = wav.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(wav) * noise_power.sqrt()
        return wav + noise

    def _perturb_volume(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply random volume perturbation."""
        gain_db = random.uniform(*self.volume_range_db)
        gain = 10 ** (gain_db / 20)
        return wav * gain
