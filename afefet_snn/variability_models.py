"""
Variability Models for AFeFET Devices

Implements realistic device-to-device and cycle-to-cycle variations
for robust neuromorphic computing.

References:
- Device variations: typically 3-10% in ferroelectric devices
- Cycle-to-cycle: ~2.1% (paper baseline)
- Temperature variation: ±20°C around 300K
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class VariabilityModel:
    """
    Comprehensive variability model for AFeFET devices

    Includes:
    1. Device-to-Device (D2D) variation: Fixed per device
    2. Cycle-to-Cycle (C2C) variation: Random per write
    3. Temperature variation: Environmental effects
    4. Retention variation: Time-dependent uncertainty
    """

    def __init__(
        self,
        d2d_V_coercive_std: float = 0.10,  # 10% V_c variation
        d2d_capacitance_std: float = 0.05,  # 5% C variation
        d2d_area_ratio_std: float = 0.08,   # 8% area ratio variation
        c2c_write_std: float = 0.021,       # 2.1% write noise (paper)
        c2c_read_std: float = 0.003,        # 0.3% read noise
        temp_range: Tuple[float, float] = (280, 320),  # ±20°C around 300K
        retention_variation_std: float = 0.15,  # 15% retention time variation
        enable_d2d: bool = True,
        enable_c2c: bool = True,
        enable_temp: bool = False,
        enable_retention_var: bool = False,
        seed: Optional[int] = None
    ):
        """
        Args:
            d2d_V_coercive_std: Device-to-device coercive voltage std (relative)
            d2d_capacitance_std: Capacitance variation std (relative)
            d2d_area_ratio_std: Area ratio variation std (relative)
            c2c_write_std: Cycle-to-cycle write variation (paper: 2.1%)
            c2c_read_std: Read noise std (paper: 0.3%)
            temp_range: (T_min, T_max) in Kelvin
            retention_variation_std: Retention time constant variation
            enable_d2d: Enable device-to-device variation
            enable_c2c: Enable cycle-to-cycle variation
            enable_temp: Enable temperature variation
            enable_retention_var: Enable retention variation
            seed: Random seed for reproducibility
        """
        self.d2d_V_coercive_std = d2d_V_coercive_std
        self.d2d_capacitance_std = d2d_capacitance_std
        self.d2d_area_ratio_std = d2d_area_ratio_std
        self.c2c_write_std = c2c_write_std
        self.c2c_read_std = c2c_read_std
        self.temp_range = temp_range
        self.retention_variation_std = retention_variation_std

        self.enable_d2d = enable_d2d
        self.enable_c2c = enable_c2c
        self.enable_temp = enable_temp
        self.enable_retention_var = enable_retention_var

        if seed is not None:
            torch.manual_seed(seed)

        # Cache for device-to-device variations (fixed per device)
        self._d2d_cache = {}

    def generate_d2d_variations(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        key: str = "default"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate device-to-device variations (fixed per physical device)

        Returns:
            dict with keys: 'V_coercive_factor', 'capacitance_factor', 'area_ratio_factor'
        """
        if not self.enable_d2d:
            return {
                'V_coercive_factor': torch.ones(shape, device=device),
                'capacitance_factor': torch.ones(shape, device=device),
                'area_ratio_factor': torch.ones(shape, device=device),
                'retention_factor': torch.ones(shape, device=device)
            }

        # Use cache if available (device parameters are fixed)
        cache_key = f"{key}_{shape}_{device}"
        if cache_key in self._d2d_cache:
            return self._d2d_cache[cache_key]

        # Generate new D2D variations
        variations = {
            'V_coercive_factor': torch.clamp(
                torch.normal(1.0, self.d2d_V_coercive_std, shape, device=device),
                0.7, 1.3  # ±30% bounds
            ),
            'capacitance_factor': torch.clamp(
                torch.normal(1.0, self.d2d_capacitance_std, shape, device=device),
                0.8, 1.2  # ±20% bounds
            ),
            'area_ratio_factor': torch.clamp(
                torch.normal(1.0, self.d2d_area_ratio_std, shape, device=device),
                0.75, 1.25  # ±25% bounds
            ),
            'retention_factor': torch.clamp(
                torch.normal(1.0, self.retention_variation_std, shape, device=device),
                0.5, 1.5  # ±50% bounds (retention has high variation)
            ) if self.enable_retention_var else torch.ones(shape, device=device)
        }

        self._d2d_cache[cache_key] = variations
        return variations

    def apply_c2c_write_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Apply cycle-to-cycle write variation"""
        if not self.enable_c2c:
            return value

        noise = torch.randn_like(value) * self.c2c_write_std
        return value * (1 + noise)

    def apply_c2c_read_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Apply cycle-to-cycle read variation"""
        if not self.enable_c2c:
            return value

        noise = torch.randn_like(value) * self.c2c_read_std
        return value * (1 + noise)

    def sample_temperature(self, device: torch.device) -> torch.Tensor:
        """Sample operating temperature from range"""
        if not self.enable_temp:
            return torch.tensor(300.0, device=device)  # Nominal 300K

        T_min, T_max = self.temp_range
        return torch.FloatTensor(1).uniform_(T_min, T_max).to(device).item()

    def get_summary(self) -> Dict[str, any]:
        """Get variability configuration summary"""
        return {
            'D2D_enabled': self.enable_d2d,
            'D2D_V_coercive_std': self.d2d_V_coercive_std,
            'D2D_capacitance_std': self.d2d_capacitance_std,
            'D2D_area_ratio_std': self.d2d_area_ratio_std,
            'C2C_enabled': self.enable_c2c,
            'C2C_write_std': self.c2c_write_std,
            'C2C_read_std': self.c2c_read_std,
            'Temperature_enabled': self.enable_temp,
            'Temp_range_K': self.temp_range,
            'Retention_variation_enabled': self.enable_retention_var,
            'Retention_std': self.retention_variation_std if self.enable_retention_var else 0.0
        }


class VariabilityAwareDevice:
    """
    AFeFET device wrapper with variability injection

    Wraps MFMISDevice with realistic fabrication variations
    """

    def __init__(
        self,
        base_device,
        variability_model: VariabilityModel,
        device_id: str = "layer_default"
    ):
        """
        Args:
            base_device: MFMISDevice instance
            variability_model: VariabilityModel instance
            device_id: Unique identifier for D2D variation caching
        """
        self.base_device = base_device
        self.variability = variability_model
        self.device_id = device_id
        self._d2d_vars = None  # Lazy initialization

    def _get_d2d_variations(self, shape, device):
        """Get or create D2D variations for this device"""
        if self._d2d_vars is None:
            self._d2d_vars = self.variability.generate_d2d_variations(
                shape, device, key=self.device_id
            )
        return self._d2d_vars

    def switching_probability(
        self,
        V_applied: torch.Tensor,
        pulse_width: float,
        use_nls: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Switching probability with variability

        Returns: (P_switch, stm_mask, V_FE)
        """
        # Get D2D variations
        d2d = self._get_d2d_variations(V_applied.shape, V_applied.device)

        # Apply D2D variation to V_coercive
        original_V_c = self.base_device.V_coercive
        self.base_device.V_coercive = original_V_c * d2d['V_coercive_factor'].mean().item()

        # Apply D2D variation to area_ratio
        original_area_ratio = self.base_device.area_ratio
        self.base_device.area_ratio = original_area_ratio * d2d['area_ratio_factor'].mean().item()

        # Sample temperature if enabled
        if self.variability.enable_temp:
            temp = self.variability.sample_temperature(V_applied.device)
        else:
            temp = self.base_device.temperature

        # Call base device with potentially varied temperature
        P_switch, stm_mask, V_FE = self.base_device.switching_probability_with_variability(
            V_applied, pulse_width, use_nls=use_nls, temperature=temp
        )

        # Restore original parameters
        self.base_device.V_coercive = original_V_c
        self.base_device.area_ratio = original_area_ratio

        return P_switch, stm_mask, V_FE

    def write_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Apply write noise with C2C variation"""
        return self.variability.apply_c2c_write_noise(value)

    def read_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Apply read noise with C2C variation"""
        return self.variability.apply_c2c_read_noise(value)

    def retention_decay(
        self,
        state: torch.Tensor,
        time_elapsed: float,
        mode: str,
        detrapping_factor: float = 0.0
    ) -> torch.Tensor:
        """Retention decay with variability"""
        # Get D2D retention variation
        d2d = self._get_d2d_variations(state.shape, state.device)

        # Apply retention variation to time constants
        original_stm = self.base_device.stm_retention
        original_ltm = self.base_device.ltm_retention

        retention_factor = d2d['retention_factor'].mean().item()
        self.base_device.stm_retention = original_stm * retention_factor
        self.base_device.ltm_retention = original_ltm * retention_factor

        # Call base device
        result = self.base_device.retention_decay(
            state, time_elapsed, mode, detrapping_factor
        )

        # Restore original
        self.base_device.stm_retention = original_stm
        self.base_device.ltm_retention = original_ltm

        return result

    def calculate_energy(self, *args, **kwargs):
        """Forward energy calculation to base device"""
        return self.base_device.calculate_energy(*args, **kwargs)


def create_variability_scenarios() -> Dict[str, VariabilityModel]:
    """
    Create standard variability scenarios for experiments

    Returns:
        dict: {scenario_name: VariabilityModel}
    """
    scenarios = {
        'perfect': VariabilityModel(
            enable_d2d=False,
            enable_c2c=False,
            enable_temp=False,
            enable_retention_var=False
        ),
        'c2c_only': VariabilityModel(
            enable_d2d=False,
            enable_c2c=True,
            enable_temp=False,
            enable_retention_var=False
        ),
        'd2d_only': VariabilityModel(
            enable_d2d=True,
            enable_c2c=False,
            enable_temp=False,
            enable_retention_var=False,
            d2d_V_coercive_std=0.10,
            d2d_capacitance_std=0.05,
            d2d_area_ratio_std=0.08
        ),
        'd2d_moderate': VariabilityModel(
            enable_d2d=True,
            enable_c2c=True,
            enable_temp=False,
            enable_retention_var=False,
            d2d_V_coercive_std=0.10,  # 10%
            d2d_capacitance_std=0.05,  # 5%
            d2d_area_ratio_std=0.08    # 8%
        ),
        'd2d_high': VariabilityModel(
            enable_d2d=True,
            enable_c2c=True,
            enable_temp=False,
            enable_retention_var=False,
            d2d_V_coercive_std=0.15,  # 15%
            d2d_capacitance_std=0.10,  # 10%
            d2d_area_ratio_std=0.12    # 12%
        ),
        'realistic': VariabilityModel(
            enable_d2d=True,
            enable_c2c=True,
            enable_temp=True,
            enable_retention_var=True,
            d2d_V_coercive_std=0.10,
            d2d_capacitance_std=0.05,
            d2d_area_ratio_std=0.08,
            temp_range=(280, 320),  # ±20°C
            retention_variation_std=0.15
        )
    }

    return scenarios


if __name__ == "__main__":
    """Demonstration of variability models"""

    print("=" * 60)
    print("AFeFET Variability Models Demonstration")
    print("=" * 60)

    # Create scenarios
    scenarios = create_variability_scenarios()

    print("\nAvailable Scenarios:")
    for name, model in scenarios.items():
        print(f"\n{name}:")
        summary = model.get_summary()
        for key, val in summary.items():
            print(f"  {key}: {val}")

    # Demo D2D variation
    print("\n" + "=" * 60)
    print("Device-to-Device Variation Example")
    print("=" * 60)

    model = scenarios['d2d_moderate']
    shape = (100, 100)  # 100x100 weight matrix
    device = torch.device('cpu')

    d2d_vars = model.generate_d2d_variations(shape, device, key="layer1")

    print(f"\nWeight matrix shape: {shape}")
    print(f"Total devices: {shape[0] * shape[1]}")

    for name, tensor in d2d_vars.items():
        print(f"\n{name}:")
        print(f"  Mean: {tensor.mean():.4f} (target: 1.0)")
        print(f"  Std: {tensor.std():.4f}")
        print(f"  Min: {tensor.min():.4f}")
        print(f"  Max: {tensor.max():.4f}")

    # Demo C2C variation
    print("\n" + "=" * 60)
    print("Cycle-to-Cycle Variation Example")
    print("=" * 60)

    original_weight = torch.ones(10)
    print(f"\nOriginal weight: {original_weight[0]:.4f}")

    writes = []
    for i in range(5):
        noisy = model.apply_c2c_write_noise(original_weight)
        writes.append(noisy[0].item())
        print(f"Write {i+1}: {noisy[0]:.4f}")

    print(f"\nWrite variation std: {torch.tensor(writes).std():.4f}")
    print(f"Expected (2.1%): ~0.021")

    print("\n" + "=" * 60)
