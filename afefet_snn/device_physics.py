"""
AFeFET Device Physics Modeling
Implements MFMIS (Metal-Ferroelectric-Metal-Insulator-Semiconductor) device dynamics
"""

import torch
import torch.nn as nn


class MFMISDevice:
    """
    MFMIS Device Model with reconfigurability

    Key parameters:
    - area_ratio: A_MIS/A_FE controls voltage division (CORRECTED)
    - pulse_width: Controls STM (volatile) vs LTM (non-volatile) mode
    - retention_time: Weight retention characteristics
    """

    def __init__(self, area_ratio=1.0, base_voltage=4.5,
                 stm_retention=5e-2, ltm_retention=1e6,
                 V_coercive=3.0, k=3.0, width_threshold=100e-6,
                 use_realistic_energy=True, temperature=300.0, 
                 per_element_variation=False, seed=None):
        """
        Args:
            area_ratio: A_MIS/A_FE ratio (CORRECT definition from paper)
                - Low ratio (<1): Less V_FE, harder to reach LTM
                - High ratio (>1): More V_FE, easier LTM switching
                Paper Fig. 3c: larger A_MIS/A_FE → larger V_FE → lower critical V for LTM
            base_voltage: Operating voltage (V)
            stm_retention: STM retention time (seconds) - default 50ms (paper Fig. 2h)
            ltm_retention: LTM retention time (seconds) - default 1e6s (~11.6 days) for true non-volatility
            V_coercive: Coercive voltage for AFE switching (V)
            k: Sigmoid steepness for switching probability
            width_threshold: Pulse width threshold for STM/LTM (seconds)
            use_realistic_energy: If True, scale to fF-range capacitance (~0.15 pJ/spike)
            temperature: Operating temperature in Kelvin (300K default)
            per_element_variation: If True, enable per-element device variation for large arrays
            seed: Random seed for reproducible device variation (optional)
        """
        # Set random seed for reproducible simulations if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        self.area_ratio = area_ratio  # A_MIS/A_FE
        self.base_voltage = base_voltage
        self.stm_retention = stm_retention
        self.ltm_retention = ltm_retention
        self.temperature = temperature  # Operating temperature
        self.use_realistic_energy = use_realistic_energy  # Store for energy calculations

        # Configurable switching parameters
        self.V_coercive = V_coercive
        self.k = k
        self.width_threshold = width_threshold

        # Device parameters (from typical MFMIS)
        # Paper: ~0.15 pJ for LTM (large area ratio) → requires fF-range capacitance
        # Paper Suppl. Table 3: 15 nm HZO, 6.5 nm Al₂O₃, effective C ~ 10-30 fF
        if use_realistic_energy:
            # Realistic capacitance after voltage division effects
            # Target: ~0.15 pJ at 5V → C ≈ 12 fF from E = 0.5×C×V²
            self.C_FE = 20e-15  # 20 fF (paper-accurate for 15nm HZO)
            self.C_MIS = 20e-15  # 20 fF
        else:
            # Simulation values (qualitative, ~30× higher energy)
            self.C_FE = 1e-12  # 1 pF
            self.C_MIS = 1e-12  # 1 pF
        
        # Stack parameters for 28-nm HKMG-compatible process
        self.t_HZO = 15e-9      # 15 nm HZO ferroelectric layer
        self.t_Al2O3 = 6.5e-9   # 6.5 nm Al2O3 dielectric layer
        self.epsilon_HZO = 25   # Relative permittivity of HZO
        self.epsilon_Al2O3 = 9  # Relative permittivity of Al2O3
        
        # Device-specific gamma for NLS (grain-to-grain variation)
        # Each physical device keeps its own γ for lifetime
        gamma_base = 0.5  # Base half-width (V)
        gamma_std = 0.05  # Standard deviation for variation
        self.gamma = torch.clamp(torch.normal(gamma_base, gamma_std, size=(1,)), 0.1, 1.0).item()
        
        # Optional: per-element device variation (for large arrays)
        self.enable_per_element_variation = per_element_variation  # Exposed parameter
        
        # Per-element variation parameters (used when enable_per_element_variation=True)
        self.vc_sigma = 0.05  # V, coercive voltage variation
        self.gamma_sigma = 0.05  # gamma variation
        
        # Energy calculation constants (parameterizable for sensitivity analysis)
        self.I_leak = 1e-9  # 1 nA leakage current
        self.C_eff_realistic = 20e-15  # 20 fF realistic capacitance

    def _per_element_offsets(self, like):
        """
        Generate per-element device variation offsets
        
        Args:
            like: Reference tensor for shape and device
            
        Returns:
            vc_off: Coercive voltage offsets (0.0 if variation disabled)
            gamma_map: Per-element gamma values (scalar gamma if variation disabled)
        """
        if not self.enable_per_element_variation:
            return 0.0, self.gamma  # no offsets, shared gamma
            
        vc_off = torch.randn_like(like) * self.vc_sigma
        gamma_map = torch.clamp(
            torch.normal(self.gamma, self.gamma_sigma, size=like.shape, device=like.device, dtype=like.dtype),
            0.1, 1.0
        )
        return vc_off, gamma_map

    def voltage_division(self, V_applied):
        """
        Calculate voltage across FE and MIS based on area ratio

        CORRECTED from paper (Fig. 3c):
        - area_ratio = A_MIS / A_FE
        - C_MIS_eff = C_MIS * area_ratio (scale MIS capacitance, not FE!)
        - Higher area_ratio → more V_FE → easier LTM switching

        Capacitive divider:
        V_FE = V_applied * (C_MIS_eff / (C_FE + C_MIS_eff))
        V_MIS = V_applied * (C_FE / (C_FE + C_MIS_eff))
        """
        C_MIS_eff = self.C_MIS * self.area_ratio  # CORRECTED: scale C_MIS, not C_FE
        C_total = self.C_FE + C_MIS_eff

        V_FE = V_applied * (C_MIS_eff / C_total)  # ↑area_ratio → ↑V_FE (matches paper)
        V_MIS = V_applied * (self.C_FE / C_total)

        return V_FE, V_MIS

    def switching_probability(self, V_applied, pulse_width, use_nls=True, temperature=None):
        """
        AFE switching probability based on voltage and pulse width

        ENHANCED with NLS temporal accumulation and temperature dependence:
        - STM: Partial AFE switching (V_FE < Vc) + back-switching (volatile)
        - LTM: Complete AFE switching (V_FE >= Vc) + de-trapping (non-volatile)
        - Both modes driven by V_FE (not V_MIS!)
        - Paper Fig. 3: 100 µs - 60 ms pulse width range for STM/LTM
        - Paper Fig. 1c: Nucleation-Limited Switching with Lorentzian distribution

        Args:
            use_nls: If True, use CDF-based temporal accumulation (paper-accurate)
                    If False, use sigmoid (faster for training)
            temperature: Temperature in Kelvin (uses self.temperature if None)
            
        Note:
            For tensor V_applied with enable_per_element_variation=True:
            - P_switch and mode are per-element with device-to-device variation
            - Otherwise mode reflects the broadcasted condition across all elements
        """
        V_FE, V_MIS = self.voltage_division(V_applied)

        # Temperature-dependent coercive voltage
        # Use self.temperature by default, allow override
        T = self.temperature if temperature is None else temperature
        # V_c(T) = V_c(300K) * (1 - α(T-300)) with α ≈ 1.5e-3 K^-1 for HZO
        alpha_temp = 1.5e-3  # K^-1 for HZO
        V_coercive_T = self.V_coercive * (1 - alpha_temp * (T - 300.0))

        # Per-element device variation (if enabled)
        vc_off, gamma_map = self._per_element_offsets(V_FE)
        V_coercive_T_eff = V_coercive_T + vc_off

        # Switching probability is function of V_FE (AFE field)
        if use_nls:
            # ENHANCED: NLS with CDF-based temporal accumulation
            # Paper Fig. 1b: τ(V) follows Lorentzian distribution
            # P_switch = 1 - exp(-t_pulse/τ(V_FE))
            
            # Use device-specific or per-element gamma
            gamma_used = gamma_map if self.enable_per_element_variation else self.gamma
            
            # Characteristic time from Lorentzian distribution
            # τ(V) decreases as |V - Vc| increases (faster switching at higher field)
            # Baseline time constant (paper: ~170 ns for AFE switching)
            tau_0 = 170e-9  # seconds
            V_delta = V_FE - V_coercive_T_eff
            # CORRECTED: tau decreases as |V - Vc| increases (Lorentzian peak form)
            tau_V = tau_0 * (gamma_used**2 / (V_delta**2 + gamma_used**2))
            # Boundary protection: prevent numerical overflow and maintain reasonable gradients
            tau_V = torch.clamp(tau_V, min=1e-12)  # ~1 ps floor to avoid P→1 instantly
            
            # CDF-based switching probability: P = 1 - exp(-t/τ)
            # This captures temporal accumulation of switched domains
            P_switch = 1 - torch.exp(-pulse_width / tau_V)
            P_switch = torch.clamp(P_switch, 0.0, 1.0)
        else:
            # Sigmoid approximation (use configurable parameters)
            P_switch = torch.sigmoid((V_FE - V_coercive_T_eff) * self.k)

        # Mode selection based on pulse width AND V_FE
        # Paper: STM = partial AFE + back-switch; LTM = complete AFE + de-trapping
        # Enhanced: area-ratio dependent threshold (larger devices need longer pulses)
        width_thr = self.width_threshold / (1 + 0.5 * self.area_ratio)
        
        # Ensure pulse_width is a tensor broadcastable to V_FE
        if not torch.is_tensor(pulse_width):
            pulse_width_t = torch.as_tensor(pulse_width, device=V_FE.device, dtype=V_FE.dtype)
        else:
            pulse_width_t = pulse_width.to(device=V_FE.device, dtype=V_FE.dtype)
        
        # Element-wise STM mask (True=STM, False=LTM)
        stm_mask = (pulse_width_t < width_thr) & (V_FE < V_coercive_T_eff)

        return P_switch, stm_mask, V_FE  # return V_FE too (handy for de-trapping)

    def charge_detrapping_probability(self, V_FE, pulse_width, mode):
        """
        Electron de-trapping extends retention (LTM only)

        Paper Fig. 3: De-trapping coupled with complete AFE switching
        Suppl. Fig. 15: Area-ratio dependent (larger A_MIS/A_FE → easier de-trapping)

        - Requires both high V_FE and long pulse width
        - Only significant in LTM mode
        - τ_detrap ≈ 10⁴ s baseline (Suppl. Fig. 11)
        """
        if mode != 'LTM':
            return torch.as_tensor(0.0, device=V_FE.device if hasattr(V_FE, 'device') else 'cpu',
                                  dtype=V_FE.dtype if hasattr(V_FE, 'dtype') else torch.float32)

        # De-trapping threshold scales with area_ratio
        # Higher area_ratio → more V_FE → easier de-trapping (Suppl. Fig. 15)
        V_threshold = self.V_coercive + 0.5 / (1 + 0.5 * self.area_ratio)

        # Time factor: Suppl. Fig. 15 shows LTM achievable from 1µs with high area_ratio
        # Scale time threshold by area_ratio
        time_threshold = 1e-4 / max(1.0, self.area_ratio)  # 100µs → 50µs for area_ratio=2

        if hasattr(pulse_width, 'device'):
            time_factor = torch.sigmoid((pulse_width - time_threshold) / 1e-3)
        else:
            time_factor = torch.sigmoid(torch.as_tensor((pulse_width - time_threshold) / 1e-3,
                                                       device=V_FE.device if hasattr(V_FE, 'device') else 'cpu',
                                                       dtype=V_FE.dtype if hasattr(V_FE, 'dtype') else torch.float32))

        # Voltage factor: requires V_FE > threshold
        voltage_factor = torch.sigmoid((V_FE - V_threshold) * 2)

        # Area ratio boost: larger areas → stronger de-trapping
        area_boost = torch.sigmoid(torch.as_tensor((self.area_ratio - 1.0) * 2.0,
                                                  device=V_FE.device if hasattr(V_FE, 'device') else 'cpu',
                                                  dtype=V_FE.dtype if hasattr(V_FE, 'dtype') else torch.float32))

        return time_factor * voltage_factor * (0.5 + 0.5 * area_boost)

    def retention_decay(self, state, time_elapsed, mode, detrapping_factor=0.0):
        """
        Model weight retention over time

        ENHANCED: De-trapping extends LTM retention with temperature dependence
        Paper Suppl. Fig. 13: Log-linear drift ∝ log₁₀(t) after 10⁴ s

        - STM: Fast exponential decay (50ms timescale, paper Fig. 2h)
        - LTM: Exponential → logarithmic transition (≥ 10⁴ s, paper experimental data)
        - De-trapping baseline: τ_detrap ≈ 10⁴ s (Suppl. Fig. 11)
        - Temperature dependence: Higher T → faster decay

        Args:
            state: Current state
            time_elapsed: Time since write (seconds)
            mode: 'STM' or 'LTM'
            detrapping_factor: 0-1, strength of de-trapping (LTM only)
        """
        dev = state.device if hasattr(state, 'device') else 'cpu'
        dtype = state.dtype if hasattr(state, 'dtype') else torch.float32

        # Temperature-dependent acceleration factor
        # Arrhenius-like: faster decay at higher temperatures
        # Clamp to prevent extreme hot/cold simulations from annihilating state instantly
        temp_factor = torch.clamp(
            torch.exp(torch.tensor((self.temperature - 300.0) / 100.0, device=dev, dtype=dtype)),
            0.1, 10.0
        )

        if mode == 'STM':
            # STM: pure exponential decay (50ms timescale) with temperature dependence
            tau = self.stm_retention / temp_factor
            decay_factor = torch.as_tensor(-time_elapsed / tau, device=dev, dtype=dtype).exp()
            # Add small decay floor to prevent total weight collapse (numerical guard)
            decay_factor = torch.clamp(decay_factor, 0.1, 1.0)  # floor at 10%
            return state * decay_factor
        else:
            # LTM: Use experimental baseline ≥ 10⁴ s (Suppl. Fig. 11)
            # De-trapping extends this baseline, temperature affects decay rate
            tau_base = 1e4 / temp_factor  # Temperature-dependent baseline
            tau = tau_base * (1 + detrapping_factor * 10)  # Up to 10⁵ s with de-trapping

            # Transition to log-linear decay after 10² s (Suppl. Fig. 13)
            if time_elapsed < 100:
                # Short time: exponential with temperature dependence
                decay_factor = torch.as_tensor(-time_elapsed / tau, device=dev, dtype=dtype).exp()
            else:
                # Long time: log-linear drift (more realistic for AFE)
                # Q(t) = Q0 * (1 - c·log₁₀(1 + t/τ))
                # Drift coefficient decreases with detrapping_factor (stronger de-trapping → more stable)
                c_base = 0.05  # Base drift coefficient (5% per decade)
                c_effective = c_base * temp_factor * (1 - 0.5 * detrapping_factor)  # Stronger de-trapping reduces drift
                c_effective = torch.clamp(c_effective, 0.01, 0.2)  # Reasonable bounds
                log_term = torch.log10(torch.as_tensor(1 + time_elapsed / tau, device=dev, dtype=dtype))
                decay_factor = torch.clamp(1 - c_effective * log_term, 0.1, 1.0)  # Clamp to avoid negative

            return state * decay_factor

    def write_noise(self, value, noise_std=0.021):
        """
        Add cycle-to-cycle variation (2.1% from paper)
        """
        noise = torch.randn_like(value) * noise_std
        return value * (1 + noise)

    def read_noise(self, value, noise_std=0.003):
        """
        Read variation (typically lower than write)

        Paper Suppl. Fig. 12: ~0.3% read noise
        (Corrected from 1% to match experimental data)
        """
        noise = torch.randn_like(value) * noise_std
        return value * (1 + noise)

    def calculate_energy(self, V_applied, switching_mask, pulse_width, return_per_weight=False):
        """
        ENHANCED energy calculation per switching event
        
        Energy components:
        1. Switching energy: E_switch = 0.5 * C * V²
        2. Leakage energy: E_leak = V * I_leak * t_pulse
        
        Paper: ~0.15 pJ/spike for realistic AFeFET devices
        
        Args:
            V_applied: Applied voltage tensor
            switching_mask: Boolean mask of actual switching events
            pulse_width: Pulse duration in seconds
            return_per_weight: If True, return per-weight energy map; if False, return total scalar
            
        Returns:
            energy: Total energy (scalar) or per-weight energy map (tensor)
            num_switches: Number of actual switching events
        """
        # Shape agreement check to catch broadcast mismatches early
        assert V_applied.shape == switching_mask.shape or V_applied.numel() == 1, \
            "V_applied must be scalar or same shape as switching_mask"
        
        # Count actual switching events (not all weights)
        num_switches = torch.sum(switching_mask.float())
        
        if num_switches == 0:
            if return_per_weight:
                return torch.zeros_like(V_applied), 0
            else:
                return torch.as_tensor(0.0, device=V_applied.device, dtype=V_applied.dtype), 0
        
        # Use V_FE for energy calculation (consistent with MFMIS model)
        V_FE, _ = self.voltage_division(V_applied)
        
        # Switching energy: E = 0.5 * C_eff * V_FE²
        # Use realistic capacitance (20 fF for 28-nm HKMG stack) with area scaling
        if self.use_realistic_energy:
            C_eff = self.C_eff_realistic * self.area_ratio  # Use stored realistic capacitance
        else:
            C_eff = self.C_MIS * self.area_ratio
            
        # Per-event switching energy
        energy_per_event = 0.5 * C_eff * (V_FE ** 2)
        
        # Leakage energy during pulse
        # Standardize to 1 nA leakage current (consistent with energy_per_write)
        E_leakage = torch.abs(V_FE) * self.I_leak * pulse_width
        
        # Total per-event energy
        total_per_event = energy_per_event + E_leakage
        
        if return_per_weight:
            # Return per-weight energy map
            return total_per_event * switching_mask.float(), num_switches.item()
        else:
            # Return total scalar energy
            total_energy = torch.sum(total_per_event * switching_mask.float())
            energy_per_event = total_energy / num_switches if num_switches > 0 else total_energy
            return energy_per_event, num_switches.item()

    def energy_per_write(self, V_applied, pulse_width, switching_occurred):
        """
        Calculate mean energy per write operation across switching events

        CORRECTED with proper C_MIS_eff and 0.5 factor:
        E_write = 0.5 * C_MIS_eff * V_FE² * switching + E_leakage

        Paper reports ~0.1 pJ/spike (400 ns) for neuron, ~0.15 pJ for LTM (large area ratio)

        NOTE: 
        - Returns scalar mean energy when V_applied/switching_occurred are tensors
        - For per-weight energy maps, use calculate_energy() instead
        - Simulated energy values are qualitative. Real devices have:
          * Smaller effective capacitances (~fF range after voltage division)
          * Lower V_FE after series capacitor effects
          * This gives ~30x lower energy than naive calculation
          
        Args:
            V_applied: Applied voltage tensor
            pulse_width: Pulse duration in seconds  
            switching_occurred: Boolean mask of switching events
            
        Returns:
            float: Mean energy per write operation (scalar)
        """
        # Shape agreement check to catch broadcast mismatches early
        assert V_applied.shape == switching_occurred.shape or V_applied.numel() == 1, \
            "V_applied must be scalar or same shape as switching_occurred"
            
        V_FE, V_MIS = self.voltage_division(V_applied)

        # Switching energy with corrected capacitance
        # E = 0.5 * C * V^2 for capacitor energy
        C_MIS_eff = self.C_MIS * self.area_ratio  # CORRECTED: use C_MIS_eff
        E_switching = 0.5 * C_MIS_eff * (V_FE ** 2) * switching_occurred

        # Leakage energy (proportional to |V_FE| magnitude)
        E_leakage = torch.abs(V_FE) * self.I_leak * pulse_width

        E_total = E_switching + E_leakage

        # Safe tensor-to-scalar conversion: use mean for multi-element tensors
        if E_total.numel() > 1:
            return E_total.mean().item()
        else:
            return E_total.item() if torch.is_tensor(E_total) else E_total


class ReconfigurableWeight(nn.Module):
    """
    Reconfigurable weight with STM/LTM modes
    """

    def __init__(self, shape, voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5],
                 base_voltage=4.5, area_ratio=1.0):
        super().__init__()

        self.shape = shape
        self.voltage_levels = torch.tensor(voltage_levels)
        self.base_voltage = base_voltage

        # Device model
        self.device = MFMISDevice(area_ratio=area_ratio, base_voltage=base_voltage)

        # Weight states
        self.weight = nn.Parameter(torch.zeros(shape))

        # Mode tracking
        self.register_buffer('mode', torch.zeros(shape, dtype=torch.long))  # 0: STM, 1: LTM
        self.register_buffer('last_update_time', torch.zeros(shape))
        self.register_buffer('write_count', torch.zeros(shape, dtype=torch.long))
        self.register_buffer('detrapping_strength', torch.zeros(shape))  # NEW: for LTM enhancement

    def write(self, new_value, pulse_width=1e-3, voltage=None):
        """
        Write operation with mode selection based on pulse width

        ENHANCED: Integrates charge de-trapping mechanism with vectorized mode handling

        Args:
            new_value: Target weight value
            pulse_width: Pulse width (s)
                - < 100 μs: STM mode
                - > 100 μs: LTM mode
            voltage: Applied voltage (defaults to base_voltage)
        """
        if voltage is None:
            voltage = self.base_voltage

        # Device-safe voltage tensor
        V_tensor = torch.as_tensor(voltage, device=self.weight.device, dtype=self.weight.dtype)

        # Determine switching probability and per-element mode
        P_switch, stm_mask, V_FE = self.device.switching_probability(V_tensor, pulse_width)

        # Elementwise stochastic switching
        p_switch = P_switch.clamp(0, 1).to(self.weight.device, self.weight.dtype)
        switching_mask = (torch.rand_like(self.weight) < p_switch)

        # Apply write noise and update
        noisy_value = self.device.write_noise(new_value)
        self.weight.data = torch.where(switching_mask, noisy_value, self.weight.data)

        # Update mode tracking: 0 for STM, 1 for LTM, element-wise
        mode_tensor = torch.where(stm_mask, torch.zeros_like(self.mode), torch.ones_like(self.mode))
        self.mode = torch.where(switching_mask, mode_tensor, self.mode)

        # Per-element de-trapping (only where LTM AND switched)
        ltm_switched = switching_mask & (~stm_mask)
        if ltm_switched.any():
            P_detrapping = self.device.charge_detrapping_probability(V_FE, pulse_width, mode='LTM')
            # Broadcast to weight shape
            if P_detrapping.ndim == 0:
                P_detrapping = torch.full_like(self.detrapping_strength, P_detrapping.item())
            self.detrapping_strength = torch.where(
                ltm_switched, P_detrapping.to(self.detrapping_strength.dtype), self.detrapping_strength
            )

        # Update write count (for endurance modeling)
        self.write_count += switching_mask.long()

        return mode

    def read(self, apply_retention=True, time_elapsed=1.0):
        """
        Read operation with retention effects

        ENHANCED: Uses de-trapping strength to extend LTM retention

        Args:
            apply_retention: Apply retention decay
            time_elapsed: Time since last write (seconds)
        """
        value = self.weight.data

        if apply_retention:
            # Apply retention decay based on mode
            stm_mask = (self.mode == 0)
            ltm_mask = (self.mode == 1)

            # STM: fast decay, no de-trapping
            value = torch.where(
                stm_mask,
                self.device.retention_decay(value, time_elapsed, 'STM', detrapping_factor=0.0),
                value
            )

            # LTM: slow decay, enhanced by de-trapping
            # Use average de-trapping strength for the whole tensor (simpler, vectorized)
            if ltm_mask.any():
                avg_detrap_strength = self.detrapping_strength[ltm_mask].mean().item()
                value = torch.where(
                    ltm_mask,
                    self.device.retention_decay(value, time_elapsed, 'LTM',
                                               detrapping_factor=avg_detrap_strength),
                    value
                )

        # Apply read noise
        return self.device.read_noise(value)

    def endurance_degradation(self):
        """
        Model endurance degradation (10^8 cycles typical)
        """
        # After 10^8 cycles, ~10% degradation
        max_cycles = 1e8
        degradation = (self.write_count.float() / max_cycles) * 0.1
        degradation = torch.clamp(degradation, 0, 0.5)  # Max 50% degradation

        return self.weight.data * (1 - degradation)


class TemporalDynamics:
    """
    Temporal neuronal dynamics for AFeFET neurons
    """

    @staticmethod
    def integrate_and_fire(inputs, v_membrane, v_threshold=1.0, v_reset=0.0,
                          tau_mem=20e-3, leak_beta=None):
        """
        Leaky integrate-and-fire dynamics with AFE back-switching reset

        Paper Suppl. Eq. S1: LIF neuron with self-reset via AFE back-switching (Fig. 5b)
        
        Args:
            inputs: Input current
            v_membrane: Current membrane potential
            v_threshold: Spike threshold
            v_reset: Reset potential
            tau_mem: Membrane time constant (s) - default 20 ms
            leak_beta: Override for leakage factor (if None, computed from tau_mem)
        """
        dt = 1e-3  # 1 ms timestep

        # Compute leakage factor from tau_mem (exponential decay formulation)
        if leak_beta is None:
            alpha = torch.exp(torch.as_tensor(-dt / tau_mem, 
                                            device=v_membrane.device, 
                                            dtype=v_membrane.dtype))
            # Canonical discrete LIF: V(t+1) = α·V(t) + (1-α)·inputs
            v_membrane = alpha * v_membrane + (1 - alpha) * inputs
        else:
            # Legacy β factor mode (paper Eq. S1): V(t+1) = β·V(t) + I(t)·dt
            v_membrane = leak_beta * v_membrane + inputs * dt

        # Spike detection
        spikes = (v_membrane >= v_threshold).float()

        # Reset via AFE back-switching (intrinsic, no reset circuitry needed)
        # Device-safe tensor creation
        dev = v_membrane.device if hasattr(v_membrane, 'device') else 'cpu'
        v_membrane = torch.where(spikes.bool(),
                                v_membrane.new_tensor(v_reset),
                                v_membrane)

        return spikes, v_membrane

    @staticmethod
    def paired_pulse_facilitation(pulse1_weight, pulse2_weight,
                                  inter_pulse_interval,
                                  tau_fast=1.58e-6, tau_slow=10e-6,
                                  mix=0.5, gain=0.5):
        """
        PPF: Short-term facilitation from repeated stimulation

        CORRECTED from paper (bi-exponential, microsecond scale):
        - τ₁ ≈ 1.58 µs (fast component)
        - τ₂ ≈ 10 µs (slow component)
        - Mix of both components

        Note: If your simulator uses dt = 1 ms for LIF, this µs-scale PPF
        is qualitative (won't be fully resolved at 1 ms timesteps)
        """
        # Device-safe tensor creation
        dev = pulse2_weight.device if hasattr(pulse2_weight, 'device') else 'cpu'
        dtype = pulse2_weight.dtype if hasattr(pulse2_weight, 'dtype') else torch.float32

        # Bi-exponential facilitation (paper Fig. 4)
        f_fast = torch.as_tensor(-inter_pulse_interval / tau_fast, device=dev, dtype=dtype).exp()
        f_slow = torch.as_tensor(-inter_pulse_interval / tau_slow, device=dev, dtype=dtype).exp()
        facilitation = mix * f_fast + (1 - mix) * f_slow

        # Second pulse is facilitated
        effective_weight2 = pulse2_weight * (1 + gain * facilitation)

        return effective_weight2

    @staticmethod
    def stdp_update(pre_spike_time, post_spike_time,
                   tau_plus=2.21e-3, tau_minus=2.04e-3,
                   A_plus=0.01, A_minus=0.01,
                   device='cpu', dtype=torch.float32):
        """
        Spike-timing-dependent plasticity

        CORRECTED from paper (millisecond scale):
        - τ_plus ≈ 2.21 ms (LTP time constant)
        - τ_minus ≈ 2.04 ms (LTD time constant)

        Args:
            pre_spike_time: Pre-synaptic spike time
            post_spike_time: Post-synaptic spike time
            tau_plus/minus: STDP time constants
            A_plus/minus: Learning rates
            device/dtype: For device-safe tensor creation
        """
        delta_t = post_spike_time - pre_spike_time

        if delta_t > 0:
            # Post after pre → potentiation (LTP)
            weight_change = A_plus * torch.as_tensor(-delta_t / tau_plus, device=device, dtype=dtype).exp()
        else:
            # Pre after post → depression (LTD)
            weight_change = -A_minus * torch.as_tensor(delta_t / tau_minus, device=device, dtype=dtype).exp()

        return weight_change


def demonstrate_reconfigurability():
    """
    Demonstration of STM/LTM reconfigurability
    """
    print("=" * 60)
    print("AFeFET Reconfigurability Demonstration")
    print("=" * 60)

    # Create device with moderate area ratio
    weight = ReconfigurableWeight(shape=(10, 10), area_ratio=1.0)

    print("\n1. STM Mode (Short pulse, volatile)")
    initial_value = torch.randn(10, 10) * 0.5
    mode = weight.write(initial_value, pulse_width=10e-6)  # 10 μs
    print(f"   Mode: {mode}")
    print(f"   Immediate read: {weight.read(apply_retention=False).mean():.4f}")
    print(f"   After 1s: {weight.read(time_elapsed=1.0).mean():.4f}")
    print(f"   After 10s: {weight.read(time_elapsed=10.0).mean():.4f} (decayed!)")

    print("\n2. LTM Mode (Long pulse, non-volatile)")
    mode = weight.write(initial_value, pulse_width=1e-3)  # 1 ms
    print(f"   Mode: {mode}")
    print(f"   Immediate read: {weight.read(apply_retention=False).mean():.4f}")
    print(f"   After 100s: {weight.read(time_elapsed=100.0).mean():.4f}")
    print(f"   After 1000s: {weight.read(time_elapsed=1000.0).mean():.4f} (retained!)")

    print("\n3. Neuronal Dynamics (Integrate-and-Fire)")
    v_mem = torch.zeros(10)
    inputs = torch.randn(10) * 2
    spikes, v_mem = TemporalDynamics.integrate_and_fire(inputs, v_mem)
    print(f"   Input: {inputs[:5]}")
    print(f"   Membrane: {v_mem[:5]}")
    print(f"   Spikes: {spikes[:5]}")

    print("\n4. Paired-Pulse Facilitation (CORRECTED: µs scale)")
    w1 = torch.tensor(1.0)
    # Using corrected paper values: τ₁=1.58µs, τ₂=10µs
    w2_short = TemporalDynamics.paired_pulse_facilitation(w1, w1, 10e-6)  # 10 µs
    w2_long = TemporalDynamics.paired_pulse_facilitation(w1, w1, 100e-6)  # 100 µs
    print(f"   Baseline weight: {w1:.4f}")
    print(f"   After 10µs interval: {w2_short:.4f} (high facilitation)")
    print(f"   After 100µs interval: {w2_long:.4f} (less facilitation)")
    print(f"   Paper: τ₁=1.58µs, τ₂=10µs (bi-exponential)")

    print("\n5. STDP (CORRECTED: ms scale)")
    # Using corrected paper values: τ_plus=2.21ms, τ_minus=2.04ms
    w_change_ltp = TemporalDynamics.stdp_update(0.0, 5e-3)  # Post 5ms after pre
    w_change_ltd = TemporalDynamics.stdp_update(5e-3, 0.0)  # Pre 5ms after post
    print(f"   LTP (Δt=+5ms): {w_change_ltp:.6f}")
    print(f"   LTD (Δt=-5ms): {w_change_ltd:.6f}")
    print(f"   Paper: τ_plus=2.21ms, τ_minus=2.04ms")

    print("=" * 60)


if __name__ == "__main__":
    demonstrate_reconfigurability()
