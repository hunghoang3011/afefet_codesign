"""
AFeFET Device Physics Modeling
Implements MFMIS (Metal-Ferroelectric-Metal-Insulator-Semiconductor) device dynamics
"""

import torch
import torch.nn as nn
import numpy as np


class MFMISDevice:
    """
    MFMIS Device Model with reconfigurability

    Key parameters:
    - area_ratio: A_FE/A_MIS controls voltage division
    - pulse_width: Controls STM (volatile) vs LTM (non-volatile) mode
    - retention_time: Weight retention characteristics
    """

    def __init__(self, area_ratio=1.0, base_voltage=4.5,
                 stm_retention=1e-3, ltm_retention=1e4):
        """
        Args:
            area_ratio: A_FE/A_MIS ratio (0.1 to 10)
                - Low ratio (<1): More STM behavior
                - High ratio (>1): More LTM behavior
            base_voltage: Operating voltage (V)
            stm_retention: STM retention time (seconds)
            ltm_retention: LTM retention time (seconds)
        """
        self.area_ratio = area_ratio
        self.base_voltage = base_voltage
        self.stm_retention = stm_retention
        self.ltm_retention = ltm_retention

        # Device parameters (from typical MFMIS)
        self.C_FE = 1e-12  # Ferroelectric capacitance (F)
        self.C_MIS = 1e-12  # MIS capacitance (F)

    def voltage_division(self, V_applied):
        """
        Calculate voltage across FE and MIS based on area ratio
        V_FE = V_applied * (C_MIS / (C_FE + C_MIS))

        With area scaling:
        C_FE_eff = C_FE * area_ratio
        """
        C_FE_eff = self.C_FE * self.area_ratio
        C_total = C_FE_eff + self.C_MIS

        V_FE = V_applied * (self.C_MIS / C_total)
        V_MIS = V_applied * (C_FE_eff / C_total)

        return V_FE, V_MIS

    def switching_probability(self, V_applied, pulse_width):
        """
        AFE switching probability based on voltage and pulse width

        Short pulses (μs): Charge trapping (STM, volatile)
        Long pulses (ms): AFE polarization switching (LTM, non-volatile)
        """
        V_FE, V_MIS = self.voltage_division(V_applied)

        # Coercive voltage for AFE switching (typical ~3V)
        V_coercive = 3.0

        # Time-dependent switching
        if pulse_width < 100e-6:  # < 100 μs → STM mode
            # Charge trapping dynamics (fast, volatile)
            P_switch = torch.sigmoid((V_MIS - 2.0) * 2)  # Charge trapping threshold
            mode = 'STM'
        else:  # > 100 μs → LTM mode
            # AFE polarization switching (slow, non-volatile)
            P_switch = torch.sigmoid((V_FE - V_coercive) * 3)
            mode = 'LTM'

        return P_switch, mode

    def retention_decay(self, state, time_elapsed, mode):
        """
        Model weight retention over time

        STM: Fast exponential decay (ms timescale)
        LTM: Slow decay (hours/days timescale)
        """
        if mode == 'STM':
            tau = self.stm_retention
        else:
            tau = self.ltm_retention

        # Exponential decay: Q(t) = Q0 * exp(-t/tau)
        decay_factor = torch.exp(torch.tensor(-time_elapsed / tau))
        return state * decay_factor

    def write_noise(self, value, noise_std=0.021):
        """
        Add cycle-to-cycle variation (2.1% from paper)
        """
        noise = torch.randn_like(value) * noise_std
        return value * (1 + noise)

    def read_noise(self, value, noise_std=0.01):
        """
        Read variation (typically lower than write)
        """
        noise = torch.randn_like(value) * noise_std
        return value * (1 + noise)

    def energy_per_write(self, V_applied, pulse_width, switching_occurred):
        """
        Calculate energy per write operation

        E_write = C * V² * switching_fraction + E_leakage
        """
        V_FE, V_MIS = self.voltage_division(V_applied)

        # Switching energy (only if switching occurred)
        E_switching = self.C_FE * self.area_ratio * (V_FE ** 2) * switching_occurred

        # Leakage energy (proportional to pulse width)
        I_leakage = 1e-9  # 1 nA typical
        E_leakage = V_applied * I_leakage * pulse_width

        return E_switching + E_leakage


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

    def write(self, new_value, pulse_width=1e-3, voltage=None):
        """
        Write operation with mode selection based on pulse width

        Args:
            new_value: Target weight value
            pulse_width: Pulse width (s)
                - < 100 μs: STM mode
                - > 100 μs: LTM mode
            voltage: Applied voltage (defaults to base_voltage)
        """
        if voltage is None:
            voltage = self.base_voltage

        # Determine switching probability and mode
        P_switch, mode = self.device.switching_probability(
            torch.tensor(voltage), pulse_width
        )

        # Apply write noise
        noisy_value = self.device.write_noise(new_value)

        # Update weight (probabilistic switching)
        switching_mask = torch.rand_like(self.weight) < P_switch.item()
        self.weight.data = torch.where(switching_mask, noisy_value, self.weight.data)

        # Update mode tracking
        mode_value = 0 if mode == 'STM' else 1
        self.mode = torch.where(switching_mask,
                                torch.full_like(self.mode, mode_value),
                                self.mode)

        # Update write count (for endurance modeling)
        self.write_count += switching_mask.long()

        return mode

    def read(self, apply_retention=True, time_elapsed=1.0):
        """
        Read operation with retention effects

        Args:
            apply_retention: Apply retention decay
            time_elapsed: Time since last write (seconds)
        """
        value = self.weight.data

        if apply_retention:
            # Apply retention decay based on mode
            stm_mask = (self.mode == 0)
            ltm_mask = (self.mode == 1)

            value = torch.where(
                stm_mask,
                self.device.retention_decay(value, time_elapsed, 'STM'),
                value
            )
            value = torch.where(
                ltm_mask,
                self.device.retention_decay(value, time_elapsed, 'LTM'),
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
                          tau_mem=20e-3):
        """
        Leaky integrate-and-fire dynamics

        Args:
            inputs: Input current
            v_membrane: Current membrane potential
            v_threshold: Spike threshold
            v_reset: Reset potential
            tau_mem: Membrane time constant (s)
        """
        dt = 1e-3  # 1 ms timestep

        # Leaky integration: dV/dt = (V_rest - V) / tau + I
        dv = (-v_membrane + inputs) / tau_mem * dt
        v_membrane = v_membrane + dv

        # Spike detection
        spikes = (v_membrane >= v_threshold).float()

        # Reset
        v_membrane = torch.where(spikes.bool(),
                                torch.tensor(v_reset),
                                v_membrane)

        return spikes, v_membrane

    @staticmethod
    def paired_pulse_facilitation(pulse1_weight, pulse2_weight,
                                  inter_pulse_interval, tau_ppf=100e-3):
        """
        PPF: Short-term facilitation from repeated stimulation

        Facilitation decays with time constant tau_ppf
        """
        # Facilitation factor (increases with shorter intervals)
        facilitation = torch.exp(torch.tensor(-inter_pulse_interval / tau_ppf))

        # Second pulse is facilitated
        effective_weight2 = pulse2_weight * (1 + facilitation * 0.5)

        return effective_weight2

    @staticmethod
    def stdp_update(pre_spike_time, post_spike_time,
                   tau_plus=20e-3, tau_minus=20e-3,
                   A_plus=0.01, A_minus=0.01):
        """
        Spike-timing-dependent plasticity

        Args:
            pre_spike_time: Pre-synaptic spike time
            post_spike_time: Post-synaptic spike time
            tau_plus/minus: STDP time constants
            A_plus/minus: Learning rates
        """
        delta_t = post_spike_time - pre_spike_time

        if delta_t > 0:
            # Post after pre → potentiation (LTP)
            weight_change = A_plus * torch.exp(torch.tensor(-delta_t / tau_plus))
        else:
            # Pre after post → depression (LTD)
            weight_change = -A_minus * torch.exp(torch.tensor(delta_t / tau_minus))

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

    print("\n4. Paired-Pulse Facilitation")
    w1 = torch.tensor(1.0)
    w2_short = TemporalDynamics.paired_pulse_facilitation(w1, w1, 10e-3)  # 10 ms
    w2_long = TemporalDynamics.paired_pulse_facilitation(w1, w1, 100e-3)  # 100 ms
    print(f"   Baseline weight: {w1:.4f}")
    print(f"   After 10ms interval: {w2_short:.4f} (facilitated)")
    print(f"   After 100ms interval: {w2_long:.4f} (less facilitation)")

    print("\n5. STDP")
    w_change_ltp = TemporalDynamics.stdp_update(0.0, 10e-3)  # Post 10ms after pre
    w_change_ltd = TemporalDynamics.stdp_update(10e-3, 0.0)  # Pre 10ms after post
    print(f"   LTP (post after pre): {w_change_ltp:.6f}")
    print(f"   LTD (pre after post): {w_change_ltd:.6f}")

    print("=" * 60)


if __name__ == "__main__":
    demonstrate_reconfigurability()
