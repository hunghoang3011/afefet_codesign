# analysis/make_figs.py
import os, json, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Matplotlib defaults
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.linestyle": ":"})

# Repo root and imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from afefet_snn.device_physics import MFMISDevice, TemporalDynamics
except Exception as e:
    MFMISDevice = None
    TemporalDynamics = None
    print(f"[warn] device_physics import failed ({e}). Physics-only figures will be skipped if classes are missing.")

# Paths
METRICS_PATH = os.path.join(ROOT, "results", "metrics", "afefet_full_results.json")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    print(f"✓ saved: {path}")
    plt.close()

def load_results():
    if not os.path.exists(METRICS_PATH):
        print(f"[info] {METRICS_PATH} not found. Will only render physics figures.")
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

# ---------------------- Figures that depend on training logs ----------------------
def fig_acc_vs_epoch(history):
    if not history:
        return
    epochs_axis = np.arange(1, len(history["train_acc"])+1)
    plt.figure()
    plt.plot(epochs_axis, history["train_acc"], label="Train Acc")
    plt.plot(epochs_axis, history["test_acc"], label="Test Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epoch"); plt.legend()
    savefig("acc_vs_epoch.png")

def fig_energy_per_epoch(history):
    if not history or not history.get("energy_per_epoch"):
        return
    plt.figure()
    y = np.array(history["energy_per_epoch"]) * 1e12  # J -> pJ
    plt.plot(np.arange(1, len(y)+1), y, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Energy per op (pJ)")
    plt.title("Average Write Energy per Epoch")
    savefig("energy_per_epoch.png")

def fig_alpha_vs_epoch(history):
    if not history:
        return
    plt.figure()
    epochs_axis = np.arange(1, len(history["alpha_fc1"])+1)
    plt.plot(epochs_axis, history["alpha_fc1"], label="alpha fc1")
    plt.plot(epochs_axis, history["alpha_fc2"], label="alpha fc2")
    plt.xlabel("Epoch"); plt.ylabel("Alpha"); plt.title("Alpha Evolution"); plt.legend()
    savefig("alpha_vs_epoch.png")

def fig_scenario_comparison(scenario_results):
    if not scenario_results:
        return
    labels = ["low_power", "balanced", "high_accuracy"]
    vals = [scenario_results.get(k, np.nan) for k in labels]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Accuracy (%)")
    plt.title("Scenario Comparison (No Retention)")
    savefig("scenario_comparison.png")

def fig_retention_accuracy(retention_results):
    if not retention_results:
        return
    def order_xy(d):
        t = np.array([float(k) for k in d.keys()])
        a = np.array([float(v) for v in d.values()])
        idx = np.argsort(t); return t[idx], a[idx]
    if "STM" in retention_results and "LTM" in retention_results:
        t_stm, a_stm = order_xy(retention_results["STM"])
        t_ltm, a_ltm = order_xy(retention_results["LTM"])
        plt.figure()
        plt.semilogx(t_stm, a_stm, marker="o", label="STM (volatile)")
        plt.semilogx(t_ltm, a_ltm, marker="o", label="LTM (stable)")
        plt.xlabel("Time since write (s)"); plt.ylabel("Accuracy (%)")
        plt.title("Retention Impact on Accuracy"); plt.legend()
        savefig("retention_accuracy.png")

# ---------------------- Physics figures (no training required) ----------------------
def fig_voltage_division(area_ratios=(0.5, 1.0, 1.5, 2.0), V_applied=4.5):
    if MFMISDevice is None: return
    plt.figure()
    for r in area_ratios:
        dev = MFMISDevice(area_ratio=r, base_voltage=V_applied)
        V_FE, _ = dev.voltage_division(torch.tensor(V_applied))
        plt.scatter([r], [float(V_FE)], label=f"A_MIS/A_FE={r}")
    plt.xlabel("Area ratio (A_MIS / A_FE)")
    plt.ylabel("V_FE (V)")
    plt.title("Voltage Division vs Area Ratio")
    plt.legend()
    savefig("phys_voltage_division.png")

def fig_phase_map(Vmin=2.0, Vmax=6.0, Tmin=1e-7, Tmax=1e-1, nV=80, nT=80, area_ratio=1.0):
    """
    Vectorized mode logic in device_physics uses scalar comparisons and raises an error
    when V is a vector. Work around by iterating over scalar V.
    """
    if MFMISDevice is None: return
    V_grid = torch.linspace(Vmin, Vmax, nV)
    T_grid = torch.logspace(np.log10(Tmin), np.log10(Tmax), nT)
    dev = MFMISDevice(area_ratio=area_ratio)
    P = torch.zeros(nT, nV)
    for j, V in enumerate(V_grid):
        for i, t in enumerate(T_grid):
            p, stm_mask, _ = dev.switching_probability(V, float(t), use_nls=True)  # scalar V and scalar t
            P[i, j] = p
    plt.figure()
    plt.imshow(P.numpy(), aspect="auto", origin="lower",
               extent=[Vmin, Vmax, np.log10(Tmin), np.log10(Tmax)])
    cbar = plt.colorbar(); cbar.set_label("P(switch)")
    plt.xlabel("V_applied (V)"); plt.ylabel("log10(pulse width (s))")
    plt.title(f"Switching Probability Map @ area_ratio={area_ratio}")
    savefig("phys_phase_map.png")

def fig_retention_model(times=np.logspace(-3, 5, 120), mode="STM"):
    if MFMISDevice is None: return
    dev = MFMISDevice(area_ratio=1.0)
    state0 = torch.tensor(1.0)
    y = []
    for t in times:
        y.append(float(dev.retention_decay(state0, float(t), mode, detrapping_factor=0.5)))
    plt.figure()
    plt.semilogx(times, y)
    plt.xlabel("Time (s)"); plt.ylabel("Normalized state")
    plt.title(f"Retention Decay ({mode})")
    savefig(f"phys_retention_{mode.lower()}.png")

def fig_ppf_demo():
    if TemporalDynamics is None: return
    intervals = np.logspace(-6, -2, 60)  # 1 µs → 10 ms
    w1 = torch.tensor(1.0)
    y = []
    for dt in intervals:
        w2 = TemporalDynamics.paired_pulse_facilitation(w1, w1, float(dt))
        y.append(float(w2 / w1))
    plt.figure()
    plt.semilogx(intervals, y, marker="o")
    plt.xlabel("Inter-pulse interval (s)"); plt.ylabel("w2 / w1")
    plt.title("Paired-Pulse Facilitation")
    savefig("phys_ppf.png")

def fig_stdp_demo():
    if TemporalDynamics is None: return
    dts = np.linspace(-0.02, 0.02, 200)  # -20 ms → +20 ms
    y = []
    for dt in dts:
        y.append(float(TemporalDynamics.stdp_update(0.0, float(dt))))
    plt.figure()
    plt.plot(dts*1e3, y)
    plt.xlabel("Δt (ms, post - pre)"); plt.ylabel("Δw")
    plt.title("STDP Window")
    savefig("phys_stdp.png")

def main():
    results = load_results()

    # Training-dependent figures (if results exist)
    if results is not None:
        hist = results.get("history")
        scen = results.get("scenario_accuracies")
        ret  = results.get("retention_results")
        fig_acc_vs_epoch(hist)
        fig_energy_per_epoch(hist)
        fig_alpha_vs_epoch(hist)
        fig_scenario_comparison(scen)
        fig_retention_accuracy(ret)

    # Physics figures (always try)
    fig_voltage_division(area_ratios=(0.5, 1.0, 1.5, 2.0), V_applied=4.5)
    fig_phase_map(area_ratio=1.0)
    fig_retention_model(mode="STM")
    fig_retention_model(mode="LTM")
    fig_ppf_demo()
    fig_stdp_demo()

    print("Done.")

if __name__ == "__main__":
    main()
