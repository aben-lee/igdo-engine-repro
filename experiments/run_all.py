"""Run all reproducibility experiments in order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import (  # noqa: E402
    exp_microseismic, exp_scalability, exp_insar, exp_intent_switch)


def main():
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results"

    print("=" * 70)
    print("Experiment 1: microseismic S1/S2 (static vs IGDO)")
    print("=" * 70)
    s1 = exp_microseismic.run(30, 30, outdir)
    exp_microseismic.plot(s1, outdir)
    print(s1.to_string(index=False))

    print("\n" + "=" * 70)
    print("Experiment 2: planner scalability")
    print("=" * 70)
    s2 = exp_scalability.run(100, outdir)
    exp_scalability.plot(s2, outdir)
    print(s2.to_string(index=False))

    print("\n" + "=" * 70)
    print("Experiment 3: cross-modality InSAR Mogi (static vs IGDO)")
    print("=" * 70)
    s3 = exp_insar.run(25, 25, outdir)
    exp_insar.plot(s3, outdir)
    print(s3.to_string(index=False))

    print("\n" + "=" * 70)
    print("Experiment 4: intent-driven dynamic re-planning (InSAR)")
    print("=" * 70)
    s4 = exp_intent_switch.run(outdir)
    exp_intent_switch.plot(s4, outdir)
    print(s4.to_string(index=False))

    print(f"\nAll outputs in {outdir}")


if __name__ == "__main__":
    main()
