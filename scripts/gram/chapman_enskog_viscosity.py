"""Append a Chapman--Enskog/Wilke estimate of dynamic viscosity to GRAM CSVs.

Original CSV fields and rows are retained unchanged. Species mole percent and
local temperature are taken from the original GRAM output. Pure species dilute-
gas viscosities use the Lennard--Jones (12-6) Chapman--Enskog first approximation
with the Neufeld collision-integral correlation, and mixed by Wilke's rule.

References:
  - IDAES (Chapman--Enskog + Neufeld):
    https://idaes-pse.readthedocs.io/en/2.3.0/explanations/components/property_package/general/transport_properties/chapman_enskog.html
  - Cantera GRI-Mech 3.0 transport parameters:
    https://raw.githubusercontent.com/Cantera/cantera/main/data/gri30.yaml
  - Helium transport parameters (Sandia report SAND2006-2518):
    https://www.osti.gov/servlets/purl/1177051
  - Wilke's rule (C. R. Wilke, 1950): DOI 10.1063/1.1747673

Ozone is excluded: in the Earth CSV its mole fraction never exceeds 0.0007845%,
so remaining mole fractions are renormalized. This is a trace-species approximation.
The calculated transport coefficients are illustrative gas-phase dilute-mixture
values; not a validation against measured multi-component viscosity at every
planetary altitude, nor an assertion of continuum validity at high Knudsen number.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Each entry: molar mass in g/mol, LJ sigma in Angstrom, epsilon/k_B in Kelvin
# Cantera GRI-3.0 transport library except He: Sandia SAND2006-2518.
SPECIES = {
    "N2": (28.0134, 3.621, 97.53),
    "O2": (31.9988, 3.458, 107.4),
    "Ar": (39.948, 3.33, 136.5),
    "CO2": (44.0095, 3.763, 244.0),
    "CO": (28.0101, 3.65, 98.1),
    "He": (4.002602, 2.576, 10.22),
    "H": (1.00794, 2.05, 145.0),
    "N": (14.0067, 3.298, 71.4),
    "O": (15.999, 2.75, 80.0),
    "CH4": (16.04246, 3.746, 141.4),
    "N2O": (44.0128, 3.828, 232.4),
    "H2O": (18.01528, 2.605, 572.4),
    "H2": (2.01588, 2.92, 38.0),
}
# No well-sourced O3 LJ parameters in adopted transport database; max O3
# mole fraction in supplied Earth CSV = 7.8449e-6, negligible in this model.
OMITTED_TRACE = {"O3"}
OUTPUT_COLUMN = "DynamicViscosity_ChapmanEnskog_Wilke_Pa_s"


def species_mu(T: float, species: str) -> float:
    M, sigma, well_depth = SPECIES[species]
    tstar = T / well_depth
    # Neufeld et al. collision integral for LJ 12-6 viscosity
    omega = (1.16145 * tstar**(-0.14874)
             + 0.52487 * math.exp(-0.77320 * tstar)
             + 2.16178 * math.exp(-2.43787 * tstar))
    # M: g/mol, sigma: Angstrom, T: K, viscosity: Pa*s
    return 2.6693e-6 * math.sqrt(M * T) / (sigma * sigma * omega)


def mixture_mu(T: float, x: dict[str, float]) -> float:
    species = [s for s, val in x.items() if val > 0]
    if not species:
        raise ValueError('Zero mixture')
    total = sum(x[s] for s in species)
    mol = {s: x[s]/total for s in species}
    if len(species) == 1:
        return species_mu(T, species[0])
    mu = {s: species_mu(T, s) for s in species}
    out = 0.0
    for i in species:
        mi = SPECIES[i][0]
        phi_denom = 0.0
        for j in species:
            mj = SPECIES[j][0]
            phi = ((1 + math.sqrt(mu[i] / mu[j]) * (mj / mi)**0.25)**2
                   / math.sqrt(8 * (1 + mi / mj)))
            phi_denom += mol[j] * phi
        out += mol[i] * mu[i] / phi_denom
    return out


def process(source: str, output: str):
    src = ROOT / source
    dst = ROOT / output
    with src.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        original_headers = reader.fieldnames
        assert original_headers is not None
        mole_headers = [c for c in original_headers if c.endswith('mole_pct')]
        unknown = [c for c in mole_headers if c[:-8] not in SPECIES and c[:-8] not in OMITTED_TRACE]
        if unknown:
            raise ValueError(f'Unrecognised species columns: {unknown}')
        rows = []
        max_omitted = 0.0
        min_Tstar = float('inf')
        max_Tstar = -float('inf')
        for row in reader:
            T = float(row['Temperature_K'])
            if not T > 0 or not math.isfinite(T):
                raise ValueError(f'Invalid temperature: {T}')
            x = {}
            for column in mole_headers:
                name = column[:-len('mole_pct')]
                frac = float(row[column]) / 100.0
                if not math.isfinite(frac) or frac < -1e-9:
                    raise ValueError(f'Invalid mole fraction for {name}: {frac}')
                frac = max(0.0, frac)
                if name in OMITTED_TRACE:
                    max_omitted = max(max_omitted, frac)
                elif frac > 0:
                    x[name] = frac
                    min_Tstar = min(min_Tstar, T / SPECIES[name][2])
                    max_Tstar = max(max_Tstar, T / SPECIES[name][2])
            if not 0.99 < sum(x.values()) + sum(max(0.0,float(row[c])/100) for c in mole_headers if c[:-len('mole_pct')] in OMITTED_TRACE) < 1.01:
                raise ValueError(f'Invalid species balance at height={row["Height_km"]}')
            viscosity = mixture_mu(T, x)
            if not viscosity > 0 or not math.isfinite(viscosity):
                raise ValueError('Invalid viscosity')
            row[OUTPUT_COLUMN] = f'{viscosity:.12g}'
            rows.append(row)
    with dst.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[*original_headers, OUTPUT_COLUMN])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{output}: {len(rows)} rows; min/max viscosity '
          f'{min(float(r[OUTPUT_COLUMN]) for r in rows):.6g} / '
          f'{max(float(r[OUTPUT_COLUMN]) for r in rows):.6g} Pa s, '
          f'largest omitted species fraction {max_omitted:.7g}, '
          f'T* of present species {min_Tstar:.3f}-{max_Tstar:.3f}')
    for alt in [0,150,500,1000,2200]:
        row = rows[alt]
        print(f"  {alt:4d} km: T={float(row['Temperature_K']):8.3f} K, "
              f"mu={float(row[OUTPUT_COLUMN]):.8e} Pa s")
    return dst


if __name__ == '__main__':
    for source, output in [
        ('Earth_GRAM_global_reference_mean_0-2200km_1km.csv', 'earth1.csv'),
        ('Mars_GRAM_global_reference_mean_0-2200km_1km.csv', 'mars1.csv'),
        ('Titan_GRAM_final_0-2200km_1km.csv', 'titan1.csv'),
    ]:
        process(source, output)
