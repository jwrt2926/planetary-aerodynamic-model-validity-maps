#!/usr/bin/env python3
"""Reproduce illustrative isothermal-vs-GRAM classification decisions.

Data: the provided processed Earth/Mars/Titan CSVs.
Does NOT independently reconstruct NASA GRAM input runs or validate aerodynamics.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

HERE=Path(__file__).resolve().parent
DATA=Path(__file__).resolve().parents[2] / 'data'
PLANETS={'Earth':('earth3.csv',90.0),'Mars':('mars3.csv',78.0),'Titan':('titan3.csv',350.0)}
L=1.0; U=100.0
KNCRIT=0.01; MACRIT=0.3; REMIN=1e4; RETR=5e5
MU='DynamicViscosity_ChapmanEnskog_Wilke_Pa_s'; LAM='MeanFreePath_ViscosityBased_m'

def select(re,ma,kn):
    if kn>=KNCRIT: return 'Rarefied/slip-corrected'
    if ma>=MACRIT: return 'Compressible Navier-Stokes'
    if re<REMIN: return 'Incompressible Navier-Stokes'
    if re<RETR: return 'Blasius'
    return 'Broader boundary-layer'

def crossing(z,lam,target):
    # log-space PCHIP crossing, with monotonicity in the first-crossing bracket
    z=np.asarray(z,dtype=float)
    lam=np.asarray(lam,dtype=float)
    hits=np.where(lam>=target)[0]
    if not len(hits): raise ValueError('No crossing within original CSV range')
    i=int(hits[0])
    if i==0: return float(z[0])
    if lam[i]==target:return float(z[i])
    # smoothly interpolate log(lambda) across full height, then root in first bracket
    from scipy.optimize import brentq
    f=PchipInterpolator(z,np.log(lam))
    return float(brentq(lambda q:float(f(q)-np.log(target)),z[i-1],z[i]))

rows=[];cross=[];sources={}
for planet,(fname,z_example) in PLANETS.items():
    fpath=DATA/fname
    sources[fname]=hashlib.sha256(fpath.read_bytes()).hexdigest()
    df=pd.read_csv(fpath).sort_values('Height_km').reset_index(drop=True)
    z=df['Height_km'].to_numpy(float)
    assert len(z)==2201 and np.array_equal(z,np.arange(2201,dtype=float))
    s=df.iloc[0];z1=df.iloc[1]
    H0=float(-1.0/np.log(z1['Pressure_Pa']/s['Pressure_Pa']))
    # Example points intentionally selected within the different Kn-boundary bands.
    x=df.iloc[int(z_example)]
    rho0=float(s['Density_kgm3']);a0=float(s['SpeedOfSound_ms']);mu0=float(s[MU]);lam0=float(s[LAM])
    r_true=float(x['Density_kgm3']);a_true=float(x['SpeedOfSound_ms']);mu_true=float(x[MU]);lam_true=float(x[LAM]);
    rho_iso=rho0*np.exp(-z_example/H0)
    lam_iso=lam0*np.exp(z_example/H0)
    re_iso=rho_iso*U*L/mu0;ma_iso=U/a0;kn_iso=lam_iso/L
    re_true=r_true*U*L/mu_true;ma_true=U/a_true;kn_true=lam_true/L
    z_true=crossing(z,df[LAM].to_numpy(float),KNCRIT*L)
    z_iso=H0*np.log(KNCRIT*L/lam0)
    cross.append({'planet':planet,'surface_pressure_scale_height_km':H0,'gram_Kn_boundary_km':z_true,'isothermal_Kn_boundary_km':z_iso,'isothermal_minus_gram_km':z_iso-z_true, 'isothermal_minus_gram_percent_of_gram':(z_iso/z_true-1)*100})
    rows.append({'planet':planet,'altitude_km':z_example,'velocity_m_per_s':U,'length_m':L,
      'gram_Re':re_true,'gram_Ma':ma_true,'gram_Kn':kn_true,'gram_selected_model':select(re_true,ma_true,kn_true),
      'isothermal_Re':re_iso,'isothermal_Ma':ma_iso,'isothermal_Kn':kn_iso,'isothermal_selected_model':select(re_iso,ma_iso,kn_iso),
      'different_model_selected':select(re_iso,ma_iso,kn_iso)!=select(re_true,ma_true,kn_true)})

OUT=Path(__file__).resolve().parents[2] / 'results' / 'reproduced'
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT/'baseline_decision_examples.csv',index=False,float_format='%.12g')
pd.DataFrame(cross).to_csv(OUT/'boundary_comparison.csv',index=False,float_format='%.12g')
(OUT/'baseline_results.json').write_text(json.dumps({'method':'1-km surface-pressure-calibrated isothermal model versus actual GRAM-derived atmospheric properties; U=100m/s, L=1m','thresholds':{'Kn':KNCRIT,'Mach':MACRIT,'Re_min':REMIN,'Re_trans':RETR},'sources_sha256':sources,'examples':rows,'crossings':cross},indent=2))
for r in rows:print(f"{r['planet']} @ {r['altitude_km']:g} km: {r['gram_selected_model']} (GRAM Kn={r['gram_Kn']:.4g}) vs {r['isothermal_selected_model']} (iso Kn={r['isothermal_Kn']:.4g})")
for c in cross:print(f"{c['planet']}: GRAM {c['gram_Kn_boundary_km']:.4f} km; iso {c['isothermal_Kn_boundary_km']:.4f} km; Δ {c['isothermal_minus_gram_km']:+.2f} km")
