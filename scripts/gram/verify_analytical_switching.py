#!/usr/bin/env python3
"""Test the *surface-calibrated* isothermal Re/Ma/Kn switching relations.

Inputs: the three UNMODIFIED, uploaded GRAM-derived CSVs supplied separately.
Outputs: analytical_switching_comparison.csv, kn_scale_height_comparison.csv,
         verification_results.json.

Assumptions: isothermal hydrostatic exponential rho=p=rho0*exp(-z/H0), constant mu,
constant a, and lambda=lambda0*exp(z/H0), with surface H0 derived from p(0),p(1 km).
This is an intentionally simplified comparison, NOT an independently validated model.
Original altitude resolution 1 km; crossings use log-PCHIP interpolation on the raw
samples, not new GRAM runs. Do not modify source files.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

SPEC = {'Earth':('earth3.csv',100),'Mars':('mars3.csv',100),'Titan':('titan3.csv',500)}
KN_CRIT=0.01; L=1.0; U=10.0; MA_CRIT=.3
RE_LIST=(1e4,5e5); L_LIST=(0.1,1.,10.)

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()

def crossings_from_loginterpolant(x, y, level):
 """Return all crossings from positive y, interpolating log y on native 1 km nodes."""
 f=PchipInterpolator(x,np.log(y))
 v=np.log(y)-np.log(level)
 ix=np.flatnonzero((v[:-1]*v[1:]<=0)&(v[:-1]!=v[1:]))
 return [float(brentq(lambda z:float(f(z)-np.log(level)),x[i],x[i+1])) for i in ix]

def writerows(path, rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2] / 'data')
 ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
 args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
 allrows=[];knrows=[];records={}
 for planet,(name,zmax) in SPEC.items():
  path=args.data_dir/name
  d=pd.read_csv(path)
  cols=['Height_km','Pressure_Pa','Density_kgm3','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s','SpeedOfSound_ms','MeanFreePath_ViscosityBased_m']
  assert all(c in d.columns for c in cols), (planet,cols)
  assert len(d)==2201
  z=d.Height_km.to_numpy(float)
  assert np.allclose(z,np.arange(2201))
  for c in cols[1:]:
   assert (np.isfinite(d[c]) & (d[c]>0)).all(),(planet,c)
  rho=d.Density_kgm3.to_numpy(float);mu=d.DynamicViscosity_ChapmanEnskog_Wilke_Pa_s.to_numpy(float)
  a=d.SpeedOfSound_ms.to_numpy(float);lam=d.MeanFreePath_ViscosityBased_m.to_numpy(float)
  pres=d.Pressure_Pa.to_numpy(float)
  H0=1/np.log(pres[0]/pres[1]);R0=float(rho[0]*U*L/mu[0]);a0=float(a[0]); lam0=float(lam[0]);a_cr=MA_CRIT*a[:zmax+1]
  re=rho*U*L/mu
  planet_cases=[]
  for limit in RE_LIST:
   ziso=float(H0*np.log(R0/limit))
   xall=crossings_from_loginterpolant(z,re,limit)
   valids=[x for x in xall if 0<=x<=zmax]
   actual=float(valids[0]) if valids else None
   row={'planet':planet,'Re_threshold':int(limit),'U_ms':U,'L_m':L,
        'isothermal_z_km':round(ziso,6) if ziso>=0 else None,
        'GRAM_z_km':round(actual,6) if actual is not None else None,
        'analytical_minus_GRAM_km':round(ziso-actual,6) if actual is not None else None,
        'isothermal_U_Ma03_ms':round(MA_CRIT*a0,6),
        'GRAM_U_Ma03_min_ms':round(float(a_cr.min()),6),
        'GRAM_U_Ma03_max_ms':round(float(a_cr.max()),6),
        'GRAM_U_Ma03_max_abs_delta_from_isothermal_ms':round(float(np.max(np.abs(a_cr-MA_CRIT*a0))),6),
        'H0_km':round(float(H0),6)}
   allrows.append(row);planet_cases.append(row)
  lk=PchipInterpolator(z,np.log(lam))
  zkn0=crossings_from_loginterpolant(z,lam,KN_CRIT*L)[0]
  Hloc=float(1/lk.derivative()(zkn0))
  zisoKn=float(H0*np.log(KN_CRIT*L/lam0))
  zkns={str(v):crossings_from_loginterpolant(z,lam,KN_CRIT*v)[0] for v in L_LIST}
  slopes_left=(zkns['1.0']-zkns['0.1'])/np.log(10)
  slopes_right=(zkns['10.0']-zkns['1.0'])/np.log(10)
  knrow={'planet':planet,'H0_surface_pressure_km':round(H0,6),
    'z_Kn_GRAM_L1_km':round(zkn0,6), 'z_Kn_isothermal_L1_km':round(zisoKn,6),
    'isothermal_minus_GRAM_km':round(zisoKn-zkn0,6),
    'isothermal_relative_error_pct':round((zisoKn-zkn0)/zkn0*100,6),
    'Hlambda_local_at_L1_km':round(Hloc,6),
    'slope_z_vs_lnL_L0p1_to_1_km':round(slopes_left,6),
    'slope_z_vs_lnL_L1_to_10_km':round(slopes_right,6),
    'surface_H0_predicted_slope_km':round(H0,6),
    'actual_z_Kn_L0p1_km':round(zkns['0.1'],6),
    'actual_z_Kn_L10_km':round(zkns['10.0'],6)}
  knrows.append(knrow)
  records[planet]={'source_filename':name,'source_sha256':sha256(path),'range_km':[0,zmax],
     'surface_reference':{'rho0':float(rho[0]),'mu0':float(mu[0]),'a0':a0,'lambda0':lam0,'H0':H0},
     'Re_switch_cases':planet_cases,'Kn_scale_height':knrow}
 writerows(args.out_dir/'analytical_switching_comparison.csv',allrows)
 writerows(args.out_dir/'kn_scale_height_comparison.csv',knrows)
 report={'method':{'U_mps':U,'L_m':L,'H0':'1 km / ln[p(0km)/p(1km)]',
  'Re_analytical':'Re0 exp(-z/H0), Re0=rho0*U*L/mu0',
  'Mach_analytical':'Ma=U/a0; boundary speed=0.3*a0',
  'Kn_analytical':'lambda0 exp(z/H0) / L',
  'actual':'Re(z) from rho,mu; Mach critical speed=0.3 a(z); Kn from lambda; PCHIP log for crossover locations'},
  'planets':records}
 (args.out_dir/'verification_results.json').write_text(json.dumps(report,indent=2)+'\n')
 print('\nRe/Mach analytical comparison:')
 for r in allrows:print(r)
 print('\nKn/slope comparison:')
 for r in knrows:print(r)
if __name__=='__main__':main()
