"""Reproduce rarefaction regime thresholds from original recovered CSVs.

Read-only. Uses first root of ln(lambda(z)/L) - ln(Kncrit) within 0..2200 km;
PCHIP interpolation of logged mean free path between native GRAM-derived nodes.
Does NOT validate high-altitude model fidelity, recompute GRAM, or change classifier.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

ORIGINALS = {'Earth': 'earth3.csv', 'Mars':'mars3.csv', 'Titan':'titan3.csv'}
Z_PLOT = {'Earth':100.,'Mars':100.,'Titan':500.}
THRESHOLDS = [0.01, 0.1, 10.0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2] / 'data')
    ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--length-m',type=float,default=1.)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    results=[];meta={}
    for planet,filename in ORIGINALS.items():
        path=args.data_dir / filename
        raw=path.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
        d=pd.read_csv(path)
        z=d.Height_km.to_numpy(float)
        lam=d.MeanFreePath_ViscosityBased_m.to_numpy(float)
        assert np.all(np.isfinite(z)) and np.all(np.isfinite(lam)) and np.all(lam>0)
        assert len(z)==2201 and np.allclose(z,np.arange(2201))
        curve=PchipInterpolator(z,np.log(lam/args.length_m),extrapolate=False)
        alt={}
        for kn in THRESHOLDS:
            v=np.log(lam/args.length_m)-np.log(kn)
            cross=np.flatnonzero((v[:-1] <= 0)&(v[1:]>0))
            assert len(cross)==1, (planet,kn,len(cross))
            i=int(cross[0])
            alt[str(kn)] = float(brentq(lambda x:float(curve(x)-np.log(kn)),z[i],z[i+1],xtol=1e-11))
        bounds=[alt['0.01'],alt['0.1'],alt['10.0']]
        kn_top=float(np.exp(curve(Z_PLOT[planet])))
        row={'planet':planet,'z_kn_0_01_km':bounds[0], 'z_kn_0_1_km':bounds[1],
             'z_kn_10_km':bounds[2], 'slip_interval_km':bounds[1]-bounds[0],
             'transition_interval_km':bounds[2]-bounds[1],
             'rarefied_to_free_molecular_interval_km':bounds[2]-bounds[0],
             'plotted_zmax_km':Z_PLOT[planet], 'Kn_at_plot_top':kn_top,
             'regime_at_plot_top':('no-slip continuum' if kn_top<0.01 else 'slip' if kn_top<0.1 else 'transitional' if kn_top<10 else 'free molecular')}
        results.append(row)
        meta[planet]={'source_filename':filename,'sha256':checksum,'n_rows':len(d),'n_columns':len(d.columns),
                      'altitude_min_km':float(z[0]),'altitude_max_km':float(z[-1])}
        print(planet, {k:round(row[k],3) for k in ['z_kn_0_01_km','z_kn_0_1_km','z_kn_10_km']},'plotted upper regime',row['regime_at_plot_top'])
    c=args.out_dir/'rarefaction_regime_crossings_L1m.csv'
    with c.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0].keys()));writer.writeheader();writer.writerows(results)
    gamma_ref=1.4;ma_max=0.3;re_min=1e4
    q=np.sqrt(np.pi*gamma_ref/2)
    analytic={'gamma_assumed':gamma_ref,'Ma_upper':ma_max,'Re_lower':re_min,
              'Kn_L_at_corner':float(q*ma_max/re_min),
              'delta99_over_L_at_corner':float(5/np.sqrt(re_min)),
              'Kn_delta99_at_corner':float(q*ma_max/(5*np.sqrt(re_min))),
              'Kn_delta99_to_Kn_L_ratio':float(np.sqrt(re_min)/5),
              'interpretation':'Illustrative Blasius delta99; not actual gradient-length Kn or turbulent BL thickness.'}
    j=args.out_dir/'rarefaction_regime_audit.json'
    j.write_text(json.dumps({'data_sources':meta,'method':{'L_m':args.length_m,'thresholds':THRESHOLDS,
       'interpolation':'PCHIP on log(lambda) vs native 1 km z (no new GRAM evaluations)',
       'uncertainty':'Upper-altitude extrapolation within existing GRAM-derived profile is not validation.'},
       'planets':results,'analytical_corner_check':analytic},indent=2)+'\n')
    print('At Re=1e4, Ma=.3, gamma=1.4:',analytic)

if __name__=='__main__': main()
