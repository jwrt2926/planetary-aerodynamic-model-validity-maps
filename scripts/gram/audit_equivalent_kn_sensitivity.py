"""Verify exact Kn-criterion/vehicle-length/mfp degeneracies and local approximation.
Reads untouched original CSVs; produces CSV, JSON, and SHA256 manifest. No original GRAM re-runs.
"""
from pathlib import Path
import numpy as np, pandas as pd, json, csv, hashlib
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

FILES={'Earth':'earth3.csv','Mars':'mars3.csv','Titan':'titan3.csv'}
H0_ORIG = {'Earth':8.456506252543749,'Mars':10.362494007031147,'Titan':19.783876637773936}
OUT=Path(__file__).resolve().parents[2] / 'results' / 'reproduced'
OUT.mkdir(parents=True, exist_ok=True)
src=Path(__file__).resolve().parents[2] / 'data'
rows=[]; reports={}
for planet,name in FILES.items():
    path=src/name
    d=pd.read_csv(path)
    z=d['Height_km'].to_numpy(float)
    lam=d['MeanFreePath_ViscosityBased_m'].to_numpy(float)
    assert np.array_equal(z,np.arange(2201,dtype=float)) and np.all(np.isfinite(lam)) and np.all(lam>0)
    f=PchipInterpolator(z,np.log(lam),extrapolate=False)
    def zx(kncrit=.01,L=1.,lamfactor=1.):
        threshold=np.log(kncrit*L/lamfactor)
        dv=f(z)-threshold
        ix=np.flatnonzero((dv[:-1]*dv[1:])<=0)
        assert len(ix)==1, (planet,kncrit,L,lamfactor,len(ix))
        i=int(ix[0]);return float(brentq(lambda x: float(f(x)-threshold),z[i],z[i+1]))
    base=zx()
    grad=float(f.derivative()(base))
    Hlocal=1/grad
    H0=1/np.log(float(d.Pressure_Pa.iloc[0])/float(d.Pressure_Pa.iloc[1]))
    assert abs(H0-H0_ORIG[planet])<1e-9
    values=[]
    for label,kt,L,lf in [('Baseline',.01,1,1),('Kn threshold 0.008',.008,1,1),('Kn threshold 0.012',.012,1,1),('Kn threshold 0.1',.1,1,1),('Length L=10m',.01,10,1),('Lambda +10%',.01,1,1.1),('Equivalent Kn threshold for lambda +10%',.01/1.1,1,1),('Lambda -10%',.01,1,.9),('Equivalent Kn threshold for lambda -10%',.01/.9,1,1),('Lambda +20%',.01,1,1.2),('Lambda -20%',.01,1,.8)]:
        cross=zx(kt,L,lf)
        actual_delta=cross-base
        effective_factor=(kt*L/lf)/(.01*1)
        approx_delta=Hlocal*np.log(effective_factor)
        row={'Planet':planet,'Scenario':label,'Kncrit':kt,'L_m':L,'lambda_factor':lf,'Crossing_km':cross,'Delta_vs_baseline_km':actual_delta,'Local_approx_shift_km':approx_delta,'Approx_minus_exact_km':approx_delta-actual_delta}
        rows.append(row);values.append(row)
    a=values[3]['Crossing_km'];b=values[4]['Crossing_km']
    assert abs(a-b)<1e-9
    assert abs(values[5]['Crossing_km']-values[6]['Crossing_km'])<1e-9
    assert abs(values[7]['Crossing_km']-values[8]['Crossing_km'])<1e-9
    reports[planet]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'H_pressure_surface_km':H0,'Hlambda_local_km':Hlocal,'Hlambda_over_H0':Hlocal/H0,'dlnlambda_dz_perkm':grad,'z0_km':base,'kn0p1_z_km':a,'kn0p01_L10_z_km':b,'plus10_shift_exact_km':values[5]['Delta_vs_baseline_km'],'plus10_shift_local_km':values[5]['Local_approx_shift_km'],'minus10_shift_exact_km':values[7]['Delta_vs_baseline_km'],'minus10_shift_local_km':values[7]['Local_approx_shift_km']}
with (OUT/'issue5_boundary_equivalence.csv').open('w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
(OUT/'issue5_calculations.json').write_text(json.dumps({'input_files':FILES,'planet_results':reports,'method':'PCHIP on log lambda as previously used. Exact Kn crossing for each criterion, length and uniformly-scaled mean-free-path.'},indent=2)+'\n')
for p,r in reports.items():
    print(p, 'H_lambda',r['Hlambda_local_km'],'H0',r['H_pressure_surface_km'],'ratio',r['Hlambda_over_H0'], 'Kn .1 vs L10',r['kn0p1_z_km'],r['kn0p01_L10_z_km'])
    print('  +/-10%, exact',r['plus10_shift_exact_km'],r['minus10_shift_exact_km'],'approximated',r['plus10_shift_local_km'],r['minus10_shift_local_km'])
