"""Sensitivity analysis: audit classification-area metric, wider thresholds, and axes.

Reads verified uploaded original earth3.csv, mars3.csv, titan3.csv.
No atmospheric data modifications. No new GRAM calls.
Classification according to published nominal branch order:
  Kn >= 0.01: rarefied / slip-corrected (4)
  Ma >= 0.3: compressible NS (3)
  Re < (5/epsilon)^2: incompressible NS (0)
  Re < Re_transition: Blasius (1)
  else: broader boundary layer (2)
Grid z step 0.1 km, log velocity 0.1–1000 m/s, 1201 samples.
"""
from pathlib import Path
import argparse, hashlib, csv, json
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

FNS={"Earth":"earth3.csv","Mars":"mars3.csv","Titan":"titan3.csv"}
ZMAX={"Earth":100,"Mars":100,"Titan":500}
KNC=.01; MAC=.3; RE_MIN=1e4; RE_T=.5e6

def profile(p, datadir):
    path=datadir/FNS[p]; df=pd.read_csv(path)
    z=df.Height_km.to_numpy(float)
    return {"path":path,"df":df,"z":z,
      **{key:PchipInterpolator(z,np.log(df[col].to_numpy(float)),extrapolate=False)
       for key,col in [('lam','MeanFreePath_ViscosityBased_m'),('mu','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s'),('rho','Density_kgm3'),('a','SpeedOfSound_ms'),('p','Pressure_Pa')]}}

def crossing(f, kncrit=KNC):
    z=f['z']; v=f['lam'](z)-np.log(kncrit)
    candidates=np.where(v[:-1]*v[1:]<=0)[0]
    if len(candidates)!=1:raise ValueError(f'Expected unique crossing, got {len(candidates)}')
    i=candidates[0]
    return float(brentq(lambda zz:float(f['lam'](zz)-np.log(kncrit)),z[i],z[i+1]))

def classifier(f, end_z, retr=RE_T, kncrit=KNC, ma_crit=MAC, emin=RE_MIN, velocity_pts=1201):
    zz=np.linspace(0,end_z,int(round(end_z*10))+1)
    uu=np.geomspace(.1,1000,velocity_pts)
    re=np.exp(f['rho'](zz)-f['mu'](zz))[:,None]*uu[None,:]
    ma=uu[None,:]/np.exp(f['a'](zz))[:,None]
    kn=np.exp(f['lam'](zz))[:,None]  # L=1m
    cls=np.zeros_like(re,dtype='uint8')
    cls[(re>=emin)&(re<retr)&(ma<ma_crit)&(kn<kncrit)]=1
    cls[(re>=retr)&(ma<ma_crit)&(kn<kncrit)]=2
    cls[(ma>=ma_crit)&(kn<kncrit)]=3
    cls[np.broadcast_to(kn>=kncrit,cls.shape)]=4
    return cls

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--data-dir',default=str(Path(__file__).resolve().parents[2] / 'data'))
    pa.add_argument('--output-dir',default=str(Path(__file__).resolve().parent))
    opts=pa.parse_args();d=Path(opts.data_dir);out=Path(opts.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[];window=[];bl=[];oat=[];profiles={};summary=[]
    for planet,base in ZMAX.items():
        f=profile(planet,d);profiles[planet]=f
        sha=hashlib.sha256(f['path'].read_bytes()).hexdigest()
        cb=crossing(f)
        H0=1/np.log(f['df'].Pressure_Pa.iloc[0]/f['df'].Pressure_Pa.iloc[1]);lambdaH=1/f['lam'].derivative()(cb)
        kb={key:crossing(f,key) for key in [.008,.01,.012,.1]}
        summary.append({'planet':planet,'sha256':sha,'H0_km':float(H0),'nominal_Kn_cross_km':cb,
                        'kn_008_cross_km':kb[.008],'kn_012_cross_km':kb[.012],
                        'kn_01_slip_cross_km':kb[.1],'z_over_surface_H':cb/H0,
                        'local_Hlambda_at_Kn_km':lambdaH,'nominal_window_km':base})
        nominal=classifier(f,base)
        # Re-evaluate the four narrow, one-at-a-time threshold ranges.
        # These percentages are descriptive for the chosen plotting window only.
        for parameter,vals in [('Kn',[.008,.012]),('Ma',[.25,.35]),
                               ('epsilon',[.04,.06]),('Re_tr',[4e5,6e5])]:
            for value in vals:
                arg={'kncrit':KNC,'ma_crit':MAC,'emin':RE_MIN,'retr':RE_T}
                if parameter=='Kn':arg['kncrit']=value
                elif parameter=='Ma':arg['ma_crit']=value
                elif parameter=='epsilon':arg['emin']=(5/value)**2
                elif parameter=='Re_tr':arg['retr']=value
                pert=classifier(f,base,**arg)
                oat.append({'planet':planet,'parameter':parameter,'value':value,
                            'classification_cells_changed_pct':float(100*np.mean(nominal!=pert)),
                            'window_zmax_km':base})
        for threshold in [.008,.012,.1]:
            pert=classifier(f,base,kncrit=threshold)
            pct=100*np.mean(pert!=nominal)
            displacement=kb[threshold]-cb
            rows.append({'planet':planet,'baseline_zmax_km':base,'Kncrit':threshold,'Kn_cross_shift_km':displacement,
                         'area_changed_percent':pct,'approx_stripe_area_pct':100*min(abs(displacement),base)/base,
                         'slip_envelope_not_no_slip':threshold==.1})
        for window_end in [base,base*2]:
            nom2=classifier(f,window_end);t2=classifier(f,window_end,kncrit=.008)
            window.append({'planet':planet,'window_km':window_end,'area_changed_pct':float(100*np.mean(nom2!=t2)),
                           'physical_boundary_shift_km':kb[.008]-cb})
        for retr in [1e5,4e5,5e5,6e5,1e6,1e7]:
            cat=classifier(f,base,retr=retr)
            bl.append({'planet':planet,'Re_transition':int(retr),
                       'broader_BL_fraction_pct':float(100*np.mean(cat==2)),
                       'changed_from_nominal_percent':float(100*np.mean(nominal!=cat))})
    for name,dat in [('boundary_sensitivity.csv',rows),('window_dependence.csv',window),('transition_threshold_scenarios.csv',bl),('nominal_one_at_a_time_sensitivity.csv',oat),('physical_atmosphere_summary.csv',summary)]:
        with (out/name).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(dat[0]));w.writeheader();w.writerows(dat)
    (out/'robustness_audit.json').write_text(json.dumps({'method':{'altitude_spacing_km':.1,'velocity_count':1201,'velocity_range_ms':[.1,1000],
                   'length_m':1,'baseline_criteria':{'Kn':KNC,'Mach':MAC,'Re_min':RE_MIN,'Re_transition':RE_T},
                   'interpolation':'PCHIP on log of strictly positive atmospheric properties',
                   'area_fraction_note':'Equal weight for each z and log U grid cell; not a physical probability or invariant robustness metric'},
              'planets':summary,'Kn_sensitivity':rows,'windows':window,'Re_thresholds':bl,'nominal_one_at_a_time':oat},indent=2))
    print('--- summary ---')
    for x in summary:print(x['planet'], 'Kn=0.01',round(x['nominal_Kn_cross_km'],3),'z/H0',round(x['z_over_surface_H'],3), 'Kn=0.1',round(x['kn_01_slip_cross_km'],3))
    print('--- domain dependence ---')
    for x in window:print(x)
    print('--- maximum narrow-perturbation map change ---')
    for planet in ZMAX:
        r=[x for x in oat if x['planet']==planet]; mx=max(r,key=lambda x:x['classification_cells_changed_pct']);print(mx)
    print('--- BL fractions ---')
    for x in bl:
        if x['Re_transition'] in (int(5e5),int(1e7)):print(x)

if __name__=='__main__':main()
