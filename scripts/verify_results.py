"""Independent numerical and file integrity checks for processed profiles."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np,pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
BASE=Path(__file__).resolve().parents[1]
SHA={'earth':'f3e887d7f4efe4c6d6881600d9a58f7588298cee05563f9350748655f7131465',
'mars':'cc16136c0ab502f5ccebe7f5cc0fd67230a1b18bef4d22a40a148f3681db9618',
'titan':'72894c407099d78eaac4aec8a04a8f15bc6183ead6c9643b9a9712044a561eed'}
EXPECTED={'earth':85.7417827,'mars':75.1652216,'titan':470.20841}
results={}
for planet in SHA:
    path=BASE/'data'/f'{planet}3.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA[planet],f'{planet} data SHA mismatch'
    d=pd.read_csv(path); assert len(d)==2201 and np.array_equal(d.Height_km.to_numpy(),np.arange(2201))
    for col in ['Temperature_K','Pressure_Pa','Density_kgm3','SpeedOfSound_ms','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s','MeanFreePath_ViscosityBased_m']:
        assert np.all(np.isfinite(d[col])) and np.all(d[col]>0),f'Invalid {planet} {col}'
    z=d.Height_km.to_numpy(float); lam=d.MeanFreePath_ViscosityBased_m.to_numpy(float)
    ks=np.log(lam/.01); idx=np.where(np.diff(np.sign(ks)))[0];assert len(idx)>=1
    f=PchipInterpolator(z,np.log(lam)); crossing=brentq(lambda x:float(f(x)-math.log(.01)),z[idx[0]],z[idx[0]+1])
    assert abs(crossing-EXPECTED[planet])<.01,(planet,crossing)
    rho=d.Density_kgm3.iloc[0];mu=d.DynamicViscosity_ChapmanEnskog_Wilke_Pa_s.iloc[0];a=d.SpeedOfSound_ms.iloc[0]
    Lcrit=5e5*mu/(.3*rho*a)
    results[planet]={'rows':len(d),'sha256':SHA[planet],'kn01_crossing_L1_km':crossing,'reference_surface_Lcrit_m':Lcrit}
assert abs(results['mars']['reference_surface_Lcrit_m']-4.32449)<.01
cases=pd.read_csv(BASE/'data/sampling/individual_48profile_Kn_crossings.csv');stats=json.loads((BASE/'data/sampling/individual_profile_summary.json').read_text())
assert len(cases)==96
for s in stats:
 d=cases[cases.planet==s['planet']];assert len(d)==48
 w=d.weight.to_numpy(float);w/=w.sum();v=d.continuum_crossing_km.to_numpy(float)
 assert abs(w@v-s['weighted_mean_km'])<1e-8
 assert abs(np.sqrt(w@((v-w@v)**2))-s['weighted_std_km'])<1e-8
for i in range(1,10):
 for ext in ('pdf','png','svg'):
  f=BASE/'figures'/f'fig{i:02d}.{ext}';assert f.is_file() and f.stat().st_size>1024,f
(BASE/'results'/'verification.json').write_text(json.dumps({'profile_checks':results,'individual_cases':len(cases),'figure_count':9},indent=2)+'\n')
print('PASS: 3 data hashes and 2,201-row profiles, crossings, Mars Lcrit, 96 case statistics, 9 x 3 figure files')
