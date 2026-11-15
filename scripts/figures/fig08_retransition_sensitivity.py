"""Reproduce non-redundant Fig 6 from original GRAM-derived CSVs.

Computes fraction of map classified as broader BL for Re_transition in [1e5,1e7].
Grid: z 0.1 km, Earth/Mars 0-100, Titan 0-500; U log spaced, 1201
points in 0.1 to 1000 m/s. L=1m, Ma<0.3, Kn<0.01, Re>=Re_tr.
The area fraction is a visual-map fraction, not an uncertainty probability.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN=Path(__file__).resolve().parent  # Portable staged-data directory
OUT=Path(__file__).resolve().parent
FILES={'Earth':'earth3.csv','Mars':'mars3.csv','Titan':'titan3.csv'}
ZMAX={'Earth':100,'Mars':100,'Titan':500}
COL={'Earth':'#2776ad','Mars':'#d07843','Titan':'#29906d'}
THRESHOLDS=np.array([1e5,2e5,4e5,5e5,6e5,1e6,2e6,5e6,1e7])
U=np.geomspace(.1,1000,1201)
result={}
for planet,path in FILES.items():
    f=IN/path
    d=pd.read_csv(f)
    z=d.Height_km.to_numpy(float)
    zp=np.linspace(0,ZMAX[planet],int(ZMAX[planet]/.1)+1)
    def interp(col):
        vals=d[col].to_numpy(float)
        return np.exp(PchipInterpolator(z,np.log(vals))(zp))
    rho=interp('Density_kgm3')
    mu=interp('DynamicViscosity_ChapmanEnskog_Wilke_Pa_s')
    a=interp('SpeedOfSound_ms')
    lam=interp('MeanFreePath_ViscosityBased_m')
    # Instead of categorical arrays for each threshold, calculate the
    # broader-BL classification directly as a boolean mask.
    Re=rho[:,None]*U[None,:]/mu[:,None]
    Ma=U[None,:]/a[:,None]
    valid=(Ma < .3) & (lam[:,None]<.01)
    fractions=[float(100*np.count_nonzero(valid & (Re>=threshold))/Re.size) for threshold in THRESHOLDS]
    result[planet]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'Re_thresholds':THRESHOLDS.tolist(),'BL_fraction_pct':fractions}
    print(planet,'at 5e5',fractions[3],'at 1e7',fractions[-1])
fig,ax=plt.subplots(figsize=(7.3,4.3))
for planet in FILES:
    v=result[planet]
    ax.plot(THRESHOLDS,v['BL_fraction_pct'],marker='o',markersize=4.6,lw=2,label=planet,color=COL[planet])
ax.axvline(5e5,lw=1,ls='--',color='#767b86')
ax.annotate('Nominal threshold',xy=(5e5,ax.get_ylim()[1]*.85),xytext=(7e5,ax.get_ylim()[1]*.9),fontsize=9,color='#58616b',arrowprops=dict(arrowstyle='-',color='#767b86',lw=.7))
ax.set_xscale('log')
ax.set_xlim(1e5,1e7)
ax.set_ylim(bottom=0)
ax.set_xlabel(r'Assumed laminar-transition Reynolds number, $\mathrm{Re}_{L,\mathrm{crit}}$',fontsize=11)
ax.set_ylabel('Broader BL category (% of plotted grid)',fontsize=11)
ax.grid(axis='y',alpha=.2)
ax.legend(loc='upper right',ncol=3,frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.subplots_adjust(left=.13,right=.97,top=.95,bottom=.18)
base=OUT/'Fig6_ReTransition_Sensitivity_Standalone'
for ext in ('pdf','png','svg'):
    fig.savefig(base.with_suffix('.'+ext),dpi=280 if ext=='png' else None,facecolor='white')
plt.close(fig)
(OUT/'retransition_figure_results.json').write_text(json.dumps({'method':{'z_step_km':.1,'U_n':1201,'U_range_ms':[.1,1000],'L_m':1,'thresholds':{'Kn':.01,'Ma':.3},'area_dependency':'Depends on plotted altitude and log velocity window'},'planets':result},indent=2)+'\n')
