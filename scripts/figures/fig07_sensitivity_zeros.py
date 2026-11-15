"""Reconstruct Figure 5 with explicit zero bars and verify isothermal Eq (48).
Run from any directory: python refine_fig5_and_test_eq48.py
Original manuscript figures and CSVs remain unchanged.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
root = HERE  # Portable staged-data directory
rows = list(csv.DictReader((HERE/'nominal_one_at_a_time_sensitivity.csv').open()))
planets = ['Earth','Mars','Titan']
parameters = [('Kn','Kn threshold'),('Ma','Mach threshold'),('epsilon','BL thinness'),('Re_tr','Transition Re')]
colors = ['#41769E','#D68B55']
fig,axes = plt.subplots(1,3,figsize=(11.5,3.6),sharey=True)
for ax,planet in zip(axes,planets):
    x=np.arange(4)
    for j in range(2):
        y=[]
        for code,label in parameters:
            recs=[r for r in rows if r['planet']==planet and r['parameter']==code]
            recs.sort(key=lambda r:float(r['value']))
            assert len(recs)==2, (planet,code)
            y.append(float(recs[j]['classification_cells_changed_pct']))
        ax.bar(x+(j-.5)*.34, y, width=.33, color=colors[j], label=['Lower perturbation','Upper perturbation'][j])
    ax.set_title(planet, fontsize=12, weight='bold')
    ax.set_xticks(x, [s for _,s in parameters], rotation=24, ha='right', fontsize=8.5)
    ax.grid(axis='y', alpha=.22)
    ax.set_axisbelow(True)
    ax.set_ylim(0,2.8)
    if planet=='Mars':
        ax.annotate('0.00 / 0.00',xy=(3,0),xytext=(3,.20),ha='center',fontsize=9,weight='bold',color='#3C4248',arrowprops={'arrowstyle':'-', 'lw':0.6,'color':'#666'})
axes[0].set_ylabel('Grid cells changing classification (%)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.06),frameon=False)
fig.subplots_adjust(top=.79,bottom=.28,left=.075,right=.99,wspace=.15)
for ext in ('pdf','svg','png'):
    fig.savefig(HERE/f'Fig5_Sensitivity_Explicit_Zero_Bars.{ext}', dpi=250, bbox_inches='tight')
plt.close(fig)

results=[]
for planet,filename in [('Earth','earth3.csv'),('Mars','mars3(1).csv'),('Titan','titan3.csv')]:
    df=pd.read_csv(root/filename)
    z=df['Height_km'].to_numpy(float)
    rho=df['Density_kgm3'].to_numpy(float)
    mu=df['DynamicViscosity_ChapmanEnskog_Wilke_Pa_s'].to_numpy(float)
    a=df['SpeedOfSound_ms'].to_numpy(float)
    p=df['Pressure_Pa'].to_numpy(float)
    rmax=0.3*rho*a/mu # L=1m
    H0=-1/np.log(p[1]/p[0])
    ziso=H0*np.log(rmax[0]/5e5)
    cross=np.flatnonzero(np.diff(np.sign(rmax-5e5)))
    interp=PchipInterpolator(z,np.log(rmax))
    zg=[brentq(lambda zz:interp(zz)-np.log(5e5),float(z[k]),float(z[k+1])) for k in cross]
    results.append({'planet':planet,'reference_ReMax_at_z0':rmax[0],'H0_km':H0,'isothermal_eq48_intersection_km':ziso,'GRAM_intersection_km':zg[0] if zg else None,'in_plotted_domain':bool(zg and zg[0] <= (500 if planet=='Titan' else 100))})
pd.DataFrame(results).to_csv(HERE/'eq48_reynolds_mach_intersection.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
