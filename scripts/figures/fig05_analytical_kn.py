"""Section 6.4 comparison: surface-calibrated isothermal vs NASA GRAM-derived Kn predictions.

Reproducible calculation using the same molecular mean-free-path dataset as report Figs. 1-3.
Inputs: earth3.csv, mars3.csv, titan3.csv (copy alongside this script or use --data-dir)
Outputs: Fig4_Analytical_vs_GRAM_Kn.pdf/png/svg, analytical_comparison_results.json

Method:
 H0 = (z1-z0)/ln[p(z0)/p(z1)] (km), z0=0 km, z1=1 km.
 lambda_iso(z)=lambda0 exp(z/H0) assumes isothermal hydrostatic ideal gas,
 constant g, mean molecular mass and viscosity.
 Kn(z)=lambda(z)/L, L=1 m.
 Solve crossing of Kn=0.01 using PCHIP on ln(lambda) at native 1-km nodes.
 PCHIP plotting is interpolation, not additional atmospheric measurements.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

KN_CRIT=0.01
L_M=1.0
PLANETS=('earth','mars','titan')
VIEW={'earth':(0,120),'mars':(0,110),'titan':(0,530)}

def get_data_path(folder,p):
    path=folder/f'{p}3.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run --data-dir with the location of the three GRAM-derived CSVs.')
    return path

def load_and_compute(p,folder):
    df=pd.read_csv(get_data_path(folder,p))
    req=['Height_km','Pressure_Pa','MeanFreePath_ViscosityBased_m']
    for c in req:
        if c not in df:raise ValueError(f'{p}: {c} column missing')
    z=df.Height_km.to_numpy(dtype=float)
    pressure=df.Pressure_Pa.to_numpy(dtype=float)
    mfp=df.MeanFreePath_ViscosityBased_m.to_numpy(dtype=float)
    assert len(z)==2201 and np.allclose(z,np.arange(2201))
    assert np.all(np.isfinite(pressure)) and np.all(pressure>0)
    assert np.all(np.isfinite(mfp)) and np.all(mfp>0)
    H0=(z[1]-z[0])/np.log(pressure[0]/pressure[1])
    lam0=mfp[0]
    ziso=H0*np.log(KN_CRIT*L_M/lam0)
    f=PchipInterpolator(z,np.log(mfp),extrapolate=False)
    target=np.log(KN_CRIT*L_M)
    idx=np.where((np.log(mfp[:-1])-target)*(np.log(mfp[1:])-target)<=0)[0]
    if len(idx)!=1:raise ValueError(f'{p} expected exactly one Kn crossing, got {len(idx)}')
    zreal=float(brentq(lambda x:float(f(x)-target),z[idx[0]],z[idx[0]+1],xtol=1e-12))
    Ze=np.linspace(*VIEW[p],1500)
    real=np.exp(f(Ze))/L_M
    approx=(lam0/L_M)*np.exp(Ze/H0)
    # Numeric cross-check: equal to reference values to 0.02 km
    ref=dict(earth=(100.96,85.74),mars=(81.26,75.17),titan=(275.31,470.21))[p]
    assert abs(ziso-ref[0])<.02 and abs(zreal-ref[1])<.02, (p,ziso,zreal)
    return dict(planet=p, surface_scale_height_km=float(H0),surface_mean_free_path_m=float(lam0),
                isothermal_boundary_km=float(ziso), gram_boundary_km=float(zreal),
                isothermal_minus_gram_km=float(ziso-zreal),z_plot=Ze,kn_actual=real,kn_analytic=approx)

def plot(records,folder):
    fig,axes=plt.subplots(3,1,figsize=(7.3,7.55),layout='constrained')
    for i,p in enumerate(PLANETS):
        r=records[p];ax=axes[i]
        (actual,)=ax.plot(r['z_plot'],r['kn_actual'],lw=2.25,label='GRAM-derived',zorder=4)
        (iso,)=ax.plot(r['z_plot'],r['kn_analytic'],lw=2.05,ls='--',label='Surface-calibrated isothermal',zorder=3)
        ax.axhline(KN_CRIT,color='0.37',lw=1.0,ls=':',zorder=2)
        ax.scatter([r['gram_boundary_km']],[KN_CRIT],marker='o',color=actual.get_color(),edgecolor='white',linewidth=.6,s=43,zorder=6)
        ax.scatter([r['isothermal_boundary_km']],[KN_CRIT],marker='D',color=iso.get_color(),edgecolor='white',linewidth=.6,s=39,zorder=6)
        ax.set_yscale('log')
        ax.set_xlim(*VIEW[p]);ax.grid(axis='y',which='major',alpha=.19,lw=.7)
        if p=='earth':ax.set_ylim(1e-8,10)
        elif p=='mars':ax.set_ylim(1e-6,10)
        else:ax.set_ylim(1e-9,1e5)
        ax.set_ylabel(r'Knudsen number, $\mathrm{Kn}$',fontsize=9.5)
        ax.set_xlabel('Altitude above local surface (km)',fontsize=9.3)
        ax.set_title(f'({chr(97+i)}) {p.capitalize()}',loc='left',weight='semibold',fontsize=11.2,pad=5)
        sign=r['isothermal_minus_gram_km']
        if p=='titan':
            line1=f'Isothermal: {r["isothermal_boundary_km"]:.1f} km   |   GRAM: {r["gram_boundary_km"]:.1f} km'
            xy=(.02,.96)
        else:
            line1=f'Isothermal: {r["isothermal_boundary_km"]:.1f} km   |   GRAM: {r["gram_boundary_km"]:.1f} km'
            xy=(.02,.96)
        ax.text(*xy,line1,transform=ax.transAxes,fontsize=8.7,ha='left',va='top',
                bbox=dict(boxstyle='round,pad=0.28',facecolor='white',alpha=.90,edgecolor='0.84'))
        ax.tick_params(labelsize=8.5)
        for sp in ax.spines.values():sp.set_color('0.57')
        ax.xaxis.set_major_locator(MultipleLocator(20 if p!='titan' else 100))
    axes[0].legend(handles=[actual,iso],loc='lower right',fontsize=8.0,frameon=True,framealpha=.92)
    out=folder/'Fig4_Analytical_vs_GRAM_Kn'
    for ext in ('pdf','png','svg'):
        fig.savefig(out.with_suffix('.'+ext),dpi=320 if ext=='png' else None,facecolor='white',bbox_inches='tight',pad_inches=.20)
    plt.close(fig)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data={p:load_and_compute(p,args.data_dir) for p in PLANETS}
    plot(data,args.output_dir)
    json_ready={p:{k:v for k,v in rec.items() if k not in ('planet','z_plot','kn_actual','kn_analytic')} for p,rec in data.items()}
    json_ready['method']={'reference_altitudes_km':[0,1],'Kn_critical':KN_CRIT,'L_m':L_M,
              'isothermal_model':'lambda0 * exp(z / H0)',
              'H0':'1 km / ln[p(0 km) / p(1 km)]',
              'crossover_method':'PCHIP interpolated log(lambda) crossing of log(0.01 m)'}
    (args.output_dir/'analytical_comparison_results.json').write_text(json.dumps(json_ready,indent=2)+'\n')
    for p in PLANETS:
        r=data[p]
        print(p, 'H0={:.5f} km, lam0={:.7g} m, iso={:.5f} km, GRAM={:.5f} km, delta={:+.5f} km'.format(r['surface_scale_height_km'],r['surface_mean_free_path_m'],r['isothermal_boundary_km'],r['gram_boundary_km'],r['isothermal_minus_gram_km']))

if __name__=='__main__':main()
