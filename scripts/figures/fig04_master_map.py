"""Section 6.3: reproducible Reynolds--Mach master maps with planetary trajectories.

Uses the user-generated NASA GRAM-derived reference profiles (earth3.csv,
 mars3.csv, titan3.csv). The coloured master-map uses a REFERENCE gamma=1.4
 and exact ideal-gas identity for Kn; the overlay trajectories independently
 use local NASA GRAM speed of sound, density, viscosity and mean-free-path.
 Original 1-km atmospheric profiles are interpolated with log-space PCHIP.
 All plotted points obey L=1 m. No neural or AI image generation used.

Run: python generate_master_map.py --data-dir /path/with/csvs --out-dir /output
Requires: numpy, pandas, scipy, matplotlib.
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

PLANETS=('earth','mars','titan')
SRC=['Height_km','Density_kgm3','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s','SpeedOfSound_ms','MeanFreePath_ViscosityBased_m','SpecificHeatRatio']
CATEGORIES=['Incompressible NS','Blasius','Boundary-layer','Compressible NS','Rarefied / slip']
CATS=['#447B9F','#30A48F','#E9AE4A','#D66D63','#8E79A7']
LINE={'earth':'#182f43','mars':'#a3272d','titan':'#174b1f'}
L=1.0
KNC=0.01; MAC=0.3; REMIN=1e4; RETRANS=5e5; GAMMA_REF=1.4
U_VALUES=(10.,100.)


def load_csv(data_dir, name):
    df=pd.read_csv(data_dir/f'{name}3.csv')
    for c in SRC:
        if c not in df: raise ValueError(f'Missing {c} from {name}3.csv')
    z=df['Height_km'].to_numpy(float)
    if not (z[0]==0 and len(z)==2201 and np.allclose(np.diff(z),1)):
        raise ValueError(f'{name} must have 1km samples from 0 to 2200km')
    d={'z_original':z,'raw':df}
    for c in SRC[1:]:
        a=df[c].to_numpy(float)
        if not np.all(np.isfinite(a)) or np.any(a<=0):raise ValueError(f'invalid {c} in {name}')
        d[c]=PchipInterpolator(z,np.log(a),extrapolate=False)
    return d


def inter(d,c,z):
    return np.exp(d[c](z))


def find_threshold(d,key,target):
    a=d['raw'][key].to_numpy(float)
    z=d['z_original']
    cs=np.flatnonzero((a[:-1]-target)*(a[1:]-target)<=0)
    if len(cs)!=1: raise ValueError(f'Expected one crossing {key}: got {len(cs)}')
    i=int(cs[0]); return float(brentq(lambda q:float(inter(d,key,q)-target),z[i],z[i+1]))


def category(Re,Ma,Kn):
    # original 5-branch classifier; equality goes to more general model
    return np.where(Kn>=KNC,4,
           np.where(Ma>=MAC,3,
           np.where(Re<REMIN,0,
           np.where(Re<RETRANS,1,2)))).astype(np.uint8)


def build_meta(d,u,zk):
    z=np.linspace(0,zk, max(2001,int(zk/.05)+1))
    rho=inter(d,'Density_kgm3',z); a=inter(d,'SpeedOfSound_ms',z)
    mu=inter(d,'DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',z)
    lam=inter(d,'MeanFreePath_ViscosityBased_m',z)
    gam=inter(d,'SpecificHeatRatio',z)
    Re=rho*u*L/mu; Ma=u/a; Kn=lam/L
    cref=np.sqrt(np.pi*GAMMA_REF/2)*Ma/Re
    classes=category(Re,Ma,Kn)
    refclasses=category(Re,Ma,cref)
    # crossing altitudes for each relevant threshold, where exists in [0,zk]
    crossings={}
    for key, v in [('Re_5e5',Re-RETRANS),('Re_1e4',Re-REMIN),('Ma_0p3',Ma-MAC)]:
        ix=np.flatnonzero(v[:-1]*v[1:]<0)
        if len(ix):
            cross=[]
            for ii in ix:
                # dense linear bracketing; enough to report 0.1km
                cross.append(float(z[ii]-(z[ii+1]-z[ii])*v[ii]/(v[ii+1]-v[ii])))
            crossings[key]=cross
    return {'z':z,'Re':Re,'Ma':Ma,'Kn':Kn,'gam':gam,'C':Kn/cref,
            'nominal_class':classes,'ref_class':refclasses,
            'summary':{
                'U_ms':u,'z_end_Kn_0p01_km':zk,
                'surface_Re':float(Re[0]),'surface_Ma':float(Ma[0]),
                'Re_at_Kn_boundary':float(Re[-1]),'Ma_at_Kn_boundary':float(Ma[-1]),
                'surface_class':CATEGORIES[int(classes[0])],
                'mastermap_class_disagreement_pct':float(100*np.mean(classes!=refclasses)),
                'max_gamma_in_plot':float(np.max(gam)), 'min_gamma_in_plot':float(np.min(gam)),
                'max_Kn_identity_ratio_deviation_pct':float(100*np.max(np.abs(Kn/cref-1))),
                'crossings_km':crossings
            }}


def figure(tracks,out):
    fig, axes=plt.subplots(2,1,figsize=(7.9,8.35),sharex=True,sharey=True)
    fig.patch.set_facecolor('white')
    xlim=(1e-1,2e8); ylim=(1.7e-2,1.2)
    # The master regions are vector-native analytical polygons, NOT raster pixels.
    # Only the Kn boundary uses the reference gamma=1.4 approximation.
    ma_curve=np.geomspace(*ylim,500)
    acoef=np.sqrt(np.pi*GAMMA_REF/2)/KNC
    linehalo=[pe.Stroke(linewidth=3.4,foreground='white',alpha=.92),pe.Normal()]
    for ax, u, title in zip(axes,U_VALUES,['(a) $U=10$ m s$^{-1}$','(b) $U=100$ m s$^{-1}$']):
        ax.fill_between([xlim[0],REMIN],[ylim[0]]*2,[MAC]*2,
                        color=CATS[0],alpha=.62,zorder=0,linewidth=0)
        ax.fill_between([REMIN,RETRANS],[ylim[0]]*2,[MAC]*2,
                        color=CATS[1],alpha=.62,zorder=0,linewidth=0)
        ax.fill_between([RETRANS,xlim[1]],[ylim[0]]*2,[MAC]*2,
                        color=CATS[2],alpha=.62,zorder=0,linewidth=0)
        ax.fill_between([xlim[0],xlim[1]],[MAC]*2,[ylim[1]]*2,
                        color=CATS[3],alpha=.62,zorder=0,linewidth=0)
        ax.fill_betweenx(ma_curve,np.full_like(ma_curve,xlim[0]),acoef*ma_curve,
                         color=CATS[4],alpha=.62,zorder=1,linewidth=0)
        # Reference-gamma boundary equations, drawn as continuous vector lines.
        ax.plot(acoef*ma_curve,ma_curve,color='#2d2c33',lw=1.15,ls='--',zorder=2)
        ax.plot([acoef*MAC,xlim[1]],[MAC,MAC],color='#31323b',lw=.95,ls=':',zorder=1)
        for r in [REMIN,RETRANS]:
            ax.plot([r,r],[ylim[0],MAC],color='#55575f',lw=.8,ls=':',zorder=1)
        for p in PLANETS:
            tr=tracks[p][u]
            ax.plot(tr['Re'],tr['Ma'],color=LINE[p],lw=2.2,zorder=5,path_effects=linehalo)
            # surface circle; crossing diamond
            ax.scatter([tr['Re'][0]],[tr['Ma'][0]],s=58,marker='o',color=LINE[p],edgecolor='white',linewidth=.9,zorder=8)
            ax.scatter([tr['Re'][-1]],[tr['Ma'][-1]],s=64,marker='D',color=LINE[p],edgecolor='white',linewidth=.9,zorder=8)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(*xlim);ax.set_ylim(*ylim)
        ax.set_title(title,fontsize=12.5,loc='left',pad=7,weight='semibold')
        ax.set_ylabel(r'Mach number, $\mathrm{Ma}$',fontsize=11.5)
        ax.tick_params(labelsize=10)
        ax.set_yticks([.02,.03,.05,.1,.3,1]);ax.set_yticklabels(['0.02','0.03','0.05','0.1','0.3','1'])
        ax.set_xticks([1e-1,1e1,1e3,1e5,1e7]);ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*.1));ax.xaxis.set_minor_formatter(NullFormatter())
        for s in ax.spines.values():s.set_color('#575c66')
        ax.grid(False)
    axes[1].set_xlabel(r'Reynolds number, $\mathrm{Re}_L$',fontsize=11.5,labelpad=8)
    leg_category=[Patch(color=CATS[i],label=CATEGORIES[i]) for i in range(5)]
    # panel titles identify velocities, so legend only needs planet colours and endpoint markers
    leg_planet=[Line2D([],[],color=LINE[p],lw=2.5,label=p.capitalize()) for p in PLANETS]
    leg_points=[Line2D([],[],marker='o',color='white',markerfacecolor='#333333',markersize=6,label='Surface'),
                Line2D([],[],marker='D',color='white',markerfacecolor='#333333',markersize=6,label=r'$\mathrm{Kn}=0.01$')]
    fig.legend(handles=leg_planet+leg_points,loc='lower center',ncol=5,bbox_to_anchor=(.5,.115),frameon=False,fontsize=9.8,handlelength=1.8,columnspacing=1.8)
    fig.legend(handles=leg_category,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),frameon=False,fontsize=9.6,handlelength=1.05,columnspacing=1.6,labelspacing=.65)
    fig.subplots_adjust(left=.11,right=.976,top=.955,bottom=.24,hspace=.26)
    base=out/'Fig3_Dimensionless_Master_Map_Planetary_Trajectories'
    for suffix in ['.png','.pdf','.svg']:
        fig.savefig(base.with_suffix(suffix),dpi=310 if suffix=='.png' else None,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    tracks={}; details={}
    for p in PLANETS:
        d=load_csv(args.data_dir,p)
        zk=find_threshold(d,'MeanFreePath_ViscosityBased_m',KNC*L)
        tracks[p]={}; details[p]={}
        for u in U_VALUES:
            tr=build_meta(d,u,zk)
            tracks[p][u]=tr
            details[p][f'{u:g}_ms']=tr['summary']
            # exactly reproduce input grid classifications: validation that interpolation
            # interpolated altitude reproduces original atmospheric states at integer z
            z_orig=d['z_original']; val=z_orig<=zk
            z_i=z_orig[val]
            for c in ['Density_kgm3','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s','SpeedOfSound_ms','MeanFreePath_ViscosityBased_m']:
                original=d['raw'].loc[val,c].to_numpy(float)
                if not np.allclose(inter(d,c,z_i),original,rtol=2e-12,atol=0):
                    raise AssertionError(f'{p}: {c} interpolator fails at original samples')
        details[p]['kn_boundary_altitude_km']=zk
    figure(tracks,args.out_dir)
    summary={'source':'NASA GRAM-derived profiles (not newly rerun GRAM), 1km original altitude resolution',
             'master_map':'reference gamma=1.4 and idealized Kn=sqrt(pi gamma/2)*Ma/Re; only approximate for actual planetary datasets',
             'trajectories':'independent Re=rho U L / mu, Ma=U/a_GRAM, Kn=lambda_derived/L; log-PCHIP at 0.05km or finer',
             'characteristic_length_m':L,'velocity_cases_m_s':U_VALUES,
             'thresholds':{'Kn':KNC,'Ma':MAC,'Re_min':REMIN,'Re_transition':RETRANS},
             'plotted_altitude_ranges':'0 km to first Kn=0.01 crossing for each planet',
             'planets':details}
    (args.out_dir/'master_map_numerical_results.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
