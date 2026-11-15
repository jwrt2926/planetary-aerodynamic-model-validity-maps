"""Length-scale analysis using NASA GRAM-derived atmospheric reference profiles.

Recreates report Fig. 2 and a supplementary Mars-regime comparison.
Requirements: numpy, scipy, pandas, matplotlib.
Run: python generate_length_scale_figures.py
Input CSVs may be in this folder or its parent; override with --data-dir.

Physical definitions (kept consistent with the nominal L=1 m maps):
  Re_L = rho(z) * U * L / mu(z)
  Ma = U / a_GRAM(z)
  Kn = lambda_viscosity(z) / L
  Kncrit = 0.01, Macrit = 0.3, Re_min=1e4, Re_transition=5e5
The 1km input samples are the resolution of the atmospheric data.
Monotonic shape-preserving PCHIP interpolates log(positive properties);
interpolated plotting points are NOT independent atmospheric observations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

KN_CRIT = 0.01
MA_CRIT = 0.3
RE_MIN = 1.0e4
RE_TRANS = 5.0e5
LENGTHS = np.array([0.01, 0.1, 1.0, 10.0])
PLANETS = ('earth','mars','titan')
DISPLAY_ZMAX = {'earth':100, 'mars':100, 'titan':500}
CATEGORY_NAMES = ['Incompressible NS','Blasius','Boundary-layer','Compressible NS','Rarefied / slip']
CATEGORY_COLORS = ['#447B9F','#30A48F','#E9AE4A','#D66D63','#8E79A7']


def data_file(directory:Path, planet:str):
    f = directory/f'{planet}3.csv'
    if not f.exists():
        raise FileNotFoundError(f'Cannot find {f}. Place earth3.csv, mars3.csv, titan3.csv in --data-dir.')
    return f


def load_profile(directory:Path, p:str):
    df=pd.read_csv(data_file(directory,p))
    req=['Height_km','Density_kgm3','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',
         'SpeedOfSound_ms','MeanFreePath_ViscosityBased_m']
    for c in req:
        if c not in df: raise ValueError(f'Missing required column {c} for {p}')
    z=df['Height_km'].to_numpy(dtype=float)
    if len(z)!=2201 or z[0]!=0 or z[-1]!=2200 or not np.allclose(np.diff(z),1.0):
        raise ValueError(f'Unexpected altitude sampling for {p}')
    profile={'z':z}
    for key in req[1:]:
        val=df[key].to_numpy(dtype=float)
        if not np.all(np.isfinite(val)) or np.any(val<=0):
            raise ValueError(f'Nonpositive/invalid {key} for {p}')
        profile[key]=val
        profile[key+'_loginterpolator']=PchipInterpolator(z,np.log(val),extrapolate=False)
    return profile


def interp_positive(profile,key,z):
    return np.exp(profile[key+'_loginterpolator'](z))


def kn_crossing(profile,L):
    z=profile['z']; log_lam=np.log(profile['MeanFreePath_ViscosityBased_m']); target=np.log(KN_CRIT*L)
    cross=np.flatnonzero((log_lam[:-1]-target)*(log_lam[1:]-target)<=0)
    if len(cross)!=1:
        raise ValueError(f'Expected exactly one Kn crossing for L={L}, found {len(cross)}')
    i=int(cross[0])
    return float(brentq(lambda zz:float(profile['MeanFreePath_ViscosityBased_m_loginterpolator'](zz)-target),
                        z[i],z[i+1],xtol=1e-11))


def re_limit(profile,z=None):
    if z is None:
        rho=profile['Density_kgm3']; a=profile['SpeedOfSound_ms']; mu=profile['DynamicViscosity_ChapmanEnskog_Wilke_Pa_s']
    else:
        rho=interp_positive(profile,'Density_kgm3',z)
        a=interp_positive(profile,'SpeedOfSound_ms',z)
        mu=interp_positive(profile,'DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',z)
    return MA_CRIT*rho*a/mu


def make_fig2(profiles,output,values):
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    planet_colors=dict(zip(PLANETS,colors[:3]))
    fig, axes=plt.subplots(2,2,figsize=(8.27,9.45))
    fig.patch.set_facecolor('white')
    ax=axes[0,0]
    Lline=np.geomspace(0.0075,11.5,450)
    for p in PLANETS:
        coeff=values[p]['surface_max_lowmach_Re_factor_per_m']
        ax.plot(Lline,coeff*Lline,lw=2.25,label=p.capitalize(),color=planet_colors[p])
        Lc=values[p]['minimum_L_for_general_BL_m']
        ax.scatter([Lc],[RE_TRANS],s=52,zorder=5,color=planet_colors[p],edgecolor='white',linewidth=.8)
    ax.axhline(RE_TRANS,color='.22',ls='--',lw=1.1)
    ax.text(11.2,RE_TRANS*1.17,r'$\mathrm{Re}_{\mathrm{crit}}=5\times10^5$',ha='right',va='bottom',fontsize=9)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.0075,12)
    ax.set_ylim(4e2,1e9)
    ax.set_xlabel(r'Characteristic length, $L$ (m)',fontsize=10.5)
    ax.set_ylabel(r'Limiting low-Mach $\mathrm{Re}_{L}$',fontsize=10.5)
    ax.set_title('(a) Onset of broader boundary-layer branch',fontsize=11.4,pad=10,loc='left')
    ax.legend(loc='upper left',fontsize=8.5,frameon=False)
    ax.grid(which='major',alpha=.16)
    ax.text(.04,.045,'Critical length:\nTitan 0.00975 m\nEarth 0.0738 m\nMars 4.324 m',
            transform=ax.transAxes,fontsize=8.4,va='bottom',ha='left',
            bbox={'facecolor':'white','alpha':.92,'edgecolor':'.8','boxstyle':'round,pad=.45'})
    limits={'earth':(44,106),'mars':(26,101),'titan':(208,615)}
    letter={'earth':'b','mars':'c','titan':'d'}
    for k,p in enumerate(PLANETS):
        ar=axes.ravel()[k+1]
        la=np.geomspace(0.01,10,250)
        zks=np.array([kn_crossing(profiles[p],float(L)) for L in la])
        exact=values[p]['kn_boundary_altitude_km']
        ar.plot(la,zks,lw=2.3,color=planet_colors[p])
        ar.scatter(LENGTHS,[exact[f'{L:g}'] for L in LENGTHS],s=34,color=planet_colors[p],
                   edgecolor='white',lw=.7,zorder=4)
        ar.axvline(1.0,color='.55',ls=':',lw=1)
        ar.set_xscale('log');ar.set_xlim(.009,11)
        ar.set_ylim(*limits[p]);ar.grid(which='major',alpha=.14)
        ar.set_xlabel(r'Characteristic length, $L$ (m)',fontsize=10.5)
        ar.set_ylabel(r'$\mathrm{Kn}=0.01$ altitude (km)',fontsize=10.2)
        ar.set_title(f'({letter[p]}) {p.capitalize()}: rarefaction boundary',fontsize=11.4,pad=10,loc='left')
        ar.text(.035,.92,rf'$L=1$ m: {exact["1"]:.2f} km',transform=ar.transAxes,
                ha='left',va='top',fontsize=8.8,color='.25',
                bbox={'facecolor':'white','alpha':.86,'edgecolor':'none','pad':2})
    for ar in axes.flat:
        ar.tick_params(labelsize=9.5)
        for spine in ar.spines.values():spine.set_color('.45')
    fig.subplots_adjust(left=.115,right=.967,top=.953,bottom=.095,wspace=.36,hspace=.33)
    out=output/'Fig2_Characteristic_Length_Effects'
    for suffix in ('.png','.pdf','.svg'):
        fig.savefig(out.with_suffix(suffix),dpi=310 if suffix=='.png' else None,facecolor='white')
    plt.close(fig)


def region_map_mars(prof,ax,L,zmax=100):
    z=np.linspace(0,zmax,2001)
    rho=interp_positive(prof,'Density_kgm3',z)
    mu=interp_positive(prof,'DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',z)
    a=interp_positive(prof,'SpeedOfSound_ms',z)
    zk=kn_crossing(prof,L)
    uma=MA_CRIT*a
    ure_min=RE_MIN*mu/(rho*L)
    ure_trans=RE_TRANS*mu/(rho*L)
    umin,umax=.1,1000
    cont=(z<=zk)
    ax.axhspan(zk,zmax,color=CATEGORY_COLORS[4],lw=0,zorder=0)
    def region(low,up,color):
        lo=np.maximum(low,umin)
        hi=np.minimum(up,umax)
        mask=cont&(hi>lo)
        ax.fill_betweenx(z,lo,hi,where=mask,interpolate=True,color=color,linewidth=0,zorder=1)
    region(np.full_like(z,umin),np.minimum(ure_min,uma),CATEGORY_COLORS[0])
    region(ure_min,np.minimum(ure_trans,uma),CATEGORY_COLORS[1])
    region(ure_trans,uma,CATEGORY_COLORS[2])
    region(uma,np.full_like(z,umax),CATEGORY_COLORS[3])
    ax.plot(uma[cont],z[cont],color='.24',lw=.85)
    mask=cont&(ure_min<uma)&(ure_min>=umin)&(ure_min<=umax)
    ax.plot(ure_min[mask],z[mask],color='.26',lw=.7)
    mask=cont&(ure_trans<uma)&(ure_trans>=umin)&(ure_trans<=umax)
    ax.plot(ure_trans[mask],z[mask],color='.26',lw=.7)
    ax.axhline(zk,color='.16',ls='--',lw=1.2)
    ax.set_xscale('log');ax.set_xlim(umin,umax);ax.set_ylim(0,zmax)
    ax.set_xticks([.1,1,10,100,1000]);ax.set_xticklabels([r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$'])
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*.1,numticks=100));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel(r'Freestream velocity, $U$ (m s$^{-1}$)',fontsize=10.5)
    ax.set_ylabel('Altitude (km)',fontsize=10.5)
    ax.set_title(rf'Mars, $L={L:g}$ m',fontsize=12,loc='left',pad=10)
    ax.tick_params(labelsize=10)
    ax.text(.985,zk/zmax+.018,rf'$\mathrm{{Kn}}=0.01$: {zk:.2f} km',transform=ax.transAxes,
            ha='right',va='bottom',fontsize=8.6)
    for spine in ax.spines.values():spine.set_color('.5')
    return zk


def make_mars_comparison(profiles,output,values):
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7))
    for a,L in zip(axes,[1.0,10.0]):region_map_mars(profiles['mars'],a,L)
    handles=[Patch(facecolor=CATEGORY_COLORS[i],label=CATEGORY_NAMES[i]) for i in range(5)]
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.005),fontsize=9,frameon=False,
               columnspacing=.8,handlelength=1.1)
    fig.subplots_adjust(left=.075,right=.99,top=.89,bottom=.24,wspace=.22)
    fig.text(.5,.095,'At L=10 m, the general low-Mach boundary-layer branch appears below 8.10 km',
             ha='center',fontsize=10,color='.28')
    out=output/'Supplement_Mars_L1m_vs_L10m'
    for suffix in ('.png','.pdf','.svg'):
        fig.savefig(out.with_suffix(suffix),dpi=310 if suffix=='.png' else None,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=None)
    parser.add_argument('--out-dir',type=Path,default=None)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    data_dir=args.data_dir or (here if (here/'earth3.csv').exists() else here.parent)
    out=args.out_dir or here
    out.mkdir(parents=True,exist_ok=True)
    prof={p:load_profile(data_dir,p) for p in PLANETS}
    v={'input_files':{p:f'{p}3.csv' for p in PLANETS},
       'method':'NASA GRAM-derived Earth/Mars averaged, Titan nominal; Chapman-Enskog/Wilke mu; viscosity-derived lambda. PCHIP log interpolated properties, original sampling 1km.',
       'thresholds':{'Kn':KN_CRIT,'Ma':MA_CRIT,'Re_min':RE_MIN,'Re_transition':RE_TRANS},
       'length_samples_m':LENGTHS.tolist(),'velocity_domain_ms':[0.1,1000.0],
       'planetary':{}}
    for p in PLANETS:
        d=prof[p]
        # Compare all sample points within the altitude displayed in original Fig1.
        n=DISPLAY_ZMAX[p]+1
        rr=re_limit(d)[:n]
        idx=int(np.argmax(rr))
        assert idx==0, f'Maximum limiting Reynolds factor not at the surface for {p}'
        Lcrit=RE_TRANS/rr[idx]
        crossings={f'{L:g}':kn_crossing(d,float(L)) for L in LENGTHS}
        v['planetary'][p]={
            'surface_max_lowmach_Re_factor_per_m':float(rr[idx]),
            'max_Re_position_km':float(d['z'][idx]),
            'minimum_L_for_general_BL_m':float(Lcrit),
            'kn_boundary_altitude_km':crossings,
            'max_lowmach_Re_at_L1m':float(rr[idx]),
            'max_lowmach_Re_at_L3m':float(rr[idx]*3),
            'max_lowmach_Re_at_L10m':float(rr[idx]*10)}
    if abs(v['planetary']['mars']['minimum_L_for_general_BL_m']-4.32449)>1e-4:
        raise AssertionError('Unexpected Mars length threshold')
    # Mars L=10 m: maximum z at which Ma<0.3 can support Re>=5e5.
    mars=prof['mars']; y=re_limit(mars)
    crossings=np.where(np.diff(np.sign(y[:101]-RE_TRANS/10.0)))[0]
    assert len(crossings)==1
    index=int(crossings[0]);reP=PchipInterpolator(mars['z'],y)
    zmax=brentq(lambda zz:float(reP(zz)-RE_TRANS/10.0),index,index+1,xtol=1e-11)
    v['mars_L10m_general_BL_max_altitude_km']=float(zmax)
    assert abs(zmax-8.1004)<0.025
    assert v['planetary']['mars']['max_lowmach_Re_at_L3m']<RE_TRANS
    assert v['planetary']['mars']['max_lowmach_Re_at_L10m']>RE_TRANS
    # Verify L=1m Kn crossings consistent with previous report maps.
    nominal={'earth':85.742,'mars':75.1657,'titan':470.208}
    for p,reference in nominal.items():
        delta=abs(v['planetary'][p]['kn_boundary_altitude_km']['1']-reference)
        assert delta<.01,(p,delta)
    make_fig2(prof,out,v['planetary'])
    make_mars_comparison(prof,out,v)
    (out/'length_scale_results.json').write_text(json.dumps(v,indent=2))
    print(json.dumps(v,indent=2))
    print('\nGenerated:',', '.join(x.name for x in sorted(out.glob('*.pdf'))))

if __name__=='__main__':main()
