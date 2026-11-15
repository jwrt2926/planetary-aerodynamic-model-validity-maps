#!/usr/bin/env python3
"""Independent, read-only analysis of planetary flow-regime mechanisms.

Reads NASA GRAM-derived CSVs. Computes: (i) surface dimensionless-model
parameters, (ii) μ0/(ρ0*a0), (iii) surface pressure H0 from first kilometre,
(iv) log mean-free-path gradient with PCHIP interpolation and Kn=0.01 crossing,
(v) source-term decomposition of d ln λ/dz from original 1 km samples.

No GRAM model is rerun and the source CSVs are never written to.

Run: python reproduce_mechanisms.py --data-dir PATH --out-dir PATH
"""
from __future__ import annotations
import argparse, hashlib, csv, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_FILES={'Earth':'earth3.csv','Mars':'mars3(1).csv','Titan':'titan3.csv'}
ZMAX={'Earth':100,'Mars':100,'Titan':500}
# External physical-body facts, NOT fields from the supplied atmospheric CSVs.
# Titan: NASA 20220016292, Titan §2.1; Earth/Mars: NASA JPL planetary physical parameters.
G={'Earth':9.81,'Mars':3.71,'Titan':1.35}
RADIUS_KM={'Earth':6371.0,'Mars':3389.5,'Titan':2575.0}
MU='DynamicViscosity_ChapmanEnskog_Wilke_Pa_s';LAM='MeanFreePath_ViscosityBased_m'
RM='GasConstant_FromGRAMComposition_JkgK'


def get_file(folder,name):
    if name!='Mars': return folder/DEFAULT_FILES[name]
    for candidate in ['mars3(1).csv','mars3.csv']:
        if (folder/candidate).exists():return folder/candidate
    raise FileNotFoundError('mars3(1).csv or mars3.csv not found')


def read_planet(folder,planet):
    file=get_file(folder,planet)
    df=pd.read_csv(file)
    keys=['Height_km','Pressure_Pa','Density_kgm3','Temperature_K',MU,'SpeedOfSound_ms',LAM,RM]
    for key in keys:
        if key not in df:raise ValueError(f'{file.name}: missing {key}')
    x=df.Height_km.to_numpy(float)
    assert len(x)==2201 and np.array_equal(x,np.arange(2201))
    for key in keys[1:]:
        values=df[key].to_numpy(float)
        assert np.all(np.isfinite(values)) and np.all(values>0), f'{file.name}: {key} invalid'
    return file,df


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path('/mnt/data'))
    parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
    a=parser.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    entries=[];meta={'method':'1 km CSV input, positive-property log-PCHIP derivative on displayed altitude interval',
                     'earth_mars_profile':'48-sample averaged GRAM-derived data, Titan nominal GRAM-derived data',
                     'notes':['gravity external, not read from atmospheric CSV','H0 is first-1-km pressure-fitted scale height','logarithmic mean-free-path gradient is 1/H_lambda, not H_lambda','interpolated derivative is diagnostic rather than a new atmospheric measurement'],
                     'gravity_sources':{'Earth_and_Mars':'https://ssd.jpl.nasa.gov/planets/phys_par.html',
                                         'Titan':'https://ntrs.nasa.gov/api/citations/20220016292/downloads/TM-20220016292.pdf?attachment=true'},
                     'input_sha256':{},'planetary_results':{}}
    outcurve={}
    for planet in ['Earth','Mars','Titan']:
        file,df=read_planet(a.data_dir,planet)
        meta['input_sha256'][file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        zz=df.Height_km.to_numpy(float);mx=ZMAX[planet];q=np.linspace(0,mx,int(mx/.1)+1)
        ilog={k:PchipInterpolator(zz,np.log(df[k].to_numpy(float))) for k in ['Pressure_Pa','Temperature_K',MU,LAM,RM]}
        kcurve=ilog[LAM].derivative()(q)
        zcross=float(brentq(lambda z:float(ilog[LAM](z)-np.log(0.01)),0,mx))
        kcross=float(ilog[LAM].derivative()(zcross));Hlam=1/kcross
        H0=1/np.log(df.Pressure_Pa.iloc[0]/df.Pressure_Pa.iloc[1])
        l0=float(df[LAM].iloc[0]);r0=float(df.Density_kgm3.iloc[0]);u0=float(df[MU].iloc[0]);v0=float(df.SpeedOfSound_ms.iloc[0]);
        A=u0/(r0*v0)
        critlength=5e5/.3*A
        ziso=H0*np.log(0.01/l0)
        reconstructed=df[MU].iloc[:mx+1].values/df.Pressure_Pa.iloc[:mx+1].values * np.sqrt(np.pi*df[RM].iloc[:mx+1].values*df.Temperature_K.iloc[:mx+1].values/2)
        maxrel=float(np.max(np.abs(reconstructed/df[LAM].iloc[:mx+1].values-1)))
        assert maxrel<1e-7,(planet,maxrel)
        # Derivative contributions (log PCHIP values, not exact sum because fields
        # are interpolated separately). Also compute the EXACT discrete log
        # decomposition from native 1-km points around the threshold.
        k_terms={
          'pressure':-float(ilog['Pressure_Pa'].derivative()(zcross)),
          'viscosity':float(ilog[MU].derivative()(zcross)),
          'temperature':.5*float(ilog['Temperature_K'].derivative()(zcross)),
          'mixture_R':.5*float(ilog[RM].derivative()(zcross))}
        i=int(np.floor(zcross));j=i+1
        def delta_log(c):return float(np.log(df[c].iloc[j]/df[c].iloc[i]))/(zz[j]-zz[i])
        finite_terms={'pressure':-delta_log('Pressure_Pa'), 'viscosity':delta_log(MU),
                      'temperature':.5*delta_log('Temperature_K'), 'mixture_R':.5*delta_log(RM)}
        finite_lambda=delta_log(LAM)
        assert abs(finite_lambda-sum(finite_terms.values()))<2e-9, (planet,'finite log identity failed')
        ent={'planet':planet,'rho0_kgm3':r0,'mu0_Pas':u0,'a0_ms':v0,'lambda0_m':l0,
             'g0_ms2':G[planet], 'H0_pressure_km':H0, 'mu_over_rhoa_m':A,'Lcrit_m':critlength,
             'isothermal_zKn_km':ziso,'GRAM_zKn_km':zcross,'delta_isothermal_minus_GRAM_km':ziso-zcross,
             'inverse_local_lambda_scaleheight_at_crossing_per_km':kcross,'local_Hlambda_at_crossing_km':Hlam,
             'gradient_terms_PCHIP_per_km':k_terms,'gradient_terms_native_1km_per_km':finite_terms,
             'actual_native_1km_loglambda_gradient_per_km':finite_lambda,
             'max_relative_mfp_identity_error':maxrel,
             'gravity_decline_by_crossing_pct':100*(1-(RADIUS_KM[planet]/(RADIUS_KM[planet]+zcross))**2)}
        entries.append(ent);meta['planetary_results'][planet]=ent
        outcurve[planet]={'z':q,'rate':kcurve,'ref':np.full_like(q,1/H0),'zcross':zcross,'kcross':kcross}

    # Check qualitative result against source data, not assumptions.
    assert all(ent['inverse_local_lambda_scaleheight_at_crossing_per_km']>0 for ent in entries)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(3,1,figsize=(7.3,8.35))
    C=['#2879a2','#bd6d4b','#39856f']
    for ax,p,c in zip(axes,['Earth','Mars','Titan'],C):
        e=meta['planetary_results'][p];d=outcurve[p]
        ax.plot(d['z'],d['rate'],color=c,lw=1.9,label=r'GRAM-derived $d\ln\lambda/dz$')
        ax.axhline(d['ref'][0],color='#6c7079',ls='--',lw=1.3,label=r'Isothermal reference $1/H_0$')
        ax.axvline(d['zcross'],color='#323741',ls=':',lw=1.15)
        ax.scatter([d['zcross']],[d['kcross']],s=35,color=c,ec='white',lw=.6,zorder=6)
        ax.text(.99,.96,f'Kn=0.01 at {d["zcross"]:.1f} km\nLocal $H_\\lambda$={e["local_Hlambda_at_crossing_km"]:.1f} km',
                ha='right',va='top',transform=ax.transAxes,fontsize=8.7,
                bbox=dict(facecolor='white',edgecolor='#d7d9dc',alpha=.90,boxstyle='round,pad=.35'))
        ax.set_title(f'({"abc"[["Earth","Mars","Titan"].index(p)]}) {p}',loc='left',weight='semibold',fontsize=12)
        ax.set_xlim(0,ZMAX[p]);ax.set_ylim(bottom=0)
        ax.set_ylabel(r'$d\ln\lambda/dz$ (km$^{-1}$)')
        ax.set_xlabel('Altitude (km)')
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    hs=[plt.Line2D([],[],color=C[0],lw=2,label=r'GRAM-derived $d\ln\lambda/dz$'),
        plt.Line2D([],[],color='#6c7079',lw=1.3,ls='--',label=r'Isothermal $1/H_0$'),
        plt.Line2D([],[],color='#323741',lw=1.15,ls=':',label=r'$\mathrm{Kn}=0.01$ crossing')]
    fig.legend(handles=hs,loc='lower center',bbox_to_anchor=(.5,.012),ncol=3,frameon=False,fontsize=8.8)
    fig.subplots_adjust(left=.13,right=.968,bottom=.105,top=.968,hspace=.42)
    base=a.out_dir/'Fig_S2_Local_MeanFreePath_Growth'
    for ext in ('pdf','png','svg'):
        fig.savefig(base.with_suffix('.'+ext),dpi=300 if ext=='png' else None,facecolor='white')
    plt.close(fig)
    fields=['planet','rho0_kgm3','mu0_Pas','a0_ms','lambda0_m','g0_ms2','H0_pressure_km','mu_over_rhoa_m','Lcrit_m',
            'GRAM_zKn_km','inverse_local_lambda_scaleheight_at_crossing_per_km','local_Hlambda_at_crossing_km']
    with (a.out_dir/'surface_reference_parameters.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:e[k] for k in fields} for e in entries)
    (a.out_dir/'mechanism_results.json').write_text(json.dumps(meta,indent=2)+'\n')
    for e in entries:print(e['planet'],f"Q={e['mu_over_rhoa_m']:.9g} m",f"Lcrit={e['Lcrit_m']:.6f} m",f"H0={e['H0_pressure_km']:.4f} km",f"zKn={e['GRAM_zKn_km']:.4f} km",f"Hlambda={e['local_Hlambda_at_crossing_km']:.3f} km",f"dloglam/dz={e['inverse_local_lambda_scaleheight_at_crossing_per_km']:.5f}/km")

if __name__=='__main__':main()
