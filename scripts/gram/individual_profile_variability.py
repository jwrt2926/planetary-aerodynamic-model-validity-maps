#!/usr/bin/env python3
"""Re-run exact 48 Earth- and Mars-GRAM sampled profiles and compute per-profile Kn crossings.

Uses native GRAM executables and same input parameters as the user's
Earth/Mars 48-profile averaging scripts, but stops altitude grid at 220 km
because only the nominal Kn=0.01 crossing is needed here. No aggregation from
standard deviation; full individual runs. Same Chapman–Enskog/Wilke + gas-
mixture kinetic mean free path as earth3.csv and mars3.csv.

Requires NASA GRAM Suite 2.1.1, Earth's four MERRA-2 All Mean monthly binaries,
plus numpy, pandas, scipy, matplotlib. Script can regenerate 48-case crossing
statistics from the original GRAM installation. Those bulky NASA binary files
are not redistributed in the reproducibility package.
"""
import csv, json, re, subprocess, argparse, sys
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
# Required exact original molecular interaction parameters from viscosity script.
from chapman_enskog_viscosity import SPECIES as LJ
R_U=8.31446261815324
MOL={sp:pars[0] for sp,pars in LJ.items()}
MOL['O3']=47.9982
KNCRIT=0.01;L=1.;Z_N=221
RAW_PHYSICAL=['Temperature_K','Pressure_Pa','Density_kgm3','SpeedOfSound_ms','SpecificGasConstant_JkgK','SpecificHeatRatio','AverageMolecularWeight','TotalNumberDensity_m3','CompressibilityFactor','PressureScaleHeight_km','DensityScaleHeight_km']
ALL_SPS={
 'earth':['Ar','CO2','CO','N2','O2','He','H','CH4','N','O','O3','N2O','H2O'],
 'mars':['CO2','N2','Ar','CO','O2','O','H','He','H2','H2O']}

def mixture_mu_vector(T,number_density,planet):
 """Fully vectorized species viscosities and Wilke mixing (same formula as viscosity script)."""
 sps=ALL_SPS[planet]
 masses=np.array([MOL[s] for s in sps])
 ntot=number_density.sum(axis=1)
 assert np.all(ntot>0)
 x=number_density/ntot[:,None]
 Rmix=R_U/(x @ masses/1000.)
 keep=np.array([s!='O3' for s in sps]); filtered=[s for s in sps if s!='O3']
 xf=x[:,keep]; xf=xf/xf.sum(axis=1)[:,None]
 Tcol=T[:,None]
 mass=np.array([LJ[s][0] for s in filtered])
 sig=np.array([LJ[s][1] for s in filtered])
 well=np.array([LJ[s][2] for s in filtered])
 Ts=Tcol/well[None,:]
 omega=1.16145*Ts**(-.14874)+.52487*np.exp(-.77320*Ts)+2.16178*np.exp(-2.43787*Ts)
 pure=2.6693e-6*np.sqrt(mass[None,:]*Tcol)/(sig[None,:]**2*omega)
 # phi_ij = [1 + (mu_i/mu_j)^.5 (M_j/M_i)^.25]^2 / sqrt(8(1+M_i/M_j))
 phi=(1+np.sqrt(pure[:,:,None]/pure[:,None,:])*(mass[None,None,:]/mass[None,:,None])**.25)**2/np.sqrt(8*(1+mass[None,:,None]/mass[None,None,:]))
 den=np.sum(xf[:,None,:]*phi,axis=2)
 viscosity=np.sum(xf*pure/den,axis=1)
 return viscosity,Rmix

def crossing(z,lam):
 log=np.log(lam)
 d=log-np.log(KNCRIT*L)
 hits=np.flatnonzero((d[:-1]<=0)&(d[1:]>0))
 if len(hits)!=1:return None
 it=PchipInterpolator(z,log)
 i=int(hits[0]);return float(brentq(lambda zz:it(zz)-np.log(KNCRIT*L),z[i],z[i+1]))

def weighted_quantile(v,w,p):
 idx=np.argsort(v);v=np.array(v)[idx];w=np.array(w)[idx];cs=np.cumsum(w)/np.sum(w)
 return float(np.interp(p,cs,v))

def change_input(template,options):
 out=template
 for k,v in options.items():
  pat=r'(?mi)^(\s*'+re.escape(k)+r'\s*=\s*)[^\r\n]*'
  if not re.search(pat,out):raise ValueError('Missing key '+k)
  out=re.sub(pat,lambda m:m.group(1)+str(v),out,count=1)
 return out

def run_case(root,planet,template,options):
 mode='Earth' if planet=='earth' else 'Mars'
 sample=root/mode/'sample_inputs'
 executable=root/'Linux'/('EarthGRAM' if planet=='earth' else 'MarsGRAM')
 tag='var_6_5_profile_tmp'
 full=change_input(template,{**options,'ListFileName':tag+'_LIST','ColumnFileName':tag+'_OUTPUT'})
 infile=sample/f'{tag}_input.txt';infile.write_text(full)
 try:
  proc=subprocess.run([str(executable),'-file',infile.name],cwd=sample,capture_output=True,text=True,timeout=120)
  if proc.returncode or 'unrecoverable error' in proc.stdout.lower():
   raise RuntimeError(f'{planet}: {proc.stdout[-1800:]}\n{proc.stderr[-1800:]}')
  path=sample/f'{tag}_OUTPUT.csv'
  df=pd.read_csv(path)
  return df
 finally:
  for ext in ('_input.txt','_OUTPUT.csv','_LIST.md'):
   (sample/(tag+ext)).unlink(missing_ok=True)


def calc_planet(root,planet,data_dir):
 latnodes,latweights=np.polynomial.legendre.leggauss(3)
 lats=np.rad2deg(np.arcsin(latnodes));lons=[0,90,180,270]
 if planet=='earth':
  tmpl=(root/'Earth'/'sample_inputs'/'merra2_input.txt').read_text()
  seasons=[('month',m) for m in [1,4,7,10]]
 else:
  tmpl=(data_dir/'Mars_GRAM_final_0-1000km_1km_input.txt').read_text()
  dates=[datetime(2020,3,25)+timedelta(days=x) for x in [0,172,344,516]]
  seasons=[('date',d) for d in dates]
 rows=[];z=np.arange(Z_N,dtype=float)
 avgT=np.zeros(Z_N);avgP=np.zeros(Z_N);avgRho=np.zeros(Z_N)
 for la,wlat in zip(lats,latweights):
  for lon in lons:
   if planet=='earth':
    common={
    'InitialLatitude':f'{la:.9f}','InitialLongitude':str(lon),
    'InitialHeight':'0.0','NumberOfPositions':'1',
    'DeltaHeight':'0.0','DeltaLatitude':'0.0','DeltaLongitude':'0.0','DeltaTime':'0.0',
    'M2Hour':'9','M2MinimumLatitude':'-90','M2MaximumLatitude':'90',
    'M2MinimumLongitude':'0','M2MaximumLongitude':'360',
    'UseNCEP':'0','UseRRA':'0',
    'RandomPerturbationScale':'0.0',
    'HorizontalWindPerturbationScale':'0.0','VerticalWindPerturbationScale':'0.0',
    'NumberOfMonteCarloRuns':'1','ExtraPrecision':'1','UseLegacyOutputs':'0',
    'AP':'10.0','DailyF10':'110.0','MeanF10':'110.0',
    'Month':'1','Day':'15','Year':'2020','Hour':'12','Minute':'0','Seconds':'0.0'}
    probe=run_case(root,planet,tmpl,common)
    surface=float(probe['SurfaceHeight_km'].iloc[0])
   else:
    common={'InitialLatitude':f'{la:.9f}','InitialLongitude':str(lon)}
    probe=run_case(root,planet,tmpl,{**common,'NumberOfPositions':'1','InitialHeight':'0.0'})
    surface=float(probe['SurfaceHeight_km'].iloc[0])
   for stype,season in seasons:
    if planet=='earth':
     options={**common,'InitialHeight':f'{surface:.6f}',
         'NumberOfPositions':str(Z_N),'DeltaHeight':'1.0','Month':str(season)}
     label=f'2020-{season:02d}-15'
    else:
     options={**common,'InitialHeight':f'{surface:.4f}','NumberOfPositions':str(Z_N),
       'DeltaHeight':'1.0','DeltaLatitude':'0.0','DeltaLongitude':'0.0','DeltaTime':'0.0',
       'Month':str(season.month),'Day':str(season.day),'Year':str(season.year),
       'Hour':'12','Minute':'0','Seconds':'0.0',
       'DensityPerturbationScale':'0.0','EWWindPerturbationScale':'0.0','NSWindPerturbationScale':'0.0',
       'ComputeMinMax':'0','F107':'68.0','MGCMConstantDustLevel':'0.3','MapYear':'0',
       'StormIntensity':'0.0','ExtraPrecision':'1','UseLegacyOutputs':'0','OffsetModel':'2','IsMolaHeights':'1'}
     label=season.strftime('%Y-%m-%d')
    df=run_case(root,planet,tmpl,options)
    if len(df)!=Z_N:raise ValueError('Expected '+str(Z_N)+' rows got '+str(len(df)))
    heights=df['Height_km'].to_numpy(float)
    if np.max(np.abs(heights-(surface+z)))>.02:raise ValueError('Altitude offset mismatch')
    T=df['Temperature_K'].to_numpy(float);p=df['Pressure_Pa'].to_numpy(float);rho=df['Density_kgm3'].to_numpy(float)
    nd=np.stack([np.maximum(0,df[s+'nd_m3'].to_numpy(float)) for s in ALL_SPS[planet]],axis=1)
    mu,Rmix=mixture_mu_vector(T,nd,planet)
    lam=mu/p*np.sqrt(np.pi*Rmix*T/2)
    k=crossing(z,lam)
    weight=(wlat/2)/16
    if k is None:raise ValueError(f'No unique crossing {planet}/{la}/{lon}/{label} lam0={lam[0]} lam220={lam[-1]}')
    avgT+=weight*T;avgP+=weight*p;avgRho+=weight*rho
    rows.append({'planet':planet.title(),'latitude_deg':float(la),'longitude_deg':float(lon),
      'season':label,'weight':float(weight),'continuum_crossing_km':k,
      'surface_temperature_K':float(T[0]),'surface_density_kgm3':float(rho[0])})
    print(planet,len(rows),'of 48',label,'lat',round(la,3),'lon',lon,'Kn cross',round(k,3),flush=True)
 assert len(rows)==48 and abs(sum(r['weight'] for r in rows)-1)<1e-10
 dfmean=pd.read_csv(data_dir/(planet.title()+'_GRAM_global_reference_mean_0-2200km_1km.csv'))
 # source check with 48-profile average: should match original in-process weighted mean.
 checks={}
 for name,avg in [('Temperature_K',avgT),('Pressure_Pa',avgP),('Density_kgm3',avgRho)]:
  original=dfmean[name].values[:Z_N]
  relative=np.max(np.abs(avg-original)/original)
  checks[name+'_max_rel']=float(relative)
  print('CHECK',planet,name,'maximum relative',relative,flush=True)
 assert checks['Temperature_K_max_rel']<1e-4, 'Individual sampling not reproducing reference mean'
 values=np.array([r['continuum_crossing_km'] for r in rows]);weights=np.array([r['weight'] for r in rows]);weights/=weights.sum()
 weighted_mean=float(weights@values)
 stats={'planet':planet.title(),'n':48,'weighted_mean_km':weighted_mean,
 'weighted_std_km':float(np.sqrt(weights@((values-weighted_mean)**2))),
 'min_km':float(values.min()),'max_km':float(values.max()),
 'q10_km':weighted_quantile(values,weights,.10),
 'median_km':weighted_quantile(values,weights,.5),
 'q90_km':weighted_quantile(values,weights,.9),
 'mean_profile_boundary_km':float(np.nan),
 'comparison_to_mean_profile_km':float(np.nan),
 'reference_validation':checks}
 return rows,stats


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--gram-root',type=Path,required=True,help='Path to GRAM directory within GRAM Suite 2.1.1')
 ap.add_argument('--data-dir',type=Path,default=ROOT)
 ap.add_argument('--out-dir',type=Path,default=ROOT)
 args=ap.parse_args();root=args.gram_root;out=args.out_dir;out.mkdir(exist_ok=True,parents=True)
 all_rows=[];all_stats=[]
 from compute_sensitivity import get_profile,crossing as crossing_ref
 for planet in ('earth','mars'):
  rows,stat=calc_planet(root,planet,args.data_dir)
  ref=crossing_ref(get_profile(args.data_dir,planet),KNCRIT)
  stat['mean_profile_boundary_km']=ref
  stat['comparison_to_mean_profile_km']=stat['weighted_mean_km']-ref
  all_rows.extend(rows);all_stats.append(stat)
 import csv
 p=out/'individual_48profile_Kn_crossings.csv'
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
 (out/'individual_profile_summary.json').write_text(json.dumps(all_stats,indent=2))
 # Fig7: point distribution with planet panels, weight-encoded small transparent markers
 fig,axs=plt.subplots(2,1,figsize=(7.1,5.9),sharex=False)
 rng=np.random.default_rng(1402)
 for ax,st in zip(axs,all_stats):
  planet=st['planet']; rows=[r for r in all_rows if r['planet']==planet]
  xs=np.array([r['continuum_crossing_km'] for r in rows]); ws=np.array([r['weight'] for r in rows])
  # deterministic y jitter distinguishes overlapping cases
  jitter=rng.normal(0,.115,len(xs))
  ax.scatter(xs,jitter,s=210*ws/ws.max()+12,alpha=.59,color='#42749D' if planet=='Earth' else '#CB7F52',edgecolor='white',lw=.45)
  ax.axvline(st['weighted_mean_km'],color='#264859',lw=1.7,label='Weighted mean of 48 crossings')
  ax.axvline(st['mean_profile_boundary_km'],color='#C0503F',lw=1.7,linestyle='--',label='Crossing of averaged profile')
  ax.axvspan(st['weighted_mean_km']-st['weighted_std_km'],st['weighted_mean_km']+st['weighted_std_km'],color='#7D9BA9',alpha=.13,zorder=0)
  ax.set_yticks([]);ax.set_title(f"{planet}: mean {st['weighted_mean_km']:.2f} km, SD {st['weighted_std_km']:.2f} km",loc='left',fontsize=11.5,fontweight='semibold')
  ax.set_xlabel('Altitude of Kn = 0.01 boundary (km)',fontsize=10)
  ax.grid(axis='x',alpha=.19);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False)
  ax.set_ylim(-.40,.40)
  span=xs.max()-xs.min();ax.set_xlim(xs.min()-max(2,span*.1),xs.max()+max(2,span*.1))
 fig.legend(handles=[Line2D([],[],color='#264859',lw=1.7,label='Weighted mean of individual crossings'),
     Line2D([],[],color='#C0503F',ls='--',lw=1.7,label='Crossing from averaged atmosphere')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,fontsize=9,frameon=False)
 fig.suptitle('Earth and Mars: 48-profile atmospheric variability',fontsize=13.4,fontweight='semibold',y=.985)
 fig.subplots_adjust(left=.09,right=.97,top=.88,bottom=.16,hspace=.48)
 for ext in ('pdf','png','svg'):
  fig.savefig(out/f'Fig7_Individual_Profile_Kn_Variability.{ext}',dpi=260,facecolor='white')
 plt.close(fig)
 print('FINAL',json.dumps(all_stats,indent=2),flush=True)

if __name__=='__main__':main()
