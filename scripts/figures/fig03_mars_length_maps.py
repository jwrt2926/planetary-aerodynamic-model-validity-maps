#!/usr/bin/env python3
"""Independent study of velocity and characteristic-length effects.

Reproduces Table 3 update and Supplementary Figure S1 from the *original*
Earth/Mars/Titan GRAM-derived final 1-km CSVs. The original source files are
never edited. Only display properties are interpolated; no new GRAM runs.

Usage:
  python reproduce_velocity_length.py --data-dir /path/to/CSVs --out-dir /path/to/output

Expected filenames: earth3.csv, mars3(1).csv (or mars3.csv), titan3.csv.
Requirements: numpy pandas scipy matplotlib.
"""
from __future__ import annotations
import argparse, hashlib, json, csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator

RE_MIN, RE_TRANS, MA_CRIT, KN_CRIT = 1.0e4, 5.0e5, 0.3, 0.01
UPLOT = (0.1, 1000.0)
SOURCE_NAMES = {'Earth':['earth3.csv'], 'Mars':['mars3(1).csv','mars3.csv'], 'Titan':['titan3.csv']}
ZMAX = {'Earth':100., 'Mars':100., 'Titan':500.}
COLORS={
 'incomp':'#447B9F','blasius':'#30A48F','bl':'#E9AE4A',
 'comp':'#D66D63','rarefied':'#8E79A7'}

def get_source(directory, planet):
 for filename in SOURCE_NAMES[planet]:
  q = directory / filename
  if q.is_file(): return q
 raise FileNotFoundError(f'Cannot find {SOURCE_NAMES[planet]} in {directory}')

def load(directory, name):
 path=get_source(directory,name)
 d=pd.read_csv(path)
 expected=['Height_km','Density_kgm3','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',
           'SpeedOfSound_ms','MeanFreePath_ViscosityBased_m']
 if any(n not in d for n in expected):raise ValueError(f'{name} missing required columns')
 z=d.Height_km.to_numpy(dtype=float)
 if len(z)!=2201 or not np.allclose(z,np.arange(2201),atol=1e-10):
  raise ValueError(f'Altitude grid mismatch for {name}')
 itp={}
 for c in expected[1:]:
  y=d[c].to_numpy(float)
  if not np.all(np.isfinite(y)) or np.any(y<=0):raise ValueError(f'Invalid {c} in {path}')
  itp[c]=PchipInterpolator(z,np.log(y),extrapolate=False)
 return {'name':name,'file':path,'d':d,'z':z,'itp':itp,
         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def at(p,c,z):return np.exp(p['itp'][c](z))

def crossing(p,L):
 loglam=p['itp']['MeanFreePath_ViscosityBased_m']
 original=np.log(p['d']['MeanFreePath_ViscosityBased_m'].to_numpy(float))
 target=np.log(KN_CRIT*L)
 ii=np.flatnonzero(np.diff(np.sign(original-target)))
 if len(ii)!=1:raise ValueError(f"{p['name']}: crossing count {len(ii)} for L={L}")
 i=int(ii[0]);return float(brentq(lambda h: float(loglam(h)-target),i,i+1))

def Mach_limit(p,z):return MA_CRIT*at(p,'SpeedOfSound_ms',z)

def re_at_mach_limit(p,z,L=1.):
 rho=at(p,'Density_kgm3',z)
 a=at(p,'SpeedOfSound_ms',z)
 mu=at(p,'DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',z)
 return MA_CRIT*rho*a*L/mu

def critical_length(p):
 # Verified maximum of this factor over the *reported* altitude domain;
 # it occurs at z=0 km for all 3 source datasets.
 z=np.arange(ZMAX[p['name']]+1)
 limiting=re_at_mach_limit(p,z)
 k=int(np.argmax(limiting))
 if k!=0:raise AssertionError(f"{p['name']} limiting Re maximum no longer at surface")
 return RE_TRANS/limiting[k],float(limiting[k])

def _fill(ax,z,lo,hi,enabled,colour):
 lo=np.maximum(lo,UPLOT[0]);hi=np.minimum(hi,UPLOT[1]);m=enabled & (hi>lo)
 ax.fill_betweenx(z,lo,hi,where=m,interpolate=True,color=colour,linewidth=0,zorder=2)

def plot_mars_pair(mars,out):
 # Only 0--25 km is shown to make the new wider BL region easy to inspect;
 # this is explicitly a close-up, not the full altitude domain.
 z=np.arange(0,25.00001,.025)
 rho=at(mars,'Density_kgm3',z); mu=at(mars,'DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',z)
 a=at(mars,'SpeedOfSound_ms',z); lam=at(mars,'MeanFreePath_ViscosityBased_m',z)
 fig,axes=plt.subplots(1,2,figsize=(10.0,4.9),sharey=True)
 for ax,L in zip(axes,[1.,10.]):
  uma=MA_CRIT*a
  umin=RE_MIN*mu/(rho*L)
  utr=RE_TRANS*mu/(rho*L)
  valid=lam/L<KN_CRIT
  ones=np.full_like(z,UPLOT[0]); upp=np.full_like(z,UPLOT[1])
  ax.axhspan(0,25,color=COLORS['rarefied'],zorder=0)
  _fill(ax,z,ones,np.minimum(umin,uma),valid,COLORS['incomp'])
  _fill(ax,z,umin,np.minimum(utr,uma),valid,COLORS['blasius'])
  _fill(ax,z,utr,uma,valid,COLORS['bl'])
  _fill(ax,z,uma,upp,valid,COLORS['comp'])
  for uarr,condition in ((uma,valid),(umin,valid&(umin<uma)),(utr,valid&(utr<uma))):
   m=condition&(uarr>UPLOT[0])&(uarr<UPLOT[1]);ax.plot(uarr[m],z[m],lw=.8,color='#343944',zorder=4)
  ax.set_xscale('log');ax.set_xlim(*UPLOT);ax.set_ylim(0,25)
  ax.set_xticks([.1,1,10,100,1000]);ax.set_xticklabels([r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$'])
  ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*.1,numticks=100));ax.xaxis.set_minor_formatter(NullFormatter())
  ax.yaxis.set_major_locator(MultipleLocator(5))
  ax.set_title(fr'Mars: $L={L:g}$ m',loc='left',fontsize=12,pad=10,weight='semibold')
  ax.set_xlabel(r'Freestream velocity, $U$ (m s$^{-1}$)',fontsize=10)
  ax.tick_params(labelsize=9)
  for spine in ax.spines.values():spine.set_color('#748090')
 axes[0].set_ylabel('Altitude above surface (km)',fontsize=10)
 names=['Incompressible NS','Blasius','Broader boundary layer','Compressible NS','Rarefied/slip corrected']
 cols=[COLORS[c] for c in ('incomp','blasius','bl','comp','rarefied')]
 fig.legend([Patch(color=c) for c in cols],names,ncol=3,loc='lower center',
            bbox_to_anchor=(.5,.01),fontsize=8.8,frameon=False,labelspacing=.75,handlelength=1.1)
 fig.suptitle('Martian model validity at two length scales (0–25 km)',fontsize=13,weight='semibold',y=.985)
 fig.subplots_adjust(left=.085,right=.985,bottom=.255,top=.825,wspace=.16)
 stem=out/'Fig_S1_Mars_Model_Validity_L1m_vs_L10m_0-25km'
 for ext in ['pdf','png','svg']:
  fig.savefig(stem.with_suffix('.'+ext),dpi=300 if ext=='png' else None,facecolor='white')
 plt.close(fig)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--data-dir',type=Path,default=Path('/mnt/data'))
 parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
 args=parser.parse_args();args.out_dir.mkdir(exist_ok=True,parents=True)
 planets={name:load(args.data_dir,name) for name in SOURCE_NAMES}
 rows=[];full={}
 for name,p in planets.items():
  z=np.arange(0,int(ZMAX[name])+1,dtype=float)
  u=MA_CRIT*p['d'].SpeedOfSound_ms.iloc[:len(z)].to_numpy(float)
  Lcrit,maxn=critical_length(p)
  row={'planet':name,'Mach_threshold_surface_ms':float(u[0]),
       'Mach_threshold_min_ms':float(u.min()),'Mach_threshold_max_ms':float(u.max()),
       'Mach_threshold_min_altitude_km':float(z[u.argmin()]),
       'Mach_threshold_max_altitude_km':float(z[u.argmax()]),
       'Max_lowMach_Re_at_L1m':maxn,'Critical_length_for_broader_BL_m':Lcrit,
       'Kn0p01_altitude_L1_km':crossing(p,1.),'Kn0p01_altitude_L10_km':crossing(p,10.),
       'csv_filename':p['file'].name,'csv_sha256':p['sha256']}
  rows.append(row);full[name]=row
 # Establish Earth's U=100 m/s local transition altitude from a(z).
 e=planets['Earth']; g=lambda z:float(Mach_limit(e,z)-100.)
 zvals=np.arange(0,101)
 intersects=np.flatnonzero(np.diff(np.sign([g(x) for x in zvals])))
 if len(intersects)!=1:raise AssertionError('Expected one Earth U=100 Mach crossing')
 zcross=float(brentq(g,float(intersects[0]),float(intersects[0]+1)))
 # Mars upper altitude of the L=10 m wider boundary-layer category.
 m=planets['Mars']; zm=np.arange(0,101)
 q=np.array([re_at_mach_limit(m,zi,10.) for zi in zm])-RE_TRANS
 ix=np.flatnonzero(np.diff(np.sign(q)))
 if len(ix)!=1:raise AssertionError('Expected one Mars L=10 threshold crossing')
 mars_l10_cross=float(brentq(lambda zz:float(re_at_mach_limit(m,zz,10.)-RE_TRANS),float(ix[0]),float(ix[0]+1)))
 # Reproduces the critical Mars length and all nominal reference boundaries.
 assert abs(full['Mars']['Critical_length_for_broader_BL_m']-4.32449)<1e-4
 for name,target in {'Earth':85.74178,'Mars':75.16522,'Titan':470.20841}.items():
  assert abs(full[name]['Kn0p01_altitude_L1_km']-target)<.02,(name,full[name]['Kn0p01_altitude_L1_km'])
 full['_metadata']={'source':'The exact three uploaded GRAM-derived final CSVs (no new GRAM model runs)',
  'Mach_threshold':'Ucrit(z) = 0.3 * SpeedOfSound_ms(z)',
  'classification':'priority: Kn>=0.01 rarefied; Ma>=0.3 compressible NS; Re<1e4 incompressible; Re<5e5 Blasius; otherwise broader BL',
  'L_values_m':[1,10],'Mars_figure_altitude_range_km':[0,25],
  'display_interpolation':'log-PCHIP for positive properties; underlying profile samples 1 km apart',
  'Earth_at_U100_Mach_0p3_crossing_km':zcross,
  'Mars_at_L10_boundary_layer_max_altitude_km':mars_l10_cross}
 with (args.out_dir/'mach_and_length_summary.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (args.out_dir/'mach_and_length_results.json').write_text(json.dumps(full,indent=2)+'\n')
 plot_mars_pair(m,args.out_dir)
 for row in rows:
  print(f"{row['planet']:5} Ucrit_surface={row['Mach_threshold_surface_ms']:8.3f}; "
        f"Ucrit_range={row['Mach_threshold_min_ms']:8.3f}--{row['Mach_threshold_max_ms']:8.3f}; "
        f"Lcrit={row['Critical_length_for_broader_BL_m']:8.5f}; "
        f"Kn1km={row['Kn0p01_altitude_L1_km']:8.3f}; Kn10km={row['Kn0p01_altitude_L10_km']:8.3f}")
 print('Earth at U100 Mach crossing z=',zcross)
 print('Mars broader BL max altitude at L10=',mars_l10_cross)

if __name__=='__main__':main()
