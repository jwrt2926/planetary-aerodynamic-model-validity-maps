#!/usr/bin/env python3
"""Section 6.5: reproducible one-at-a-time classifier sensitivity from NASA GRAM-derived CSVs.

Usage: python compute_sensitivity.py [--data-dir DIR] [--out-dir DIR]
No AI-generated figures. Exports vector PDF, PNG, SVG, CSV & JSON. All intervals are
specified; mean atmospheric properties are NOT independent sampled profiles.
"""
from pathlib import Path
import argparse, json, csv
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

P={'Kn':0.01,'Ma':0.3,'epsilon':0.05,'Re_tr':5e5}
VARY={
 'Kn':[0.008,0.012],
 'Ma':[0.25,0.35],
 'epsilon':[0.04,0.06],
 'Re_tr':[4e5,6e5],
}
ZMAX={'earth':100,'mars':100,'titan':500}
ALT_STEP=0.1
U=np.geomspace(0.1,1000,1201)
COLOURS={'earth':'#2168A8','mars':'#CB6F46','titan':'#518A76'}


def get_profile(data_dir,planet):
 import pandas as pd
 df=pd.read_csv(data_dir/f'{planet}3.csv')
 zraw=df['Height_km'].values.astype(float)
 z=np.linspace(0,ZMAX[planet],round(ZMAX[planet]/ALT_STEP)+1)
 keys={'rho':'Density_kgm3','mu':'DynamicViscosity_ChapmanEnskog_Wilke_Pa_s','a':'SpeedOfSound_ms','lam':'MeanFreePath_ViscosityBased_m','temp':'Temperature_K','p':'Pressure_Pa'}
 out={}
 for name,col in keys.items():
  orig=df[col].values.astype(float)
  assert np.isfinite(orig).all() and np.all(orig>0), (planet,name)
  itp=PchipInterpolator(zraw,np.log(orig))
  out[name]=np.exp(itp(z));out[name+'_raw']=orig;out[name+'_interp']=itp
 out['z']=z;out['zraw']=zraw
 return out

def cls(re,ma,kn,p):
 """0=incompressible NS, 1=Blasius, 2=broader BL, 3=compressible NS, 4=rarefied/slip"""
 b=np.broadcast_arrays(re,ma,kn)
 out=np.zeros(b[0].shape,dtype=np.uint8)
 re,ma,kn=b
 re_min=(5.0/p['epsilon'])**2
 low=(ma<p['Ma'])&(kn<p['Kn'])
 out[low & (re>=re_min) & (re<p['Re_tr'])]=1
 out[low & (re>=p['Re_tr'])]=2
 out[(ma>=p['Ma']) & (kn<p['Kn'])]=3
 out[kn>=p['Kn']]=4
 return out

def grid(d,p,lam_factor=1):
 re=d['rho'][:,None]*U[None,:]/d['mu'][:,None] # L=1 m
 ma=U[None,:]/d['a'][:,None]
 kn=d['lam'][:,None]*lam_factor # L=1 m
 return cls(re,ma,kn,p)

def crossing(d,crit,scale=1):
 diff=d['lam_interp'](d['zraw'])+np.log(scale)-np.log(crit)
 hits=np.flatnonzero(diff[:-1]*diff[1:]<0)
 if len(hits)!=1:
  raise ValueError(f'Need unique Kn crossing, got {len(hits)}')
 i=int(hits[0]);return float(brentq(lambda zz:d['lam_interp'](zz)+np.log(scale)-np.log(crit),i,i+1))

def write_csv(path,rows,columns):
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
 ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
 args=ap.parse_args();out=args.out_dir;out.mkdir(exist_ok=True,parents=True)
 scenarios=[];knresults=[];summary={}
 for name,zmax in ZMAX.items():
  d=get_profile(args.data_dir,name)
  nominal=grid(d,P)
  basecross=crossing(d,P['Kn'])
  summary[name]={'base_kn_km':basecross,'nominal_class_fractions_pct':{str(k):float(np.mean(nominal==k)*100) for k in range(5)}}
  # check original points (preserves values of underlying atmospheric data)
  zz=d['z'];zorig=d['zraw'];ind=np.arange(0,round(zmax/ALT_STEP)+1,round(1/ALT_STEP))
  assert np.allclose(zz[ind],np.arange(zmax+1))
  # these values are exactly equal on original grid for PCHIP
  for key in ['lam','rho','mu','a']:
   assert np.allclose(d[key][ind],d[key+'_raw'][:zmax+1],rtol=1e-9)
  for key,vals in VARY.items():
   for val,qual in zip(vals,['lower','upper']):
    pert=dict(P);pert[key]=val
    altered=grid(d,pert)
    total=float(np.mean(altered!=nominal)*100)
    # z <= baseline Kn cutoff: useful context for global fraction being diluted by rarefied region
    contmask=(zz<basecross)
    cfrac=float(np.mean(altered[contmask,:]!=nominal[contmask,:])*100)
    row={'planet':name.title(),'parameter':key,'case':qual,'value':float(val),'changed_global_pct':total,
         'changed_below_nominal_Kn_pct':cfrac,'n_cells_total':int(nominal.size),
         'n_changed':int(np.count_nonzero(altered!=nominal))}
    scenarios.append(row)
  for label, factor, crit in [
   ('Kn threshold 0.008',1,0.008),('Nominal',1,0.01),('Kn threshold 0.012',1,0.012),
   ('Mean free path -10%',0.9,0.01),('Mean free path +10%',1.1,0.01),
   ('Mean free path -20%',0.8,0.01),('Mean free path +20%',1.2,0.01),
   ('Slip-envelop limit 0.1',1,0.1)]:
   c=crossing(d,crit,scale=factor)
   knresults.append({'planet':name.title(),'case':label,'lambda_factor':factor,'Kn_threshold':crit,
      'crossing_km':c,'shift_vs_nominal_km':c-basecross})
  # scenarios for wider Re critical (different from full classification rule uncertainty)
  for val in [1e6,1e7]:
   altp=dict(P);altp['Re_tr']=val
   altered=grid(d,altp)
   scenarios.append({'planet':name.title(),'parameter':'Re_tr_wide','case':f'{val:.0e}','value':float(val),
    'changed_global_pct':float(np.mean(altered!=nominal)*100),
    'changed_below_nominal_Kn_pct':float(np.mean(altered[zz<basecross,:]!=nominal[zz<basecross,:])*100),
    'n_cells_total':int(nominal.size),'n_changed':int(np.count_nonzero(altered!=nominal))})
  # local derivative of log lambda and derivative-predicted displacement at Kn 0.01
  s=float(d['lam_interp'].derivative()(basecross))
  Hlambda=1/s
  summary[name]['lambda_log_gradient_per_km']=s
  summary[name]['Hlambda_km']=Hlambda
  summary[name]['prediction_for_lambda_plus10pct_km']=-Hlambda*np.log(1.1)
  summary[name]['prediction_for_lambda_minus10pct_km']=-Hlambda*np.log(0.9)
  # compare three alternative models at native sample points for reproducible class semantics
  print(name,'Kn',basecross,'local H_lambda',Hlambda)

 write_csv(out/'threshold_sensitivity.csv',scenarios,list(scenarios[0].keys()))
 write_csv(out/'Kn_boundary_movements.csv',knresults,list(knresults[0].keys()))
 (out/'results_summary.json').write_text(json.dumps({'method':{'altitude_step_km':ALT_STEP,'velocity_ms':{'min':0.1,'max':1000,'points':len(U),'spacing':'equal spacing in log U'},'source':'earth3.csv, mars3.csv, titan3.csv','interp':'log PCHIP; no postclassification smoothing','grid_weighting':'equal weight per altitude step and logarithmically spaced velocity point; not probability weighted','baseline':P,'threshold_variations':VARY,'transport_uncertainty':'Independent uniform perturbation to mean-free-path; not a joint transport model change','Kn_0.1':'Alternative slip-permitting envelope; never interpreted as no-slip validity'},'planets':summary,'sensitivity_rows':scenarios,'Kn_rows':knresults},indent=2))

 # Fig. 5: 3 horizontal panels one per planet, 4 parameters × 2 bars
 parlabels=[r'$\mathrm{Kn}_{\mathrm{crit}}$',r'$\mathrm{Ma}_{\mathrm{crit}}$',r'$\varepsilon_{\mathrm{BL}}$',r'$\mathrm{Re}_{\mathrm{tr}}$']
 keys=list(VARY.keys())
 fig,axs=plt.subplots(3,1,figsize=(7.1,9.0),sharex=True)
 for ax,(name,zmax) in zip(axs,ZMAX.items()):
  vals=[next(r['changed_global_pct'] for r in scenarios if r['planet']==name.title() and r['parameter']==k and r['case']==t) for k in keys for t in ('lower','upper')]
  x=np.arange(4);lo=np.array(vals[::2]);hi=np.array(vals[1::2]);wid=.35
  barsA=ax.bar(x-wid/2,lo,width=wid,color='#4879A5',label='Lower threshold',edgecolor='white',linewidth=.4)
  barsB=ax.bar(x+wid/2,hi,width=wid,color='#C97450',label='Upper threshold',edgecolor='white',linewidth=.4)
  for bars in [barsA,barsB]:
   for bar in bars:
    h=bar.get_height()
    if h>=0.15:ax.text(bar.get_x()+bar.get_width()/2,h+0.15,f'{h:.1f}',fontsize=8,ha='center',color='#313846')
  ax.set_title(name.title(),loc='left',fontsize=12,weight='bold',pad=6)
  ax.set_xticks(x,parlabels,fontsize=11)
  ax.grid(axis='y',alpha=.16)
  ax.spines[['top','right']].set_visible(False)
  ax.set_axisbelow(True)
  ax.set_ylim(0,max(2,max(vals)*1.20+0.5))
  ax.set_ylabel('Changed cells (%)',fontsize=9)
 fig.legend(handles=[Patch(color='#4879A5',label='Lower value'),Patch(color='#C97450',label='Upper value')],ncol=2,loc='lower center',bbox_to_anchor=(.5,.013),frameon=False,fontsize=9.5)
 fig.suptitle('Sensitivity to nominal model-selection thresholds',fontsize=14,weight='bold',y=.98)
 fig.subplots_adjust(left=.11,right=.975,top=.92,bottom=.11,hspace=.37)
 fig.text(.5,.059,'Classification differences on a uniform-altitude / logarithmic-velocity grid; $L=1$ m',ha='center',fontsize=8.6,color='#555B63')
 for extension in ('pdf','png','svg'):
  fig.savefig(out/f'Fig5_Threshold_Classification_Sensitivity.{extension}',dpi=270,facecolor='white')
 plt.close(fig)

 # Fig. 6: two panels: (a) local Kn crossing shifts per planet for +- threshold/transport, (b) wide Re scenario
 fig,axs=plt.subplots(2,1,figsize=(7.2,7.9),gridspec_kw={'height_ratios':[1.2,1]})
 ax=axs[0];names=['Earth','Mars','Titan'];labels=['Kn 0.008','Kn 0.012','$\\lambda$ −10%','$\\lambda$ +10%','$\\lambda$ −20%','$\\lambda$ +20%']
 rows_keys=['Kn threshold 0.008','Kn threshold 0.012','Mean free path -10%','Mean free path +10%','Mean free path -20%','Mean free path +20%']
 x=np.arange(3);wid=.12
 palette=['#356D97','#C77853','#79A2BB','#CFA28D','#346A82','#B75B3E']
 for j,(rk,lab) in enumerate(zip(rows_keys,labels)):
  yy=[next(r['shift_vs_nominal_km'] for r in knresults if r['planet']==n and r['case']==rk) for n in names]
  ax.bar(x+(j-2.5)*wid,yy,width=wid,label=lab,color=palette[j],edgecolor='white',lw=.25)
 ax.axhline(0,lw=.8,color='#4E5460');ax.set_xticks(x,names,fontsize=10)
 ax.set_ylabel('Shift in Kn = 0.01 altitude (km)',fontsize=10)
 ax.set_title('(a) Rarefaction boundary sensitivity',fontsize=12,loc='left',weight='bold')
 ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 ax.legend(ncol=3,fontsize=7.75,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.14))
 ax2=axs[1];thetas=[4e5,5e5,6e5,1e6,1e7]
 for name in ZMAX:
  d=get_profile(args.data_dir,name)
  rhoa=d['rho']*d['a']/d['mu']
  # at Mach limit: max Re over continuum only; also measure fraction of log U × z continuum-low-Mach in broad BL
  percent=[]
  for tr in thetas:
   temp=dict(P);temp['Re_tr']=tr
   categories=grid(d,temp)
   percent.append(float(np.mean(categories==2)*100))
  ax2.plot(thetas,percent,'o-',markersize=5,lw=1.7,label=name.title(),color=COLOURS[name])
 ax2.set_xscale('log');ax2.set_xticks([4e5,1e6,1e7], ['4×10⁵','10⁶','10⁷'])
 ax2.axvline(5e5,color='#7B8391',lw=1,ls=':',alpha=.8)
 ax2.set_ylim(bottom=0);ax2.set_ylabel('Broader BL region (% of map)',fontsize=10)
 ax2.set_xlabel('Assumed laminar-transition Reynolds number',fontsize=10)
 ax2.set_title('(b) Wider laminar-transition scenarios',loc='left',fontsize=12,weight='bold')
 ax2.grid(axis='y',alpha=.16);ax2.set_axisbelow(True);ax2.spines[['top','right']].set_visible(False)
 ax2.legend(ncol=3,frameon=False,fontsize=9)
 fig.suptitle('Local boundary shifts and wider transition scenarios',fontsize=13.4,weight='bold',y=.985)
 fig.subplots_adjust(top=.91,bottom=.095,left=.11,right=.97,hspace=.60)
 for ext in ('pdf','png','svg'):
  fig.savefig(out/f'Fig6_Boundary_and_Extended_Sensitivity.{ext}',dpi=270,facecolor='white')
 plt.close(fig)

 # auxiliary table for wider scenario boundary response and broader bl map area
 wide=[]
 for name in ZMAX:
  d=get_profile(args.data_dir,name)
  for tr in [4e5,5e5,6e5,1e6,1e7]:
   p=dict(P);p['Re_tr']=tr
   area=float(np.mean(grid(d,p)==2)*100)
   wide.append({'planet':name.title(),'Re_transition':tr,'broader_BL_map_area_pct':area})
 write_csv(out/'wide_transition_scenarios.csv',wide,list(wide[0].keys()))
 print('\nClassification sensitivity:')
 for row in scenarios:print(row['planet'],row['parameter'],row['case'],f"{row['changed_global_pct']:.3f}%",f"{row['changed_below_nominal_Kn_pct']:.3f}%")
 print('\nBoundary movements:')
 for row in knresults:print(row['planet'],row['case'],f"{row['crossing_km']:.3f}",f"shift={row['shift_vs_nominal_km']:+.3f}")
 print('\nBroader BL percent:')
 for row in wide:print(row)

if __name__=='__main__':main()
