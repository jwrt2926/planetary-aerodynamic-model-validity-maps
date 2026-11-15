#!/usr/bin/env python3
"""Create a single reproducible, A4-portrait, three-panel planetary validity figure.

Inputs: earth3.csv, mars3.csv, titan3.csv in the same directory.
Outputs: Fig1_Planetary_Model_Validity_Combined.pdf/.svg/.png and provenance JSON.

This uses the original GRAM-derived numerical data and the nominal classifier:
  Kn>=0.01 -> rarefied/slip corrected
  Ma>=0.3 -> compressible Navier-Stokes
  Re<=1e4 -> incompressible Navier-Stokes
  Re<5e5 -> Blasius
  otherwise -> wider boundary-layer model

Positive properties are log-PCHIP interpolated from 1 km GRAM profiles solely
for smooth visualization; the interpolated curve is not additional GRAM data.
The region fills follow the exact switching functions, not raster smoothing.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parent
STEM = 'Fig1_Planetary_Model_Validity_Combined'
L=1.0
KN_CRIT=0.01
MA_CRIT=0.3
RE_MIN=1e4
RE_TRANS=5e5
UMIN,UMAX=0.1,1000.0
DZ=0.05
PLANETS=[('Earth','earth',100),('Mars','mars',100),('Titan','titan',500)]
# Same categorical colors as original reproducible report figures.
COLORS={'incompressible':'#447B9F','blasius':'#30A48F','boundary-layer':'#E9AE4A',
        'compressible':'#D66D63','rarefied':'#8E79A7'}


def read_profile(planet,zmax):
    with (ROOT/f'{planet}3.csv').open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    cols=['Height_km','Density_kgm3','DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',
          'MeanFreePath_ViscosityBased_m','SpeedOfSound_ms']
    data={key:np.array([float(row[key]) for row in rows],dtype=float) for key in cols}
    z0=data['Height_km']
    assert np.array_equal(z0,np.arange(len(z0),dtype=float))
    z=np.linspace(0,zmax,round(zmax/DZ)+1)
    d={}
    for key in cols[1:]:
        v=data[key]
        assert np.all(np.isfinite(v)) and np.all(v>0)
        d[key]=np.exp(PchipInterpolator(z0,np.log(v))(z))
    d['z']=z
    rho=d['Density_kgm3'];mu=d['DynamicViscosity_ChapmanEnskog_Wilke_Pa_s']
    lam=d['MeanFreePath_ViscosityBased_m'];a=d['SpeedOfSound_ms']
    d['u_ma']=MA_CRIT*a
    d['u_re_min']=RE_MIN*mu/(rho*L)
    d['u_re_trans']=RE_TRANS*mu/(rho*L)
    iz=np.where((lam[:-1]-KN_CRIT*L)*(lam[1:]-KN_CRIT*L)<0)[0]
    assert len(iz)==1, f'{planet} threshold crossings: {iz}'
    lnlam=PchipInterpolator(z0,np.log(data['MeanFreePath_ViscosityBased_m']))
    d['z_kn']=float(brentq(lambda zz:lnlam(zz)-np.log(KN_CRIT*L),z[iz[0]],z[iz[0]+1]))
    # Sanity: interpolated values preserve each original 1km GRAM-derived input.
    for key in cols[1:]:
        np.testing.assert_allclose(d[key][::round(1/DZ)],data[key][:zmax+1],rtol=1e-12,atol=1e-15)
    return d


def add_region(ax,z,lower,upper,eligible,color):
    lo=np.maximum(lower,UMIN)
    hi=np.minimum(upper,UMAX)
    good=eligible & (hi>lo)
    ax.fill_betweenx(z,lo,hi,where=good,interpolate=True,
                     color=color,edgecolor='none',linewidth=0,zorder=2)


def draw(ax,title,data,zmax,tag):
    z=data['z'];zk=data['z_kn'];cont=z<=zk
    uma=data['u_ma'];umin=data['u_re_min'];ut=data['u_re_trans']
    # All five model categories drawn as *vector polygons* defined by the equations.
    ax.axhspan(zk,zmax,facecolor=COLORS['rarefied'],edgecolor='none',zorder=1)
    lft=np.full_like(z,UMIN);rgt=np.full_like(z,UMAX)
    add_region(ax,z,lft,np.minimum(umin,uma),cont,COLORS['incompressible'])
    add_region(ax,z,umin,np.minimum(ut,uma),cont,COLORS['blasius'])
    add_region(ax,z,ut,uma,cont,COLORS['boundary-layer'])
    add_region(ax,z,uma,rgt,cont,COLORS['compressible'])
    # Subtle boundary lines help when printed at reduced size.
    for curve,mask,weight in [
        (uma,cont,0.75),
        (umin,cont & (umin<uma),0.60),
        (ut,cont & (ut<uma),0.60),
    ]:
        visible=mask & (curve>=UMIN) & (curve<=UMAX)
        ax.plot(curve[visible],z[visible],lw=weight,color='#343944',zorder=4)
    ax.axhline(zk,ls=(0,(4,2.5)),lw=1.1,color='#232A35',zorder=5)
    ax.set_xscale('log');ax.set_xlim(UMIN,UMAX);ax.set_ylim(0,zmax)
    ax.set_yticks(np.arange(0,zmax+1,20 if zmax==100 else 100))
    ax.set_xticks([.1,1,10,100,1000])
    ax.set_xticklabels([r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$'])
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*0.1,numticks=40))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='both',which='major',labelsize=9,length=3.8,width=.7,color='#4C5968')
    ax.tick_params(axis='x',which='minor',length=2,width=.45,color='#7A8995')
    ax.set_ylabel('Altitude (km)',fontsize=9.5,labelpad=6)
    ax.text(0.0,1.035,f'({tag})  {title}',transform=ax.transAxes,
            ha='left',va='bottom',fontsize=11.2,fontweight='bold',color='#1D2938')
    # Top right readable model-dependent Kn limit; label immediately above dash.
    ax.text(.99,zk/zmax+.035,r'$$'.replace('\u0001',''),transform=ax.transAxes) if False else None
    ax.text(.985,zk/zmax+0.025,fr'$\mathrm{{Kn}}=0.01$ at {zk:.1f} km',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=8.0,color='#202B39',
            bbox=dict(boxstyle='round,pad=.12',facecolor='white',edgecolor='none',alpha=.90),zorder=7)
    for spine in ax.spines.values():spine.set_color('#627080');spine.set_linewidth(.55)
    ax.grid(False)


def main():
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans',
                         'svg.fonttype':'none','axes.unicode_minus':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(8.27,11.69),facecolor='white')  # A4 portrait
    # Figure caption is intentionally NOT embedded; add a Word caption underneath.
    fig.text(.105,.967,'Planetary aerodynamic model-validity maps',ha='left',va='top',
             fontsize=15.8,fontweight='bold',color='#1D2938')
    fig.text(.105,.942,
             r'Nominal classification   |   $L=1\ \mathrm{m}$   |   $0.1\leq U\leq1000\ \mathrm{m\,s^{-1}}$',
             ha='left',va='top',fontsize=9.4,color='#526171')
    from matplotlib.gridspec import GridSpec
    gs=fig.add_gridspec(3,1,left=.12,right=.968,top=.900,bottom=.158,hspace=.31)
    axes=[fig.add_subplot(gs[i]) for i in range(3)]
    metadata={
      'reference':'NASA GRAM-derived atmospheres, composition-sensitive Chapman-Enskog/Wilke viscosity',
      'thresholds':{'L_m':L,'Kn':KN_CRIT,'Ma':MA_CRIT,'Re_min':RE_MIN,'Re_transition':RE_TRANS},
      'source_CSV_files': ['earth3.csv','mars3.csv','titan3.csv'],
      'source_altitude_spacing_km':1,'display_altitude_spacing_km':DZ,
      'method':'positive data log-PCHIP interpolated on display grid, analytical boundary polygons; no image generation or classification smoothing',
    }
    for ax,(name,key,zmax),tag in zip(axes,PLANETS,['a','b','c']):
        dat=read_profile(key,zmax)
        draw(ax,name,dat,zmax,tag)
        metadata[key]={'display_range_km':[0,zmax], 'Kn0p01_boundary_km':dat['z_kn'],
                       'local_Mach_boundary_ms':[float(dat['u_ma'].min()),float(dat['u_ma'].max())]}
    axes[-1].set_xlabel(r'Freestream velocity, $U$ (m s$^{-1}$)',fontsize=10.5,labelpad=7)
    legend=[Patch(facecolor=COLORS[name],edgecolor='none',label=label) for name,label in [
        ('incompressible','Incompressible Navier–Stokes'),('blasius','Blasius'),
        ('boundary-layer','Boundary-layer model'),('compressible','Compressible Navier–Stokes'),
        ('rarefied','Rarefied / slip-corrected')]]
    fig.legend(handles=legend,loc='lower center',ncol=3,bbox_to_anchor=(.52,.071),
               fontsize=9.1,frameon=False,columnspacing=1.8,labelspacing=.9,handlelength=1.5,handleheight=.85)
    fig.text(.115,.042,r'Dashed line: conventional no-slip continuum threshold  ($\mathrm{Kn}=0.01$).',
             ha='left',va='center',color='#526171',fontsize=8.4)
    fig.text(.115,.026,'Representative GRAM atmospheres; smooth curves interpolate original 1 km profiles.',
             ha='left',va='center',color='#526171',fontsize=8.1)
    fig.savefig(ROOT/(STEM+'.pdf'),format='pdf')
    fig.savefig(ROOT/(STEM+'.svg'),format='svg')
    fig.savefig(ROOT/(STEM+'.png'),format='png',dpi=260)
    plt.close(fig)
    (ROOT/'figure_provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))
    for ext in ['pdf','svg','png']:
        p=ROOT/(STEM+'.'+ext)
        print(p,p.stat().st_size)

if __name__=='__main__':main()
