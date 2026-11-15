"""Plot existing independently calculated GRAM 48-profile Kn crossings.
No GRAM install required for this visualization; regenerate source cases via
scripts/gram/individual_profile_variability.py with NASA GRAM Suite.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path(__file__).parent);p.add_argument('--out-dir',type=Path,default=Path(__file__).parent);a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.data_dir/'individual_48profile_Kn_crossings.csv')
    stats=json.loads((a.data_dir/'individual_profile_summary.json').read_text())
    assert len(d)==96 and set(d.planet)=={'Earth','Mars'}
    for st in stats:
        x=d[d.planet==st['planet']]
        w=x.weight.to_numpy(float);v=x.continuum_crossing_km.to_numpy(float)
        w=w/w.sum(); assert len(x)==48
        assert abs(float(w@v)-st['weighted_mean_km'])<1e-8
        assert abs(float(np.sqrt(w@((v-w@v)**2)))-st['weighted_std_km'])<1e-8
    fig,axs=plt.subplots(2,1,figsize=(7.1,5.9),sharex=False)
    rng=np.random.default_rng(1402)
    for ax,st in zip(axs,stats):
        planet=st['planet']; rows=d[d.planet==planet]; xs=rows.continuum_crossing_km.to_numpy(float);ws=rows.weight.to_numpy(float)
        jitter=rng.normal(0,.115,len(xs))
        ax.scatter(xs,jitter,s=210*ws/ws.max()+12,alpha=.59,color='#42749D' if planet=='Earth' else '#CB7F52',edgecolor='white',lw=.45)
        ax.axvline(st['weighted_mean_km'],color='#264859',lw=1.7)
        ax.axvline(st['mean_profile_boundary_km'],color='#C0503F',lw=1.7,linestyle='--')
        ax.axvspan(st['weighted_mean_km']-st['weighted_std_km'],st['weighted_mean_km']+st['weighted_std_km'],color='#7D9BA9',alpha=.13,zorder=0)
        ax.set_yticks([]);ax.set_title(f"{planet}: mean {st['weighted_mean_km']:.2f} km, SD {st['weighted_std_km']:.2f} km",loc='left',fontsize=11.5,fontweight='semibold')
        ax.set_xlabel('Altitude of Kn = 0.01 boundary (km)',fontsize=10)
        ax.grid(axis='x',alpha=.19);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False);ax.set_ylim(-.4,.4)
        span=xs.max()-xs.min();ax.set_xlim(xs.min()-max(2,span*.1),xs.max()+max(2,span*.1))
    fig.legend(handles=[Line2D([],[],color='#264859',lw=1.7,label='Weighted mean of individual crossings'),Line2D([],[],color='#C0503F',ls='--',lw=1.7,label='Crossing from averaged atmosphere')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,fontsize=9,frameon=False)
    fig.suptitle('Earth and Mars: 48-profile atmospheric variability',fontsize=13.4,fontweight='semibold',y=.985)
    fig.subplots_adjust(left=.09,right=.97,top=.88,bottom=.16,hspace=.48)
    for ext in ('pdf','png','svg'):fig.savefig(a.out_dir/f'Fig09_Atmospheric_Variability.{ext}',dpi=260,facecolor='white')
    plt.close(fig);print('Fig09: 96 individual profiles plotted; weighted statistics match archived values')

if __name__=='__main__':main()
