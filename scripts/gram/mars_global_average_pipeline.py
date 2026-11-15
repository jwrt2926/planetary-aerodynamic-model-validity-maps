"""Reproduce a sampled Mars-GRAM global/Mars-year reference mean (48 cases)."""
import csv,json,re,hashlib,subprocess,argparse
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
AP=argparse.ArgumentParser(description='Sample 48 Mars-GRAM profiles using an installed GRAM Suite')
AP.add_argument('--gram-root',required=True,type=Path,help='Path to separately installed NASA GRAM directory')
AP.add_argument('--out-dir',type=Path,default=Path.cwd())
AP.add_argument('--input-template',type=Path,default=Path(__file__).resolve().parent/'Mars_GRAM_final_0-1000km_1km_input.txt')
ARGS=AP.parse_args()
G=ARGS.gram_root;S=G/'Mars/sample_inputs';EXEC=G/'Linux/MarsGRAM'
TEMPLATE=ARGS.input_template.read_text()
ARGS.out_dir.mkdir(parents=True,exist_ok=True)
latnode,latweight=np.polynomial.legendre.leggauss(3)
latitudes=np.degrees(np.arcsin(latnode));lons=[0,90,180,270]
dates=[datetime(2020,3,25)+timedelta(days=x) for x in [0,172,344,516]]
HOUR_UTC=12;N=2201
fields=['Temperature_K','Pressure_Pa','Density_kgm3','SpeedOfSound_ms','SpecificGasConstant_JkgK','SpecificHeatRatio','AverageMolecularWeight','TotalNumberDensity_m3','CompressibilityFactor','PressureScaleHeight_km','DensityScaleHeight_km']
species=['CO2','N2','Ar','CO','O2','O','H','He','H2','H2O']
keys=fields+[s+'nd_m3' for s in species]
acc=np.zeros((N,len(keys)));acc2=np.zeros((N,3));samplelist=[];sites=[];clip_count=0;max_clip=0

def run(opts):
 tag='global_mean_tmp';t=TEMPLATE
 for k,v in {**opts,'ListFileName':tag+'_LIST','ColumnFileName':tag+'_OUTPUT'}.items():
  p=r'(?mi)^(\s*'+re.escape(k)+r'\s*=\s*)[^\r\n]*'
  if not re.search(p,t):raise RuntimeError('Missing '+k)
  t=re.sub(p,lambda m:m.group(1)+str(v),t,count=1)
 f=S/(tag+'_input.txt');f.write_text(t)
 proc=subprocess.run([str(EXEC),'-file',f.name],cwd=S,capture_output=True,text=True,timeout=40)
 if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
 output=S/(tag+'_OUTPUT.csv')
 df=pd.read_csv(output)
 for suffix in ('_input.txt','_OUTPUT.csv','_LIST.md'):(S/(tag+suffix)).unlink(missing_ok=True)
 return df
for ilat,(la,wla) in enumerate(zip(latitudes,latweight)):
 for lon in lons:
  loc={'InitialLatitude':f'{la:.9f}','InitialLongitude':str(lon)}
  base=run({**loc,'NumberOfPositions':'1','InitialHeight':'0.0'})
  surface=float(base['SurfaceHeight_km'].iloc[0]);wsite=float(wla)/2/len(lons)
  sites.append({'latitude_deg':float(la),'longitude_deg':lon,'surface_MOLA_km':surface,'weight':wsite})
  print('Site',len(sites),'of',len(latitudes)*len(lons),'lat',round(float(la),3),'lon',lon,flush=True)
  for date in dates:
   opts={**loc,'InitialHeight':f'{surface:.4f}','NumberOfPositions':str(N),
      'DeltaHeight':'1.0','DeltaLatitude':'0.0','DeltaLongitude':'0.0','DeltaTime':'0.0',
      'Month':str(date.month),'Day':str(date.day),'Year':str(date.year),
      'Hour':str(HOUR_UTC),'Minute':'0','Seconds':'0.0',
      'DensityPerturbationScale':'0.0','EWWindPerturbationScale':'0.0','NSWindPerturbationScale':'0.0',
      'ComputeMinMax':'0','F107':'68.0','MGCMConstantDustLevel':'0.3','MapYear':'0',
      'StormIntensity':'0.0','ExtraPrecision':'1','UseLegacyOutputs':'0',
      'OffsetModel':'2','IsMolaHeights':'1'}
   df=run(opts)
   if len(df)!=N or np.max(abs(df['Height_km'].values-(np.arange(N)+surface)))>.011:raise ValueError('Invalid height')
   vals=df[keys].to_numpy(dtype=float)
   if not np.isfinite(vals).all() or np.any(vals[:,:4]<=0):raise ValueError('Invalid values')
   neg=vals[:,len(fields):]<0
   clip_count+=int(neg.sum());
   if neg.any():max_clip=max(max_clip,float(abs(vals[:,len(fields):][neg]).max()))
   vals[:,len(fields):]=np.maximum(vals[:,len(fields):],0)
   w=wsite/len(dates)
   acc+=vals*w;acc2+=vals[:,:3]**2*w
   samplelist.append({'latitude_deg':round(float(la),6),'longitude_deg':lon,
     'surface_MOLA_km':surface,'date_UTC':date.strftime('%Y-%m-%d'),'hour_UTC':HOUR_UTC,
     'solar_longitude_deg':float(df['LongitudeOfTheSun_deg'].iloc[0]),'weight':w})
assert len(samplelist)==48 and abs(sum(x['weight'] for x in samplelist)-1)<1e-12
std=np.sqrt(np.maximum(acc2-acc[:,:3]**2,0));mean_surface=sum(s['weight']*s['surface_MOLA_km'] for s in sites)
label='Mars_GRAM_global_reference_mean_0-2200km_1km'
path=ARGS.out_dir/(label+'.csv')
cols=['Height_km']+fields+[c for sp in species for c in (sp+'mole_pct',sp+'nd_m3')]+['AltitudeReference','MeanMOLASurfaceElevation_km','TemperatureStd_K','PressureStd_Pa','DensityStd_kgm3','NumberOfSamples']
with path.open('w',newline='') as fp:
 writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
 for z in range(N):
  rec={'Height_km':z,'AltitudeReference':'above_local_MOLA_surface',
       'MeanMOLASurfaceElevation_km':f'{mean_surface:.8g}',
       'TemperatureStd_K':f'{std[z,0]:.8g}','PressureStd_Pa':f'{std[z,1]:.8g}',
       'DensityStd_kgm3':f'{std[z,2]:.8g}','NumberOfSamples':48}
  rec.update({key:f'{acc[z,i]:.9g}' for i,key in enumerate(fields)})
  nt=sum(acc[z,len(fields)+j] for j in range(len(species)))
  for j,sp in enumerate(species):
   ni=acc[z,len(fields)+j];rec[sp+'nd_m3']=f'{ni:.9g}';rec[sp+'mole_pct']=f'{100*ni/nt:.9g}' if nt>0 else ''
  writer.writerow(rec)
meta={'source_model':'MarsGRAM 2024b / GRAM Suite 2.1.1','file':path.name,
 'description':'Sampled geographically area-weighted and Mars-year time-averaged reference atmosphere; not an exact climatological global mean.',
 'rows':N,'altitude_reference':'Height_km measured ABOVE LOCAL MOLA SURFACE for each site, not a common global areoid.',
 'latitude_method':'3-point Gauss-Legendre quadrature in sin(latitude), hence proper area weighting',
 'latitudes_deg':latitudes.tolist(),'latitude_weights':(latweight/2).tolist(),
 'longitudes_deg':lons,'dates_UTC':[d.strftime('%Y-%m-%d') for d in dates], 'UTC_hour':HOUR_UTC,
 'local_solar_time_coverage':'Four longitudes separated by 90 degrees yield approximately four different local solar times per date, but geography/diurnal cycle are not fully separated.',
 'samples':len(samplelist),'fixed_conditions':{'F107':68.0,'dust_optical_depth':0.3,'MapYear':0,'random_perturbation_scales':0,'dust_storm_intensity':0,'OffsetModel':2},
 'averaging':'Area-and-time-weighted arithmetic mean of 48 profiles; species mole percentages recomputed from averaged species number densities.',
 'species_negative_clips':clip_count,'max_negative_number_density_clipped_m3':max_clip,
 'limitations':['Discrete 48-scenario approximation, not a precise global annual mean.',
   'The averaging of atmospheric properties does not reproduce the mean of flow-regime classifications.',
   'Independent averages of thermodynamic outputs are not necessarily exactly equation-of-state consistent.',
   'Mars-GRAM thermospheric extension applies above about 170 km; 1000-2200 km values should be treated as extrapolative, not independently validated.',
   'Local ground elevation at each site defines zero altitude; this differs from areoid-height averaging.',
   'Fixed dust and solar activity, no stochastic perturbations; minimal coverage of diurnal effects.',
   'Tiny negative O number densities appearing in GRAM outputs at extreme altitudes were clipped to zero as numerical artifacts; see counts.'],
 'runs':samplelist,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
mp=ARGS.out_dir/(label+'_metadata.json');mp.write_text(json.dumps(meta,indent=2))
print('SUCCESS',path,mp,'size',path.stat().st_size,'neg',clip_count,'maxabs',max_clip,flush=True)
for z in [0,100,170,300,500,1000,1500,2200]:print(z,acc[z,0],acc[z,1],acc[z,2],flush=True)
