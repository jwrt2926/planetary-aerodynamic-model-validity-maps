"""Earth-GRAM 2024b: reproducible 48-case sampled global reference mean.

Requires GRAM Suite 2.1.1 and four monthly MERRA2 All Mean binary files
in GRAM/Earth/data/MERRA2data/All Mean/MERRA2All_{01,04,07,10}.bin.

Usage: python earth_global_average_pipeline.py --gram-root '/path/to/GRAM' --out-dir '/mnt/data'
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

PHYSICAL = [
    'Temperature_K','Pressure_Pa','Density_kgm3','SpeedOfSound_ms',
    'SpecificGasConstant_JkgK','SpecificHeatRatio','AverageMolecularWeight',
    'TotalNumberDensity_m3','CompressibilityFactor',
    'PressureScaleHeight_km','DensityScaleHeight_km'
]
SPECIES=['Ar','CO2','CO','N2','O2','He','H','CH4','N','O','O3','N2O','H2O']
VALUES=PHYSICAL+[s+'nd_m3' for s in SPECIES]
N=2201

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gram-root',required=True,help='Path to separately installed NASA GRAM directory')
    parser.add_argument('--out-dir',default='.',help='Directory for generated reference CSV and metadata')
    args=parser.parse_args()
    gram=Path(args.gram_root); outdir=Path(args.out_dir)
    sample=gram/'Earth'/'sample_inputs'; executable=gram/'Linux'/'EarthGRAM'
    template=(sample/'merra2_input.txt').read_text()
    binary_root=gram/'Earth'/'data'/'MERRA2data'/'All Mean'
    months=[1,4,7,10]
    missing=[str(binary_root/f'MERRA2All_{m:02d}.bin') for m in months if not (binary_root/f'MERRA2All_{m:02d}.bin').is_file()]
    if missing:
        raise FileNotFoundError('Missing Earth-GRAM All Mean binary data (not included in main GRAM ZIP):\n'+'\n'.join(missing))
    if not executable.exists(): raise FileNotFoundError(f'Missing executable: {executable}')
    def run(options):
        tag='earth_global_tmp'; contents=template
        for k,v in {**options, 'ListFileName':tag+'_LIST','ColumnFileName':tag+'_OUTPUT'}.items():
            pattern=r'(?mi)^(\s*'+re.escape(k)+r'\s*=\s*)[^\r\n]*'
            if not re.search(pattern,contents):raise ValueError('Cannot find setting '+k)
            contents=re.sub(pattern,lambda m:m.group(1)+str(v),contents,count=1)
        inp=sample/(tag+'_input.txt');inp.write_text(contents)
        try:
            proc=subprocess.run([str(executable),'-file',inp.name],cwd=sample,capture_output=True,text=True,timeout=120)
            if proc.returncode or 'unrecoverable error' in proc.stdout.lower():
                raise RuntimeError(proc.stdout[-4500:]+proc.stderr[-2000:])
            with (sample/(tag+'_OUTPUT.csv')).open(newline='') as fp:
                rows=list(csv.DictReader(fp))
            return rows
        finally:
            for suffix in ('_input.txt','_OUTPUT.csv','_LIST.md'):(sample/(tag+suffix)).unlink(missing_ok=True)
    latnodes,latweights=np.polynomial.legendre.leggauss(3)
    latitudes=np.degrees(np.arcsin(latnodes))
    longitudes=[0,90,180,270]
    cases=[]; sites=[]
    accum=np.zeros((N,len(VALUES)),dtype=np.float64)
    accum2=np.zeros((N,3),dtype=np.float64)
    clipped=0
    for lat,latweight in zip(latitudes,latweights):
        for lon in longitudes:
            site_base={
                'InitialLatitude':f'{lat:.9f}', 'InitialLongitude':str(lon),
                'InitialHeight':'0.0','NumberOfPositions':'1',
                'DeltaHeight':'0.0','DeltaLatitude':'0.0','DeltaLongitude':'0.0','DeltaTime':'0.0',
                'M2Hour':'9','M2MinimumLatitude':'-90','M2MaximumLatitude':'90',
                'M2MinimumLongitude':'0','M2MaximumLongitude':'360',
                'UseNCEP':'0','UseRRA':'0',
                'RandomPerturbationScale':'0.0',
                'HorizontalWindPerturbationScale':'0.0',
                'VerticalWindPerturbationScale':'0.0',
                'NumberOfMonteCarloRuns':'1','ExtraPrecision':'1','UseLegacyOutputs':'0',
                'AP':'10.0','DailyF10':'110.0','MeanF10':'110.0',
                'Month':'1','Day':'15','Year':'2020','Hour':'12','Minute':'0','Seconds':'0.0'
            }
            probe=run(site_base)
            if len(probe)!=1:raise ValueError('Surface probe did not return exactly one point')
            surface=float(probe[0]['SurfaceHeight_km'])
            wsite=float(latweight)/2/len(longitudes)
            sites.append({'latitude_deg':float(lat),'longitude_deg':lon,'surface_height_km':surface,'weight':wsite})
            print(f'Site {len(sites)}/12: lat {lat:.3f}, lon {lon}°; surface height {surface:.3f} km',flush=True)
            for month in months:
                opts={**site_base,'InitialHeight':f'{surface:.6f}',
                      'NumberOfPositions':str(N),'DeltaHeight':'1.0','Month':str(month)}
                rows=run(opts)
                if len(rows)!=N:raise ValueError(f'Wrong row count {len(rows)} at {lat}/{lon}/{month}')
                heights=np.array([float(r['Height_km']) for r in rows])
                if np.max(np.abs(heights-(surface+np.arange(N))))>0.02:
                    raise ValueError('Output heights do not match surface+0..2200 km')
                data=np.array([[float(row[k]) for k in VALUES] for row in rows],dtype=np.float64)
                if not np.isfinite(data).all():raise ValueError('Non-finite atmospheric results')
                if np.any(data[:,0:3]<=0):raise ValueError('Non-positive T,p,rho values')
                negatives=data[:,len(PHYSICAL):]<0
                clipped+=int(negatives.sum())
                data[:,len(PHYSICAL):]=np.maximum(data[:,len(PHYSICAL):],0)
                weight=wsite/len(months)
                accum+=weight*data
                accum2+=weight*data[:,:3]**2
                cases.append({'latitude_deg':float(lat),'longitude_deg':lon,'month':month,
                              'day':15,'year':2020,'UTC_hour':12,'surface_height_km':surface,'weight':weight})
    assert len(cases)==48 and abs(sum(c['weight'] for c in cases)-1)<1e-12
    std=np.sqrt(np.maximum(accum2-accum[:,:3]**2,0))
    avg_surface=sum(s['weight']*s['surface_height_km'] for s in sites)
    basename='Earth_GRAM_global_reference_mean_0-2200km_1km'
    path=outdir/(basename+'.csv');outdir.mkdir(parents=True,exist_ok=True)
    cols=['Height_km']+PHYSICAL+[c for sp in SPECIES for c in (sp+'mole_pct',sp+'nd_m3')]+[
        'AltitudeReference','MeanSurfaceElevation_km','TemperatureStd_K','PressureStd_Pa','DensityStd_kgm3','NumberOfSamples']
    with path.open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for z in range(N):
            rec={'Height_km':z,'AltitudeReference':'above_local_terrain_surface',
                 'MeanSurfaceElevation_km':f'{avg_surface:.9g}',
                 'TemperatureStd_K':f'{std[z,0]:.9g}','PressureStd_Pa':f'{std[z,1]:.9g}',
                 'DensityStd_kgm3':f'{std[z,2]:.9g}','NumberOfSamples':48}
            rec.update({key:f'{accum[z,j]:.10g}' for j,key in enumerate(VALUES[:len(PHYSICAL)])})
            nsum=sum(accum[z,len(PHYSICAL)+j] for j in range(len(SPECIES)))
            for j,sp in enumerate(SPECIES):
                ni=accum[z,len(PHYSICAL)+j]
                rec[sp+'nd_m3']=f'{ni:.10g}'
                rec[sp+'mole_pct']=f'{100*ni/nsum:.10g}' if nsum>0 else ''
            writer.writerow(rec)
    metadata={
      'source':'Earth-GRAM 2024b in GRAM Suite 2.1.1 with NASA MERRA-2 All Mean climatology',
      'description':'48-case area-weighted space / equal-month sampled reference, not exact annual/global climatological mean',
      'points':N, 'altitude':'km above local terrain surface of each sample site',
      'method':'3-point Gauss-Legendre in sin(latitude) x 4 longitude x 4 seasonal months; 48 samples',
      'latitudes_deg':latitudes.tolist(),'latitude_weights':(latweights/2).tolist(),
      'longitudes_deg':longitudes,'months':months,'M2Hour':9,
      'thermosphere_model':'MET (Jacchia), thermosphere model 1',
      'fixed_solar_and_geomagnetic_parameters':{'DailyF10':110.0,'MeanF10':110.0,'AP':10.0},
      'stochastic_perturbation_scale':0,
      'limitations':['Only 4 monthly profiles, rather than all 12 monthly means.','No stochastic perturbations.',
        'Solar/geomagnetic conditions fixed rather than sampled.','Thermospheric model has limited validation at extreme altitudes.',
        'Using local surface as height zero means geometric altitudes differ by site.',
        'Means of gas properties do not necessarily satisfy all nonlinear mixture state equations exactly.',
        'Mean regime classification differs from regime classification of mean atmosphere.'],
      'negative_number_density_values_clipped':clipped,'sites':sites,'samples':cases,
      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    meta_file=outdir/(basename+'_metadata.json');meta_file.write_text(json.dumps(metadata,indent=2))
    print(f'SUCCESS: {path}, {meta_file}; {len(cases)} cases',flush=True)

if __name__=='__main__':main()
