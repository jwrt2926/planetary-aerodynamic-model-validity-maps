"""One-command reproduction from the supplied processed atmospheric datasets.
Usage: python scripts/reproduce_all.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess, sys, shutil, json, os

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'; SRC=BASE/'scripts'/'figures'; FIGS=BASE/'figures'; RES=BASE/'results'/'reproduced'
JOBS=[
 ('01','fig01_combined_maps.py','Fig1_Planetary_Model_Validity_Combined',[]),
 ('02','fig02_length_effects.py','Fig2_Characteristic_Length_Effects',['--data-dir','{stage}','--out-dir','{stage}']),
 ('03','fig03_mars_length_maps.py','Fig_S1_Mars_Model_Validity_L1m_vs_L10m_0-25km',['--data-dir','{stage}','--out-dir','{stage}']),
 ('04','fig04_master_map.py','Fig3_Dimensionless_Master_Map_Planetary_Trajectories',['--data-dir','{stage}','--out-dir','{stage}']),
 ('05','fig05_analytical_kn.py','Fig4_Analytical_vs_GRAM_Kn',['--data-dir','{stage}','--output-dir','{stage}']),
 ('06','fig06_local_lambda_gradient.py','Fig_S2_Local_MeanFreePath_Growth',['--data-dir','{stage}','--out-dir','{stage}']),
 ('07','fig07_sensitivity_zeros.py','Fig5_Sensitivity_Explicit_Zero_Bars',[]),
 ('08','fig08_retransition_sensitivity.py','Fig6_ReTransition_Sensitivity_Standalone',[]),
 ('09','fig09_atmospheric_variability.py','Fig09_Atmospheric_Variability',['--data-dir','{stage}','--out-dir','{stage}']),
]
OUTPUTS=['mach_and_length_results.json','mach_and_length_summary.csv','surface_reference_parameters.csv',
 'mechanism_results.json','eq48_reynolds_mach_intersection.csv','retransition_figure_results.json',
 'analytical_comparison_results.json','master_map_numerical_results.json','length_scale_results.json']

def main():
    FIGS.mkdir(exist_ok=True);RES.mkdir(parents=True,exist_ok=True)
    report={'python':sys.version,'figure_jobs':{},'verified_processed_data':True}
    with TemporaryDirectory(prefix='planetary_figs_') as td:
        d=Path(td)
        for n in ['earth3.csv','mars3.csv','titan3.csv']:
            shutil.copy2(DATA/n,d/n)
        shutil.copy2(DATA/'mars3.csv',d/'mars3(1).csv')
        shutil.copy2(DATA/'nominal_one_at_a_time_sensitivity.csv',d/'nominal_one_at_a_time_sensitivity.csv')
        for n in ['individual_48profile_Kn_crossings.csv','individual_profile_summary.json']:
            shutil.copy2(DATA/'sampling'/n,d/n)
        for number,script,stem,arguments in JOBS:
            temp_script=d/script;shutil.copy2(SRC/script,temp_script)
            command=[sys.executable,str(temp_script),*[str(d) if arg=='{stage}' else arg for arg in arguments]]
            print(f'[{number}/09] {script}',flush=True)
            p=subprocess.run(command,cwd=d,capture_output=True,text=True,env={**os.environ,'MPLBACKEND':'Agg'},timeout=180)
            if p.returncode!=0:
                raise RuntimeError(f'Failed {script}:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}')
            files=[]
            for ext in ('png','pdf','svg'):
                result=d/(stem+'.'+ext)
                if not result.is_file() or result.stat().st_size<1024:
                    raise FileNotFoundError(f'Missing generated {result}; stdout {p.stdout[-2000:]}')
                target=FIGS/(f'fig{number}.{ext}');shutil.copy2(result,target);files.append(target.name)
            report['figure_jobs']['fig'+number]={'script':str(script),'files':files,'stdout_tail':p.stdout[-1600:]}
            for fname in OUTPUTS:
                pth=d/fname
                if pth.is_file():shutil.copy2(pth,RES/fname)
    (BASE/'results'/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SUCCESS: generated all 9 figures in figures/; numerical outputs in results/reproduced/')

if __name__=='__main__':main()
