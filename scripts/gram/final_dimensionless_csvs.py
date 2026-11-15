"""Build plot-ready per-altitude CSVs, preserving earth2/mars2/titan2 exactly.

Reference Re, Ma and Kn are only an illustrative slice at U=100 m/s and L=1 m.
For an arbitrary U [m/s], L [m], calculate directly from the altitude factors:
  Re_L = ReL_factor_s_per_m2 * U * L
  Ma   = Ma_factor_s_per_m * U
  Kn   = Kn_factor_m / L

This avoids assigning a unique Re, Ma or Kn to altitude alone.

Blasius thickness is only a formal 5/sqrt(Re_L) thinness indicator on
continuum/incompressible/nominally pre-transition flat-plate conditions,
not a proof that Blasius theory is valid.
"""
from pathlib import Path
import csv, math, json

ROOT = Path(__file__).resolve().parent
U_REF = 100.0  # m/s, illustrative
L_REF = 1.0    # m, illustrative
KN_CRIT = 0.01
MA_CRIT = 0.3
RE_MIN = 1e4
RE_CRIT = 5e5

EXTRA_FIELDS = [
    'ReferenceVelocity_U_ms',
    'ReferenceLength_L_m',
    'ReL_factor_s_per_m2',
    'Ma_factor_s_per_m',
    'Kn_factor_m',
    'ReL_at_U100ms_L1m',
    'Ma_at_U100ms',
    'Kn_at_L1m',
    'BlasiusThinnessEstimate_delta99_over_L_at_U100ms_L1m',
    'Kn_from_ReMaGamma_at_U100ms_L1m',
    'Kn_ReMaGamma_IdentityDifference_pct',
    'SpeedOfSound_from_RmixGamma_ms',
    'SpeedOfSound_GRAM_vs_Reconstructed_pct',
]

def f(v):
    return format(v,'.12g')

def build(planet):
    original_path=ROOT / f'{planet}2.csv'
    output_path=ROOT / f'{planet}3.csv'
    with original_path.open('r', newline='', encoding='utf-8-sig') as fin:
        reader=csv.DictReader(fin)
        original_fields=list(reader.fieldnames)
        if set(EXTRA_FIELDS).intersection(original_fields):
            raise ValueError('New fields collide with prior fields')
        rows=[]
        for i, r in enumerate(reader):
            z=float(r['Height_km'])
            if z != i: raise AssertionError(f'Nonuniform altitude index {planet}: {i} {z}')
            rho=float(r['Density_kgm3'])
            mu=float(r['DynamicViscosity_ChapmanEnskog_Wilke_Pa_s'])
            lam=float(r['MeanFreePath_ViscosityBased_m'])
            a=float(r['SpeedOfSound_ms'])
            Rmix=float(r['GasConstant_FromGRAMComposition_JkgK'])
            gamma=float(r['SpecificHeatRatio'])
            T=float(r['Temperature_K'])
            if not all(math.isfinite(x) and x>0 for x in (rho,mu,lam,a,Rmix,gamma,T)):
                raise ValueError(f'Invalid base properties {planet} {z}')
            refactor=rho/mu
            mafactor=1/a
            knfactor=lam
            Re=refactor*U_REF*L_REF
            Ma=mafactor*U_REF
            Kn=knfactor/L_REF
            # On a continuum low-Mach flat-plate branch before nominal transition,
            # the 5/sqrt(Re_L) thinness expression is useful as a formal estimate.
            thinness=(5/math.sqrt(Re) if Kn < KN_CRIT and Ma < MA_CRIT and Re < RE_CRIT else None)
            knidentity=math.sqrt(math.pi*gamma/2)*Ma/Re
            soundpred=math.sqrt(gamma*Rmix*T)
            update={
                'ReferenceVelocity_U_ms':f(U_REF),
                'ReferenceLength_L_m':f(L_REF),
                'ReL_factor_s_per_m2':f(refactor),
                'Ma_factor_s_per_m':f(mafactor),
                'Kn_factor_m':f(knfactor),
                'ReL_at_U100ms_L1m':f(Re),
                'Ma_at_U100ms':f(Ma),
                'Kn_at_L1m':f(Kn),
                'BlasiusThinnessEstimate_delta99_over_L_at_U100ms_L1m':f(thinness) if thinness is not None else '',
                'Kn_from_ReMaGamma_at_U100ms_L1m':f(knidentity),
                'Kn_ReMaGamma_IdentityDifference_pct':f(100*(Kn/knidentity-1)),
                'SpeedOfSound_from_RmixGamma_ms':f(soundpred),
                'SpeedOfSound_GRAM_vs_Reconstructed_pct':f(100*(a/soundpred-1)),
            }
            r.update(update)
            rows.append(r)
        if len(rows)!=2201: raise AssertionError(f'row count {planet} is {len(rows)}')
    with output_path.open('w', newline='', encoding='utf-8') as fout:
        writer=csv.DictWriter(fout, fieldnames=original_fields + EXTRA_FIELDS)
        writer.writeheader();writer.writerows(rows)
    # Verify every original field is byte-for-byte preserved for its CSV string value
    with original_path.open('r',newline='',encoding='utf-8-sig') as a, output_path.open('r',newline='',encoding='utf-8-sig') as b:
        ra=csv.DictReader(a);rb=csv.DictReader(b)
        for idx,(u,v) in enumerate(zip(ra,rb)):
            for k in original_fields:
                if u[k]!=v[k]: raise AssertionError(f'Original changed: {planet}, {idx}, {k}')
    # Verify dynamically calculated values recompute at a non-reference U,L using coefficients.
    test_u=37.5; test_L=0.13
    for z in (0,30,80,150,200,500,1000,2200):
        r=rows[z]
        predicted=float(r['ReL_factor_s_per_m2'])*test_u*test_L
        baseline=float(r['Density_kgm3'])*test_u*test_L/float(r['DynamicViscosity_ChapmanEnskog_Wilke_Pa_s'])
        assert math.isclose(predicted,baseline,rel_tol=1e-10)
        assert math.isclose(float(r['Ma_factor_s_per_m'])*test_u,test_u/float(r['SpeedOfSound_ms']),rel_tol=1e-10)
        assert math.isclose(float(r['Kn_factor_m'])/test_L,float(r['MeanFreePath_ViscosityBased_m'])/test_L,rel_tol=1e-10)
    mis=[(float(r['Height_km']),float(r['IdealGasConsistency_p_over_rhoRT'])) for r in rows if abs(float(r['IdealGasConsistency_p_over_rhoRT'])-1)>0.1]
    print(f'{output_path.name}: {len(rows)} rows, {len(original_fields)} original + {len(EXTRA_FIELDS)} new columns, {output_path.stat().st_size:,} bytes')
    print(f'    source columns unchanged; >10% ideal-gas diagnostic deviations: {len(mis)}, first: {mis[0] if mis else "none"}')
    for z in (0,80,150,500):
        r=rows[z]; print(f'    z={z}: Re100mps1m={float(r["ReL_at_U100ms_L1m"]):.4g} Ma100mps={float(r["Ma_at_U100ms"]):.4g} Kn1m={float(r["Kn_at_L1m"]):.4g}')
    return {'rows':len(rows),'old_columns':len(original_fields),'new_columns':len(EXTRA_FIELDS),'filename':output_path.name}

if __name__=='__main__':
    summary={p:build(p) for p in ('earth','mars','titan')}
    notes={
        'purpose':'Altitude table and reusable coefficients for velocity/length sweeps. The reference dimensionless values are illustrative, NOT a unique Re/Ma/Kn for an altitude.',
        'reference_U_ms':U_REF,'reference_L_m':L_REF,
        'equations':{
            'Re_L':'ReL_factor_s_per_m2 * U_ms * L_m = Density_kgm3 * U_ms * L_m / DynamicViscosity_ChapmanEnskog_Wilke_Pa_s',
            'Ma':'Ma_factor_s_per_m * U_ms = U_ms / SpeedOfSound_ms',
            'Kn':'Kn_factor_m / L_m = MeanFreePath_ViscosityBased_m / L_m',
            'flat_plate_thinness':'5 / sqrt(Re_L); heuristic thinness only in continuum, incompressible, pre-transition flat-plate regime',
            'kn_re_ma_identity':'sqrt(pi*SpecificHeatRatio/2) * Ma/Re_L; exact only with ideal gas consistency and a=sqrt(gamma*Rmix*T) under adopted kinetic mean-free-path convention',
        },
        'nominal_thresholds':{'Kn':KN_CRIT,'Ma':MA_CRIT,'ReL_min':RE_MIN,'ReL_transition':RE_CRIT},
        'data_scope':'GRAM averaged Earth/Mars 48-case profile and nominal Titan profile. At very high altitudes datasets are model extrapolations and strongly rarefied for vehicle-scale L. Retain high-altitude rows but inspect validity and model limits before interpreting plots.',
        'existing_validation':'The previous Sutherland reference comparison columns are proxy benchmarks, NOT measured errors of a multicomponent mixture. GRAM original R and composition-derived R are both preserved; the Mars high-altitude discrepancy is unresolved.',
        'additional_dimensionless_variables':'No additional independent dimensionless number is required by the current Section 4 classifier. Prandtl number and others would require extra transport/thermal modelling and are out of scope.',
        'outputs':summary
    }
    (ROOT/'dimensionless_csv_methodology.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
