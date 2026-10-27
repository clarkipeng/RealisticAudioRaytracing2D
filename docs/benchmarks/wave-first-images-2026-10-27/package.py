"""Compare current canopy transport with the same BEM reference and archived baseline."""
import argparse,importlib.util,json,re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
earlier=ROOT/'docs/benchmarks/wave-review-2026-10-27'
spec=importlib.util.spec_from_file_location('wave_package',earlier/'package.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def main(reference,measured,baseline,output):
    base.main(reference,measured,output,125,.25,.35)
    pattern=r'(<script id="comparison-data" type="application/json">)(.*?)(</script>)'
    html=(output/'index.html').read_text()
    data=json.loads(re.search(pattern,html,re.S)[2])
    previous=json.loads((earlier/'canopy-report.json').read_text())
    config=json.loads((baseline/'scene.json').read_text())
    for key in ['source','walls','points','gridIndices','x','y','speedOfSound','sampleRate','fftSize','maxBounces','maximumDelaySeconds','rays','seeds']:
        assert config[key]==data['config'][key],f'Baseline physical/settings mismatch: {key}'
    assert base.digest(baseline/'renderer.csv')==previous['rendererCsvSha256']
    assert previous['referenceSha256']==data['report']['referenceSha256']
    rows=np.genfromtxt(baseline/'renderer.csv',delimiter=',',names=True)
    target=np.array(data['reference']['real'])+1j*np.array(data['reference']['imag'])
    fields={};changes=[]
    for case in previous['cases']:
        selected=rows[(rows['rays']==case['rays'])&(rows['seed']==case['seed'])&(rows['diffraction']==case['diffraction'])]
        selected=np.sort(selected,order='point')
        assert np.array_equal(selected['point'],np.arange(len(target)))
        assert np.all(selected['hz']==125)
        value=selected['real']+1j*selected['imag']
        error=float(np.linalg.norm(value-target)/np.linalg.norm(target))
        assert abs(error-case['complexRelativeL2'])<1e-12
        fields[case['key']]=dict(real=value.real.tolist(),imag=value.imag.tolist())
        current=next(c for c in data['report']['cases'] if c['key']==case['key'])
        changes.append(dict(key=case['key'],before=error,updated=current['complexRelativeL2'],
            beforeShadow=case['shadowComplexRelativeL2'],updatedShadow=current['shadowComplexRelativeL2']))
    data['previous']=dict(fields=fields,report=previous)
    data['initialMode']='level'
    data['links']={
        'blocks-link':dict(href='../wave-ground-truth-comparison/index.html',title='Earlier 3D benchmark'),
        'canopy-link':dict(href='index.html',title='Updated canopy vs wave'),
        'prototype-link':dict(href='../runtime-rain-integration/index.html',title='Rain listening comparison')}
    data['report']['comparison']=dict(baselineCsvSha256=previous['rendererCsvSha256'],cases=changes,
        scope='Same independent wave reference and receiver grid; before/after first-image integration. No gain fitting. This is not the broadband rain mix or the first-image-only PCM control.')
    (output/'report.json').write_text(json.dumps(data['report'],indent=2)+'\n')
    (output/'index.html').write_text(re.sub(pattern,lambda m:m[1]+json.dumps(data).replace('<','\\u003c')+m[3],html,flags=re.S))
    print(json.dumps(changes,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','measured','baseline','output']:p.add_argument(name,type=Path)
    a=p.parse_args();main(a.reference,a.measured,a.baseline,a.output)
