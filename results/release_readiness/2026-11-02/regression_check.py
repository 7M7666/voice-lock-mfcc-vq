
import ast, hashlib, json, subprocess, sys
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0, 'code/python')
import gui, srdp
from phrase_content_experiments import formal_profile
out=Path('tmp_release/readiness-20261102')
out.mkdir(parents=True, exist_ok=True)

def function(src, name):
    return next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
old=subprocess.check_output(['git','show','eb8b772:code/python/gui.py']).decode('utf8')
new=Path('code/python/gui.py').read_text(encoding='utf8')
a=function(old,'predict_with_session_model');b=function(new,'predict_with_session_model_baseline');a.name=b.name
assert ast.dump(a)==ast.dump(b), 'A implementation changed'
for name in ('assess_enrollment_take','build_session_enrollment_model','enrollment_consistency_check','enrollment_self_check','save_user_model','load_user_model'):
 assert ast.dump(function(old,name))==ast.dump(function(new,name)), name
assert subprocess.check_output(['git','diff','origin/master','--','code/python/srdp.py','code/python/speech_frames.py','code/python/dtw_paths.py'])==b''
base=srdp.load_model()
rows=srdp.read_metadata(Path('data/demo_audio/metadata_srdp.csv'))
queries=[r for r in rows if r['split']!='train']
trials=[]
for speaker in sorted(base.codebooks):
 model=formal_profile(base,speaker)
 for row in queries:
  path=Path(row['file_path'])
  a=gui.predict_with_session_model_baseline(path,model)
  e=gui.predict_with_session_model(path,model)
  assert e['decision']=='REJECT' or a['decision']!='REJECT'
  for key in ('voice_distance','voice_score','passphrase_distance','passphrase_score','voice_threshold','passphrase_threshold'):
   assert a.get(key)==e.get(key), key
  trials.append(dict(target=speaker,file_path=row['file_path'],A=a['decision'],E=e['decision'],asr=e.get('asr_text','')))
 print('Compared',speaker,len(queries),'account claims',flush=True)
# Genuine process restart; a legacy profile omits optional post-v1 fields.
model=formal_profile(base,'S01')
path=out/'legacy-profile.npz';gui.save_user_model(model,path)
import numpy as np
with np.load(path,allow_pickle=False) as data:
 values={name:data[name] for name in data.files if name not in ('voice_metric','background_speakers','decision_mode','fusion_voice_weight','fusion_threshold','calibration_mode','vad_mode','cepstral_normalization','dtw_max_run','voice_guard','passphrase_guard') and not name.startswith('background_')}
np.savez_compressed(path,**values)
model=gui.load_user_model(path)
wav=Path('data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav')
expected=gui.predict_with_session_model(wav,model)
(out/'restart-expected.json').write_text(json.dumps(expected,ensure_ascii=False),encoding='utf8')
code="import sys,json;from pathlib import Path;sys.path.insert(0,'code/python');import gui;r=gui.predict_with_session_model(Path(sys.argv[2]),gui.load_user_model(Path(sys.argv[1])));print(json.dumps(r,ensure_ascii=False))"
actual=json.loads(subprocess.check_output([sys.executable,'-c',code,str(path),str(wav)],encoding='utf8'))
assert actual==expected, 'restart changed result'
report=dict(A_code_unchanged=True,enrollment_code_unchanged=True,srdp_unchanged=True,formal_wavs=len(rows),claims=len(trials),A_accepts=sum(x['A']!='REJECT' for x in trials),E_accepts=sum(x['E']!='REJECT' for x in trials),E_subset_of_A=True,real_asr=True,legacy_profile_restart_exact_match=True,trials=trials)
(out/'regression.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='trials'},ensure_ascii=False),flush=True)
