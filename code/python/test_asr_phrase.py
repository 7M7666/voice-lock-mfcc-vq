import os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
try: import sounddevice
except OSError:
 m=types.ModuleType('sounddevice');m.rec=m.get_stream=m.stop=lambda *a,**k:None;sys.modules['sounddevice']=m
import asr_phrase,gui

class ThreeLayerTest(unittest.TestCase):
 def test_exact_normalization_and_baseline_subset(self):
  baseline={'decision':'本人','voice_passed':'True','passphrase_passed':'True','rejection_stage':''}
  with patch.object(gui,'predict_with_session_model_baseline',return_value=baseline),patch.object(gui,'recognize_fixed_phrase',return_value=' 芝麻开门！'):
   result=gui.predict_with_session_model(Path('x'),None);self.assertEqual(result['decision'],'本人');self.assertEqual(result['asr_passed'],'True')
  with patch.object(gui,'predict_with_session_model_baseline',return_value=baseline),patch.object(gui,'recognize_fixed_phrase',return_value='芝麻开灯'):
   result=gui.predict_with_session_model(Path('x'),None);self.assertEqual(result['decision'],'REJECT');self.assertEqual(result['rejection_stage'],'asr');self.assertEqual(result['asr_passed'],'False')
  self.assertEqual(baseline['decision'],'本人')
 def test_baseline_reject_never_needs_asr(self):
  with patch.object(gui,'predict_with_session_model_baseline',return_value={'decision':'REJECT'}),patch.object(gui,'recognize_fixed_phrase') as asr:
   self.assertEqual(gui.predict_with_session_model(Path('x'),None)['decision'],'REJECT');asr.assert_not_called()
 def test_failure_never_becomes_accept(self):
  with patch.object(gui,'predict_with_session_model_baseline',return_value={'decision':'本人'}),patch.object(gui,'recognize_fixed_phrase',side_effect=RuntimeError('missing model')):
   with self.assertRaisesRegex(RuntimeError,'missing model'):gui.predict_with_session_model(Path('x'),None)
 def test_missing_and_invalid_model_are_errors(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'VOICE_LOCK_ASR_DIR':d}),patch.object(asr_phrase,'_recognizer',None):
   with self.assertRaisesRegex(RuntimeError,'模型缺失或校验不通过'):asr_phrase._load_recognizer()
 def test_frozen_model_path_uses_bundled_resources(self):
  with patch.dict(os.environ,{},clear=True),patch.object(asr_phrase.sys,'frozen',True,create=True),patch.object(asr_phrase,'resource_root',return_value=Path('/resources')):
   self.assertEqual(asr_phrase.model_directory(),Path('/resources/models/asr'))
 def test_previous_history_loads_and_new_fields_persist(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'history.csv';p.write_text('time_text,source,decision,pred_speaker,voice_score,passphrase_score,fusion_score,reject_reason,latency_ms\n2026,用户验证:本人,REJECT,本人,1,1,-,test,1\n',encoding='utf8')
   old=gui.load_history(p);self.assertEqual(old[0].asr_text,'');old[0].asr_text='芝麻开门';old[0].asr_passed='True';gui.save_history(old,p);self.assertEqual(gui.load_history(p)[0].asr_text,'芝麻开门')

 def test_corrupt_model_and_tokens_are_rejected_before_loading(self):
  for name in asr_phrase.EXPECTED_HASHES:
   with self.subTest(name=name), tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'VOICE_LOCK_ASR_DIR':d}), patch.object(asr_phrase,'_recognizer',None):
    for filename in asr_phrase.EXPECTED_HASHES: (Path(d)/filename).write_bytes(b'corrupt')
    with patch.object(asr_phrase,'_sha256',side_effect=lambda path: 'bad' if path.name == name else asr_phrase.EXPECTED_HASHES[path.name]):
     with self.assertRaises(asr_phrase.AsrError): asr_phrase._load_recognizer()

 def test_model_load_and_decode_errors_have_asr_type(self):
  import numpy as np
  fake=types.SimpleNamespace(OfflineRecognizer=types.SimpleNamespace(from_paraformer=lambda **kwargs: (_ for _ in ()).throw(ValueError('bad model'))))
  with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'VOICE_LOCK_ASR_DIR':d}), patch.object(asr_phrase,'_recognizer',None):
   for name in asr_phrase.EXPECTED_HASHES: (Path(d)/name).touch()
   with patch.object(asr_phrase,'_sha256',side_effect=lambda path: asr_phrase.EXPECTED_HASHES[path.name]), patch.dict(sys.modules,{'sherpa_onnx':fake}):
    with self.assertRaisesRegex(asr_phrase.AsrError,'加载失败'): asr_phrase._load_recognizer()
  fake=types.SimpleNamespace(create_stream=lambda: (_ for _ in ()).throw(ValueError('decode failed')))
  with patch('srdp.load_audio',return_value=(np.zeros(16000),16000)), patch.object(asr_phrase,'_load_recognizer',return_value=fake):
   with self.assertRaisesRegex(asr_phrase.AsrError,'执行失败'): asr_phrase.recognize_fixed_phrase(Path('unused.wav'))

 def test_empty_or_similar_phrase_never_passes(self):
  for text in ('', '芝麻关门', '芝麻开灯', '请芝麻开门', '芝麻开门开门'):
   with self.subTest(text=text), patch.object(gui,'predict_with_session_model_baseline',return_value={'decision':'本人'}), patch.object(gui,'recognize_fixed_phrase',return_value=text):
    self.assertEqual(gui.predict_with_session_model(Path('x'),None)['decision'],'REJECT')

 def test_source_model_paths_and_override(self):
  with patch.dict(os.environ,{},clear=True), patch.object(asr_phrase.sys,'frozen',False,create=True), patch.object(asr_phrase,'resource_root',return_value=Path('/project')):
   self.assertEqual(asr_phrase.model_directory(),Path('/project/models/asr'))
  with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'VOICE_LOCK_ASR_DIR':d}):
   self.assertEqual(asr_phrase.model_directory(),Path(d).resolve())

 def test_historical_account_regression_does_not_run_asr(self):
  import acceptance_baseline
  baseline={'decision':'S01','voice_score':'0.5','passphrase_score':'0.5'}
  row={'file_path':'unused.wav','speaker_id':'S01','passphrase':'芝麻开门','environment':'dorm'}
  with patch.object(gui,'predict_with_session_model_baseline',return_value=baseline), patch.object(gui,'recognize_fixed_phrase',side_effect=AssertionError('historical A must not use E')):
   result=acceptance_baseline.replay_profile(types.SimpleNamespace(speaker_id='S01'),[row])
   self.assertTrue(result[0]['accepted'])
if __name__=='__main__':unittest.main()
