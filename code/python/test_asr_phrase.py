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
if __name__=='__main__':unittest.main()
