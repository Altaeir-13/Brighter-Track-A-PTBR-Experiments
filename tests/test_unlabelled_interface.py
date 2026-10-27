from pathlib import Path
import json
import runpy
import pytest

INTERFACE=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/predict_unlabelled.py'))

def test_unlabelled_input_accepts_only_id_and_text(tmp_path):
    p=tmp_path/'artificial.jsonl'
    p.write_text(json.dumps({'id':'synthetic-1','text':'Texto artificial de teste.'})+'\n',encoding='utf-8')
    assert INTERFACE['load_texts'](p)[0]['id']=='synthetic-1'

@pytest.mark.parametrize('row',[
    {'id':'synthetic-1','text':'Artificial','anger':1},
    {'id':'ptbr_test_track_a_00001','text':'Artificial'},
])
def test_labels_and_official_test_ids_are_rejected_before_model_access(tmp_path,row):
    p=tmp_path/'invalid.jsonl'
    p.write_text(json.dumps(row)+'\n',encoding='utf-8')
    with pytest.raises(ValueError): INTERFACE['load_texts'](p)
