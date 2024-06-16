import tempfile
from typing import List, cast

import open_clip
import pytest
import torch
from rclip.model import Model


def test_extract_query_multiplier():
  assert Model._extract_query_multiplier('1.5:cat') == (1.5, 'cat')  # type: ignore
  assert Model._extract_query_multiplier('cat') == (1., 'cat')  # type: ignore
  assert Model._extract_query_multiplier('1:cat') == (1., 'cat')  # type: ignore
  assert Model._extract_query_multiplier('0.5:cat') == (0.5, 'cat')  # type: ignore
  assert Model._extract_query_multiplier('.5:cat') == (0.5, 'cat')  # type: ignore
  assert Model._extract_query_multiplier('1.:cat') == (1., 'cat')  # type: ignore
  assert Model._extract_query_multiplier('1..:cat') == (1., '1..:cat')  # type: ignore
  assert Model._extract_query_multiplier('..:cat') == (1., '..:cat')  # type: ignore
  assert Model._extract_query_multiplier('whatever:cat') == (1., 'whatever:cat')  # type: ignore
  assert (Model._extract_query_multiplier('1.5:complex and long query') ==  # type: ignore
          (1.5, 'complex and long query'))


def test_text_model_produces_the_same_vector_as_the_main_model(monkeypatch: pytest.MonkeyPatch):
  with tempfile.TemporaryDirectory() as tmpdirname:
    monkeypatch.setenv('RCLIP_DATADIR', tmpdirname)
    model = Model()
    assert model._model_var == None  # type: ignore
    assert model._model_text_var == None  # type: ignore

    model._load_model()  # type: ignore
    assert model._model_var != None  # type: ignore
    assert model._model_var.transformer != None  # type: ignore
    assert model._model_var.visual != None  # type: ignore

    assert model._model_text_var == None  # type: ignore
    text_model = model._get_text_model(model._model_var)  # type: ignore
    assert text_model.transformer != None  # type: ignore
    assert text_model.visual == None  # type: ignore

    full_model = model._model_var  # type: ignore
    assert full_model.visual != None  # type: ignore

    def encode_text(clip_model: open_clip.CLIP, text: List[str]):
      return clip_model.encode_text(model._tokenizer(text).to(model._device))  # type: ignore

    assert torch.equal(encode_text(full_model, ['cat']), encode_text(text_model, ['cat']))
    assert torch.equal(encode_text(full_model, ['cat', 'dog']), encode_text(text_model, ['cat', 'dog']))
    assert torch.equal(encode_text(full_model, ['cat', 'dog', 'bird']), encode_text(text_model, ['cat', 'dog', 'bird']))


def test_loads_text_model_when_text_processing_only_requested_and_checkpoint_exists(monkeypatch: pytest.MonkeyPatch):
  with tempfile.TemporaryDirectory() as tmpdirname:
    monkeypatch.setenv('RCLIP_DATADIR', tmpdirname)
    model1 = Model()
    assert model1._model_var == None  # type: ignore
    assert model1._model_text_var == None  # type: ignore

    full_model = cast(open_clip.CLIP, model1._model)  # type: ignore
    assert model1._model_var != None  # type: ignore
    assert model1._model_var.transformer != None  # type: ignore
    assert model1._model_var.visual != None  # type: ignore
    assert model1._model_text_var == None  # type: ignore

    model2 = Model()
    assert model2._model_var == None  # type: ignore
    assert model2._model_text_var == None  # type: ignore

    text_model = cast(open_clip.CLIP, model2._model_text)  # type: ignore
    assert model2._model_var == None  # type: ignore
    assert model2._model_text_var != None  # type: ignore
    assert model2._model_text_var.transformer != None  # type: ignore
    assert model2._model_text_var.visual == None  # type: ignore
    assert model2._model_text_var == text_model  # type: ignore

    def encode_text(clip_model: open_clip.CLIP, text: List[str]):
      return clip_model.encode_text(model1._tokenizer(text).to(model1._device))  # type: ignore

    assert torch.equal(encode_text(full_model, ['cat']), encode_text(text_model, ['cat']))
    assert torch.equal(encode_text(full_model, ['cat', 'dog']), encode_text(text_model, ['cat', 'dog']))
    assert torch.equal(encode_text(full_model, ['cat', 'dog', 'bird']), encode_text(text_model, ['cat', 'dog', 'bird']))


def test_loads_full_model_when_text_processing_only_requested_and_checkpoint_doesnt_exist(monkeypatch: pytest.MonkeyPatch):
  with tempfile.TemporaryDirectory() as tmpdirname:
    monkeypatch.setenv('RCLIP_DATADIR', tmpdirname)
    model = Model()
    assert model._model_var == None  # type: ignore
    assert model._model_text_var == None  # type: ignore

    _ = model._model_text  # type: ignore
    assert model._model_var != None  # type: ignore
    assert model._model_var.transformer != None  # type: ignore
    assert model._model_var.visual != None  # type: ignore
    assert model._model_text_var == None  # type: ignore
