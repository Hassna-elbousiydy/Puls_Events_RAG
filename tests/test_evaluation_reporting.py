"""Régressions offline du juge et de la comptabilisation ; aucun appel API."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from scripts import evaluate_rag as runner
from src.rag import evaluation


def judgment():
    """Scores artificiels exclusivement réservés aux tests unitaires."""
    return {key: {'score': 0.5, 'justification': 'Fixture unitaire.'}
            for key in evaluation.METRICS}


@pytest.mark.parametrize('raw', ['{', '[]', '{}', '', 'null'])
def test_invalid_judge_output_preserved(raw):
    with pytest.raises(evaluation.JudgeOutputError) as caught:
        evaluation.parse_judgment(AIMessage(content=raw))
    assert caught.value.raw_response == raw


def test_truncation_rejected_even_if_json_parses():
    with pytest.raises(evaluation.JudgeOutputError) as caught:
        evaluation.parse_judgment(AIMessage(content=json.dumps(judgment()),
                                  response_metadata={'finish_reason': 'length'}))
    assert caught.value.finish_reason == 'length'


def test_json_mode_bound_on_real_construction_path(monkeypatch):
    seen = []
    def invoke(prompt, **kwargs):
        seen.append(kwargs)
        return AIMessage(content=json.dumps(judgment()))
    monkeypatch.setattr(evaluation, 'make_chat', lambda **kwargs: RunnableLambda(invoke))
    assert evaluation.judge_answer('q', 'a', 'c', 'r') == judgment()
    assert seen == [{'response_format': {'type': 'json_object'}}]


@pytest.mark.parametrize('failure', [None, 'json', '429', 'runtime'])
def test_report_counts_and_stops(tmp_path, monkeypatch, failure):
    cases = [dict(id=f'q0{i}', question='Fixture', expected_city='', expected_uids=['1'],
                  expected_answer='Fixture', category='theme') for i in range(1, 4)]
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text('\n'.join(json.dumps(c) for c in cases), encoding='utf-8')
    output = tmp_path / 'report.json'
    monkeypatch.setattr(runner, 'EventRetriever', lambda: object())
    monkeypatch.setattr(runner, 'PulsEventsRAG', lambda **kw: SimpleNamespace(
        ask=lambda *a, **k: dict(events=[{'uid': '1'}], context='Fixture', answer='Fixture')))
    calls = []
    def judge(*args):
        calls.append(args)
        if len(calls) == 2:
            if failure == 'json':
                raise evaluation.JudgeOutputError('JSON incomplet', '{', 'length')
            if failure == '429':
                raise runner.MistralError('Limite Mistral', 429)
            if failure == 'runtime':
                raise TypeError('Erreur de fixture')
        return judgment()
    monkeypatch.setattr(runner, 'judge_answer', judge)
    code = runner.main(['--generate', '--judge', '--dataset', str(dataset), '--output', str(output)])
    report = json.loads(output.read_text())
    assert code == (2 if failure else 0)
    assert report['completed_cases'] == (1 if failure else 3)
    assert report['judged_cases'] == (1 if failure else 3)
    assert report['generated_cases'] == (2 if failure else 3)
    if failure:
        assert len(calls) == 2
        assert report['results'][1]['status'] != 'completed'
        assert report['results'][1]['stage'] == 'judgment'
        assert report['error_type']
    if failure == 'json':
        assert report['results'][1]['judge_raw_response'] == '{'
    # Une reprise ciblée ne doit sélectionner qu'un seul cas.
    calls.clear()
    assert runner.main(['--generate', '--judge', '--case-id', 'q02', '--dataset',
                        str(dataset), '--output', str(output)]) == 0
    assert len(calls) == 1
    assert json.loads(output.read_text())['cases_total'] == 1
