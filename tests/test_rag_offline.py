"""Tests unitaires isolés et explicites : aucune génération réelle prétendue."""
import json
from unittest.mock import Mock
import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from src.rag.pipeline import PulsEventsRAG
from src.rag.retrieval import EventRetriever
from src.rag.errors import MistralError, invoke_safely
from scripts.preprocess_openagenda import parse_and_filter_timings
import pandas as pd


@pytest.mark.parametrize('status', [401,403,429,500])
def test_http_errors_are_not_answers(status):
    """Une erreur API produit une exception claire sans secret ni faux résultat."""
    request = httpx.Request('POST','https://api.mistral.ai/v1/chat/completions')
    response = httpx.Response(status, request=request)
    with pytest.raises(MistralError) as error:
        invoke_safely(response.raise_for_status)
    assert error.value.status_code == status


@pytest.mark.parametrize('error', [httpx.ConnectError('secret'),httpx.ReadTimeout('secret')])
def test_transport_errors_are_safe(error):
    def fail(): raise error
    with pytest.raises(MistralError) as caught: invoke_safely(fail)
    assert 'secret' not in str(caught.value)


def test_programming_error_is_not_masked():
    def broken(): raise TypeError('bug')
    with pytest.raises(TypeError): invoke_safely(broken)


@pytest.mark.parametrize('content',[None,'', '  ', []])
def test_pipeline_rejects_empty_response(content):
    retriever = Mock(); retriever.retrieve_context.return_value=([{'uid':'unit-test'}],'Contexte de test')
    llm = RunnableLambda(lambda prompt: Mock(content=content))
    rag = PulsEventsRAG(retriever=retriever,llm=llm)
    with pytest.raises(MistralError): rag.ask('Question de test')


def test_pipeline_prompt_and_sources():
    retriever=Mock(); retriever.retrieve_context.return_value=([{'uid':'unit-test'}],'Contexte de test')
    prompts=[]
    def respond(prompt):
        prompts.append(prompt); return AIMessage(content='Réponse simulée pour test unitaire')
    rag=PulsEventsRAG(retriever=retriever,llm=RunnableLambda(respond))
    result=rag.ask('Question',city='Nantes')
    assert result['context']=='Contexte de test'
    assert len(prompts)==1
    assert 'Contexte de test' in prompts[0].to_string()
    assert result['events']==[{'uid':'unit-test'}]
    assert retriever.retrieve_context.call_count==1


def test_no_events_never_calls_chat():
    retriever=Mock(); retriever.retrieve_context.return_value=([], '')
    def forbidden(prompt): raise AssertionError('Chat appelé')
    rag=PulsEventsRAG(retriever=retriever,llm=RunnableLambda(forbidden))
    assert rag.ask('Question')['events']==[]
    with pytest.raises(ValueError): rag.ask('  ')


def test_retrieval_city_dedup_context(monkeypatch):
    monkeypatch.setenv('PULS_REFERENCE_DATE','2026-09-14')
    retriever=EventRetriever.__new__(EventRetriever)
    timing=json.dumps([{'begin':'2026-10-01T10:00:00+00:00','end':'2026-10-01T12:00:00+00:00'}])
    def doc(uid,city):
        return Document(page_content='Contenu unitaire',metadata={'uid':uid,'city':city,'region':'Pays de la Loire','eligible_timings_json':timing,'chunk_id':uid+'_0','title':'Test','location_text':city,'date_range':'2026-10-01','canonical_url':'https://example.org'})
    n=doc('test-nantes','Nantes'); p=doc('test-paris','Paris')
    def search(question,k,filter,fetch_k):
        return [(d,s) for d,s in [(n,0.1),(n,0.2),(p,0.3)] if filter(d.metadata)]
    retriever.vectorstore=Mock(); retriever.vectorstore.index.ntotal=3
    retriever.vectorstore.similarity_search_with_score.side_effect=search
    retriever.documents_by_uid={'test-nantes':[n,n]}
    events,context=retriever.retrieve_context('Test',city=' nAnTeS ')
    assert len(events)==1
    assert events[0]['best_score']==0.1
    assert context.count('Contenu unitaire')==1
    assert not retriever.retrieve_events('Test',city='Nantes',end_date='2026-09-30')
    with pytest.raises(ValueError): retriever.retrieve_events('Test',top_events=0)


def test_invalid_and_expired_timings():
    cutoff=pd.Timestamp('2025-09-14',tz='UTC')
    assert parse_and_filter_timings('invalid',cutoff)==[]
    for begin,end in [('2025-09-13T10:00Z','2025-09-13T11:00Z'),('2026-01-02T10:00Z','2026-01-01T10:00Z')]:
        assert parse_and_filter_timings(json.dumps([{'begin':begin,'end':end}]),cutoff)==[]
    assert len(parse_and_filter_timings(json.dumps([{'begin':'2027-01-01T10:00Z','end':'2027-01-01T11:00Z'}]),cutoff))==1


def test_judge_rejects_invalid_scores():
    from src.rag.evaluation import judge_answer
    with pytest.raises(ValueError):
        judge_answer('Q','A','C','R',RunnableLambda(lambda p:AIMessage(content='{}')))


def test_model_configurable_without_retry(monkeypatch):
    import src.rag.config as config
    monkeypatch.setenv('MISTRAL_API_KEY','unit-test-placeholder')
    monkeypatch.setenv('MISTRAL_CHAT_MODEL','unit-test-model')
    constructor=Mock(); monkeypatch.setattr(config,'ChatMistralAI',constructor)
    config.make_chat()
    assert constructor.call_args.kwargs['model']=='unit-test-model'
    assert constructor.call_args.kwargs['max_retries']==0


def test_rebuild_preserves_active_index_on_api_failure(tmp_path,monkeypatch):
    import scripts.build_vectorstore as build
    target=tmp_path/'index'; target.mkdir(); (target/'previous').write_text('valid index')
    monkeypatch.setenv('MISTRAL_API_KEY','unit-test-placeholder')
    monkeypatch.setattr(build,'parse_args',lambda:__import__('argparse').Namespace(batch_size=1,limit=None,output_dir=target,report_path=tmp_path/'report.json'))
    monkeypatch.setattr(build,'load_chunks',lambda p:[{'chunk_id':'test','uid':'test','content':'fixture'}])
    embedding=Mock();embedding.embed_documents.side_effect=RuntimeError('API failed')
    monkeypatch.setattr(build,'MistralAIEmbeddings',lambda **kwargs:embedding)
    with pytest.raises(RuntimeError): build.main()
    assert (target/'previous').read_text()=='valid index'


def test_evaluation_references_match_dataset():
    from pathlib import Path
    path=Path('data/processed/events_processed.csv')
    if not path.exists(): pytest.skip('Dataset absent')
    source=pd.read_csv(path,dtype={'uid':str},keep_default_na=False).set_index('uid')
    evaluation=Path(__file__).resolve().parents[1]/'data/evaluation/rag_evaluation.jsonl'
    cases=[json.loads(line) for line in evaluation.read_text().splitlines()]
    assert 20<=len(cases)<=30
    for case in cases:
        for ref in case['references']:
            assert ref['uid'] in source.index
            for field in ['title','city','canonical_url','date_range','location_text']:
                assert ref[field]==source.loc[ref['uid'],field]
