"""Évaluation réelle et séquentielle ; conserve les blocages au lieu d'inventer des scores."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
from src.rag.retrieval import EventRetriever
from src.rag.pipeline import PulsEventsRAG
from src.rag.errors import MistralError
from src.rag.evaluation import judge_answer


def retrieval_metrics(expected, retrieved):
    """Précision/rappel au niveau UID ; les labels doivent être interprétés comme ciblés."""
    if not expected: return None
    relevant=set(expected); found=set(retrieved)
    ranks=[i for i,uid in enumerate(retrieved,1) if uid in relevant]
    return {'uid_precision':len(relevant&found)/len(found) if found else 0,
            'uid_recall':len(relevant&found)/len(relevant),
            'reciprocal_rank':1/min(ranks) if ranks else 0}


def main() -> int:
    """Sauvegarde chaque résultat, s'arrête au premier échec Mistral, sort avec code 2."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--judge',action='store_true', help='Grille sémantique Mistral, nécessite --generate')
    parser.add_argument('--dataset',type=Path,default=Path('data/evaluation/rag_evaluation.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('reports/generated/rag_evaluation.json'))
    args=parser.parse_args()
    if args.judge and not args.generate: parser.error('--judge exige --generate')
    cases=[json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    results=[]; failed=False
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'mode':'generation' if args.generate else 'retrieval',
            'cases_total':len(cases),'results':results,'status':'running',
            'generation_metrics':{'faithfulness':None,'answer_relevancy':None},
            'note':'Les métriques UID ne sont pas les métriques sémantiques Ragas.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding='utf-8')
        temporary.replace(args.output)
    try:
        retriever=EventRetriever(); rag=PulsEventsRAG(retriever=retriever) if args.generate else None
        for case in cases:
            row={'id':case['id'],'question':case['question'],'status':'running'}; results.append(row)
            kwargs={'city':case['expected_city'] or None,'top_events':5,
                    'start_date':case.get('start_date'),'end_date':case.get('end_date')}
            try:
                if rag:
                    result=rag.ask(case['question'],**kwargs)
                    events,context=result['events'],result['context']; row['answer']=result['answer']
                else:
                    events,context=retriever.retrieve_context(case['question'],**kwargs)
                retrieved=[e['uid'] for e in events]
                row.update(status='completed',retrieved_uids=retrieved,context=context,
                    metrics=retrieval_metrics(case['expected_uids'],retrieved))
                if args.judge:
                    row['semantic_judgment']=judge_answer(case['question'],row['answer'],context,case['expected_answer'])
                if case['category']=='absence': row['correct_abstention']=not events
                if case['category']=='ambigu': row['requires_human_judgment']=True
            except MistralError as error:
                row.update(status='blocked_mistral_429' if error.status_code==429 else 'failed_api',error=str(error))
                failed=True; break
            finally: save()
    except Exception as error:
        report['error_type']=type(error).__name__; failed=True
        # Unknown programming errors remain failures and must be investigated.
    report['status']='blocked_or_failed' if failed else 'completed'
    scored=[r['metrics'] for r in results if r.get('metrics')]
    report['completed_cases']=sum(r['status']=='completed' for r in results)
    report['mean_retrieval_metrics']={k:sum(m[k] for m in scored)/len(scored) for k in scored[0]} if scored else None
    save(); print(f"Évaluation : {report['status']} ; {report['completed_cases']}/{len(cases)} cas exécutés. {args.output}")
    return 2 if failed else 0


if __name__=='__main__': raise SystemExit(main())
