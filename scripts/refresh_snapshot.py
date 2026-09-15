"""Actualise la fenêtre temporelle sans recalculer les embeddings inchangés.

Produit un NOUVEAU dossier autonome. L'index source reste intact.
Les textes sont conservés octet pour octet ; seuls les créneaux et leurs
bornes sont actualisés. Ne récupère pas de nouveaux événements OpenAgenda.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


class NoEmbeddings(Embeddings):
    """Interdit les appels API pendant la maintenance de l'index."""
    def embed_documents(self, texts):
        raise RuntimeError('Aucun nouvel embedding autorisé.')
    def embed_query(self, text):
        raise RuntimeError('Aucun nouvel embedding autorisé.')


def refresh(output: Path, reference_date: str) -> dict:
    """Filtre les données et recopie les vecteurs par chunk_id vérifié."""
    if output.exists():
        raise FileExistsError(f'La destination existe déjà : {output}')
    chunks_path = Path('data/processed/events_chunks.jsonl')
    old_report = json.loads(Path('reports/generated/vectorstore_build_report.json').read_text())
    if hashlib.sha256(chunks_path.read_bytes()).hexdigest() != old_report['source_sha256']:
        raise ValueError('Le rapport source ne correspond pas aux chunks.')
    old = FAISS.load_local('vectorstore/faiss_index', NoEmbeddings(), allow_dangerous_deserialization=True)
    if old.index.d != 1024 or old_report['embedding_model'] != 'mistral-embed':
        raise ValueError('Embeddings sources incompatibles.')
    rows = pd.read_csv('data/processed/events_processed.csv', dtype={'uid': str}, keep_default_na=False)
    cutoff = (pd.Timestamp(reference_date, tz='Europe/Paris') - pd.DateOffset(years=1)).tz_convert('UTC')
    def eligible(raw):
        return [t for t in json.loads(raw) if pd.Timestamp(t['begin']) <= pd.Timestamp(t['end'])
                and pd.Timestamp(t['end']) >= cutoff]
    timings = rows.eligible_timings_json.map(eligible)
    selected = rows.loc[timings.map(bool) & rows.region.eq('Pays de la Loire')].copy()
    for idx in selected.index:
        ts = timings.loc[idx]
        selected.at[idx, 'eligible_timings_json'] = json.dumps(ts, ensure_ascii=False)
        selected.at[idx, 'eligible_first_begin'] = min(t['begin'] for t in ts)
        selected.at[idx, 'eligible_last_end'] = max(t['end'] for t in ts)
    by_uid = selected.set_index('uid')
    records = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    positions = {docid: i for i, docid in old.index_to_docstore_id.items()}
    kept, vectors, docs = [], [], {}
    for record in records:
        if str(record['uid']) not in by_uid.index:
            continue
        chunk_id = record['chunk_id']
        source = old.docstore.search(chunk_id)
        if source.page_content != record['content'] or any(source.metadata.get(k) != v for k,v in record.items() if k != 'content'):
            raise ValueError('Divergence entre docstore et chunks sources.')
        row = by_uid.loc[str(record['uid'])]
        for field in ['eligible_timings_json', 'eligible_first_begin', 'eligible_last_end']:
            record[field] = row[field]
        kept.append(record)
        vectors.append(old.index.reconstruct(positions[chunk_id]))
        docs[chunk_id] = Document(page_content=record['content'], metadata={k:v for k,v in record.items() if k != 'content'})
    if not kept:
        raise ValueError('Aucun événement admissible.')
    processed = output / 'data/processed'; processed.mkdir(parents=True)
    selected.to_csv(processed / 'events_processed.csv', index=False)
    new_chunks = processed / 'events_chunks.jsonl'
    new_chunks.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in kept), encoding='utf-8')
    index = faiss.IndexFlatL2(1024); index.add(np.asarray(vectors, dtype='float32'))
    store = FAISS(NoEmbeddings(), index, InMemoryDocstore(docs), {i:r['chunk_id'] for i,r in enumerate(kept)})
    index_path = output / 'vectorstore/faiss_index'; store.save_local(str(index_path))
    loaded = FAISS.load_local(str(index_path), NoEmbeddings(), allow_dangerous_deserialization=True)
    if loaded.index.ntotal != len(kept):
        raise RuntimeError('Échec du rechargement.')
    report = {**old_report, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': hashlib.sha256(new_chunks.read_bytes()).hexdigest(),
        'chunks_available': len(kept), 'chunks_indexed': len(kept), 'faiss_ntotal': len(kept),
        'docstore_entries': len(kept), 'unique_events_indexed': len(selected), 'limit': None,
        'reference_date': reference_date, 'reused_vectors': True, 'new_api_calls': 0,
        'removed_events': len(rows)-len(selected), 'reload_validation': True,
        'index_faiss_bytes': (index_path/'index.faiss').stat().st_size,
        'index_pickle_bytes': (index_path/'index.pkl').stat().st_size}
    report['source_build_elapsed_seconds'] = report.pop('elapsed_seconds', None)
    reports = output / 'reports/generated'; reports.mkdir(parents=True)
    (reports/'vectorstore_build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    """Crée une copie actualisée et affiche les résultats réellement obtenus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-date', default=pd.Timestamp.now(tz='Europe/Paris').date().isoformat())
    args = parser.parse_args()
    print(json.dumps(refresh(args.output, args.reference_date), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
