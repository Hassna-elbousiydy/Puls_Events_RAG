"""Reconstruit et valide dans une zone isolée avant d'activer les résultats."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def activate(staging: Path, root: Path):
    """Installe les dossiers validés avec sauvegarde et rollback sur erreur.

    Maintenance locale uniquement : arrêter la démo pendant l'activation.
    Les sauvegardes restent disponibles pour une restauration manuelle.
    """
    moved=[]
    try:
        for relative in ['data/raw', 'data/processed', 'vectorstore/faiss_index', 'reports/generated']:
            source=staging/relative
            if not source.exists(): continue
            target=root/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            backup=target.with_name(target.name+'.previous-'+str(time.time_ns()))
            existed=target.exists()
            if existed: target.rename(backup)
            moved.append((target,backup,existed))
            shutil.move(str(source),str(target))
    except Exception:
        for target,backup,existed in reversed(moved):
            if target.exists(): shutil.rmtree(target)
            if existed: backup.rename(target)
        raise


def main():
    """Acquisition, nettoyage, chunks, Mistral, FAISS puis tests : arrêt au premier échec."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-date', default=datetime.now(ZoneInfo('Europe/Paris')).date().isoformat())
    parser.add_argument('--from-snapshot', action='store_true', help='Réutilise le CSV brut existant ; aucun nouvel événement téléchargé.')
    args=parser.parse_args()
    datetime.strptime(args.reference_date,'%Y-%m-%d')
    staging=Path(tempfile.mkdtemp(prefix='.rebuild-',dir=ROOT))
    env=os.environ.copy(); env['PULS_REFERENCE_DATE']=args.reference_date
    env['PYTHONPATH']=str(ROOT)
    # Charger le .env de la racine, sans le copier dans la zone de travail.
    from dotenv import dotenv_values
    for key,value in dotenv_values(ROOT/'.env').items():
        if value is not None: env.setdefault(key,value)
    def run(command):
        subprocess.run([sys.executable,*command],cwd=staging,env=env,check=True)
    try:
        if args.from_snapshot:
            raw=staging/'data/raw'; raw.mkdir(parents=True)
            shutil.copy2(ROOT/'data/raw/evenements-publics-openagenda.csv',raw/'evenements-publics-openagenda.csv')
        else:
            run([str(ROOT/'scripts/fetch_openagenda.py'),'--reference-date',args.reference_date])
        run([str(ROOT/'scripts/preprocess_openagenda.py'),'--reference-date',args.reference_date])
        run([str(ROOT/'scripts/build_chunks.py')])
        run([str(ROOT/'scripts/build_vectorstore.py')])
        run(['-m','pytest','-q',str(ROOT/'tests')])
        activate(staging,ROOT)
        print('RECONSTRUCTION VALIDÉE ET ACTIVÉE')
    except subprocess.CalledProcessError:
        print(f'RECONSTRUCTION ÉCHOUÉE : fichiers actifs conservés. Diagnostic dans {staging}')
        raise SystemExit(2)
    finally:
        if staging.exists() and not any(staging.iterdir()): staging.rmdir()


if __name__=='__main__': main()
