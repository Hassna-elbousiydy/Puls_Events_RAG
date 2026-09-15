# Compte rendu d’exécution

## Diagnostic et corrections

Mission et cours lus avant modifications. Analyse du ZIP, de l’arborescence,
des scripts, modules, tests, données, index et de l’historique à partir de 2ba6c5f.
L’accès GitHub a retourné 404 à chaque vérification ; travail sur la copie fournie.
La nouvelle pièce jointe est identique à la précédente pour les fichiers projet.

Problèmes corrigés : modèle chat non configurable, retries excessifs, absence de
messages API explicites, destruction possible de l’ancien index avant embeddings,
contrôles figés au 8 septembre, absence de suite offline pour le pipeline,
documentation obsolète, manque de workflow d’évaluation et reconstruction.
Le retrieval existant a été conservé et complété pour la fraîcheur et les filtres.

## Commandes exécutées

Les commandes API ont reçu la clé uniquement en mémoire depuis l’environnement
fourni dans le ZIP, jamais dans un fichier créé ou un argument affiché.

```bash
git status --short
git log --oneline
git switch -c fix/complete-rag-poc
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt socksio
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze
.venv/bin/python -m pytest -q tests/test_rag_offline.py
.venv/bin/python -m scripts.check_mistral_rate_limit
.venv/bin/python -m scripts.test_mistral_minimal
.venv/bin/python -m scripts.evaluate_rag
.venv/bin/python -m scripts.evaluate_rag --generate --output reports/generated/rag_generation_evaluation.json
.venv/bin/python -m scripts.refresh_snapshot --output .refresh-20260915 --reference-date 2026-09-15
.venv/bin/python -m pytest -q
git diff --check
git ls-files .env
git status --short
```

L’activation locale a utilisé scripts.rebuild_pipeline.activate après les 63 tests
de la zone de maintenance. Les scripts Python d’audit ont contrôlé les effectifs,
les hash, les métadonnées et les secrets connus dans tous les blobs Git accessibles.
La reconstruction complète avec nouveaux embeddings et Docker n’a pas été exécutée.

## Résultats exacts

- Installation : pip check, « No broken requirements found. »
- Tests finaux : 63 passed, 1 warning in 32.12s, code de sortie 0.
- Avertissement : annonce de fin de maintenance langchain-community ; technologie conservée.
- Index au 15 septembre : 13 312 événements, 15 149 chunks, dimension 1024.
- Direct Mistral : modèles HTTP 200, modèle disponible, chat HTTP 429 code 1300.
- Minimal LangChain corrigé : HTTP 429, code de sortie 2.
- Retrieval : 25/25 cas terminés, code de sortie 0, sur le snapshot du 14 septembre.
- Génération : 0/25, blocked_mistral_429, code de sortie 2.
- Moyennes sur les 22 cas avec UID : précision 0.21515151515151515,
  rappel 1.0, rang réciproque 0.8787878787878788.
- Deux cas d’absence corrects ; un cas ambigu à examiner humainement.
- Secrets connus : zéro correspondance dans les candidats et l’historique.
- GitHub : HTTP 404, aucun push, aucune PR. Main inchangée.

## Fichiers créés et modifiés depuis la base

A = créé, M = modifié. Les gros artefacts locaux sont ignorés par Git.

```text
M	.dockerignore
M	.env.example
M	.gitignore
M	README.md
A	data/evaluation/README.md
A	data/evaluation/rag_evaluation.jsonl
A	docs/CHECKPOINT.md
A	docs/CONFORMITE.md
A	docs/rapport_technique.md
A	docs/soutenance.md
A	pytest.ini
A	reports/evidence/README.md
A	reports/evidence/mistral_diagnostic.json
A	reports/evidence/pytest_final.txt
A	reports/evidence/rag_evaluation.json
A	reports/evidence/rag_generation_evaluation.json
A	reports/evidence/security_check.json
A	reports/evidence/vectorstore_build_report.json
A	requirements-lock.txt
M	scripts/build_vectorstore.py
M	scripts/check_mistral_rate_limit.py
A	scripts/demo.py
A	scripts/evaluate_rag.py
M	scripts/preprocess_openagenda.py
A	scripts/rebuild_pipeline.py
A	scripts/refresh_snapshot.py
M	scripts/test_mistral_chat.py
A	scripts/test_mistral_minimal.py
A	scripts/test_rag_pipeline.py
A	src/rag/config.py
A	src/rag/errors.py
A	src/rag/evaluation.py
A	src/rag/pipeline.py
M	src/rag/retrieval.py
A	tests/conftest.py
M	tests/test_chunking.py
M	tests/test_preprocessing.py
A	tests/test_rag_offline.py
M	tests/test_vectorstore.py
```

## Commits au moment du relevé

```text
0b2367e Document compliance and preserve actual test and evaluation results
e4ddeaf Keep retrieved timings within the current calendar window
d395e16 Checkpoint RAG fixes tests evaluation and safe rebuild work
```

Le commit qui ajoute ce compte rendu apparaît dans l’historique du bundle joint.
La branche locale est fix/complete-rag-poc. Aucun résultat incomplet n’est annoncé
comme une validation de génération. Voir CONFORMITE.md pour la matrice exhaustive.
