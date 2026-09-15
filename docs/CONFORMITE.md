# Audit de conformité — Puls-Events

Référence principale : Mission(7).docx, texte identique à Mission(6).docx fourni.
Référence complémentaire : Cours_Mettez en place un RAG pour un LLM(2).docx,
texte identique au cours fourni précédemment. Les deux textes ont été lus intégralement.
Les anciennes mentions Mission(4).docx ne désignent pas une pièce disponible distincte.
Date de contrôle : 15 septembre 2026. Branche locale : fix/complete-rag-poc.

La conformité à 100 % n'est pas acquise : le chat réel, la revue humaine des
annotations et les livrables de soutenance restent à valider. Les tests utilisant
un faux client contrôlent le code, jamais la disponibilité réelle de Mistral.

## Matrice des exigences

| Exigence mission | Fichier / implémentation | Test effectué | Résultat / limite | Statut |
|---|---|---|---|---|
| Python, Pandas, LangChain, Mistral, FAISS CPU | requirements.txt, requirements-lock.txt | installation Python 3.12 et pip check | dépendances résolues sans conflit | ✅ Conforme |
| Environnement Docker reproductible | Dockerfile, compose.yaml, README | inspection des fichiers | Docker indisponible dans Work ; image non reconstruite ici | ⚠️ Conforme avec limite documentée |
| Région définie | scripts/preprocess_openagenda.py | tests/test_preprocessing.py | contrôle de chaque ligne Pays de la Loire | ✅ Conforme |
| Un an d'historique et événements futurs | preprocessing, scripts/refresh_snapshot.py, retrieval.py | contrôle des créneaux et actualisation quotidienne explicite | fenêtre contrôlée au 15/09/2026 ; renouveler l'acquisition pour obtenir les nouveautés | ⚠️ Conforme avec limite documentée |
| Nettoyage, valeurs manquantes, UID uniques | preprocessing | tests/test_preprocessing.py | contrôles réels du corpus ; ville manquante possible, lieu conservé | ✅ Conforme |
| Reconstruction depuis OpenAgenda | scripts/rebuild_pipeline.py | inspection, tests de prétraitement et protection de l'index en cas d'échec | acquisition et vectorisation complètes non rejouées ensemble ; quota externe | ⚠️ Conforme avec limite documentée |
| Chunking avec contexte et overlap | scripts/build_chunks.py | tests/test_chunking.py | texte non vide, métadonnées et couverture contrôlés | ✅ Conforme |
| Embeddings Mistral inchangés | scripts/build_vectorstore.py, refresh_snapshot.py | dimension et hash des sources, réutilisation des vecteurs | mistral-embed, 1024 dimensions | ✅ Conforme |
| FAISS CPU persistant et rechargeable | vectorstore/faiss_index, tests/test_vectorstore.py | rechargement réel et taille | IndexFlatL2 ; tous les chunks admissibles conservés | ✅ Conforme |
| Recherche, ville, déduplication | src/rag/retrieval.py | tests offline et évaluation API distincte | filtre strict ville/région/créneaux ; recherche exhaustive de ce petit index | ✅ Conforme |
| Pipeline modulaire LangChain → Mistral | src/rag/pipeline.py | tests/test_rag_offline.py | exécution contrôlée offline ; chat réel à vérifier séparément | ⚠️ Conforme avec limite documentée |
| Réponse française sourcée, pas d'invention | prompt de pipeline.py | vérification du prompt et du contexte | règles présentes ; qualité réelle des générations non démontrée | ⚠️ Conforme avec limite documentée |
| Pas de mémoire conversationnelle obligatoire | pipeline.py | inspection | aucun historique nécessaire | ✅ Conforme |
| 401/403, 429, timeout, réseau, réponse vide | src/rag/errors.py, config.py | tests paramétrés | erreurs explicites, aucun retry chat automatique | ✅ Conforme |
| Diagnostic minimal indépendant du retrieval | scripts/check_mistral_rate_limit.py | appel HTTP réel | modèle accessible ; chat 429 code 1300 dans le rapport enregistré | ⚠️ Conforme avec limite documentée |
| Tests unitaires sans appels API accidentels | pytest.ini, tests/conftest.py | socket interdite pendant pytest | scripts smoke exclus de la collecte par défaut | ✅ Conforme |
| Jeu Q/R annoté vérifiable | data/evaluation/rag_evaluation.jsonl | comparaison aux UID/titres/URLs du CSV | 25 cas fondés sur les sources ; human_reviewed=false, validation humaine nécessaire | ⚠️ Conforme avec limite documentée |
| Évaluation retrieval | scripts/evaluate_rag.py | appels réels enregistrés | consulter status et completed_cases du JSON, pas une simulation | ⚠️ Conforme avec limite documentée |
| Évaluation sémantique génération | src/rag/evaluation.py, evaluate_rag.py --generate --judge | validation offline de la grille | quatre axes préparés, aucun score de qualité Mistral certifié | ❌ Non conforme |
| Démo question → sources → réponse | scripts/demo.py | composants offline, code inspecté | sources affichées avant le chat ; réponse réelle dépend du compte Mistral | ⚠️ Conforme avec limite documentée |
| Documentation reproductible | README.md, .env.example | revue des commandes et chemins | acquisition, maintenance, reconstruction, tests, démo documentés | ✅ Conforme |
| Rapport final de 5–10 pages | docs/rapport_technique.md | contenu préparatoire | document PDF/Word final et mesures de génération manquants | ❌ Non conforme |
| Support de 10–15 diapositives | docs/soutenance.md | scénario de 12 diapositives préparé | fichier PowerPoint final non produit | ❌ Non conforme |
| Secrets exclus du code versionné | .gitignore, .dockerignore | git ls-files et scan des fichiers candidats | aucune clé autorisée dans les commits ; .env exclu | ✅ Conforme |
| Branche distincte de main | historique Git local | git branch, git log | corrections locales sur fix/complete-rag-poc uniquement | ✅ Conforme |
| Publication GitHub et PR | connecteur GitHub | get_repo sur le dépôt demandé | HTTP 404 ; accès au dépôt à accorder | ❌ Non conforme |

## Les quatre étapes, cinq rubriques chacune

| Étape | Rubrique | Contrôle explicite | Statut |
|---|---|---|---|
| 1 — Environnement | Prérequis | Python 3.12, projet existant et dépendances installées | ✅ Conforme |
| 1 | Résultat attendu | Imports exécutés par les tests ; Docker non exécuté ici | ⚠️ Conforme avec limite documentée |
| 1 | Recommandations | venv et versions enregistrées dans un lock | ✅ Conforme |
| 1 | Points de vigilance | pip check sans conflit ; API compatible avec langchain-mistralai 1.1.6 | ✅ Conforme |
| 1 | Outils | Python, LangChain, Mistral, faiss-cpu conservés | ✅ Conforme |
| 2 — Données | Prérequis | export OpenAgenda réel présent, provenance conservée | ✅ Conforme |
| 2 | Résultat attendu | corpus propre, régional et temporel contrôlé | ✅ Conforme |
| 2 | Recommandations | un an d'historique plus futur ; actualisation sans réembedding possible | ✅ Conforme |
| 2 | Points de vigilance | créneaux invalides rejetés ; snapshot ne fournit pas les nouvelles annonces | ⚠️ Conforme avec limite documentée |
| 2 | Outils | Pandas et scripts Python réutilisés | ✅ Conforme |
| 3 — Vectorisation | Prérequis | dataset et chunks validés avant indexation | ✅ Conforme |
| 3 | Résultat attendu | FAISS persistant complet, métadonnées et UID vérifiés | ✅ Conforme |
| 3 | Recommandations | chunking avant embeddings, overlap, IndexFlatL2 adapté au POC | ✅ Conforme |
| 3 | Points de vigilance | index ancien conservé sur échec ; provenance SHA256 contrôlée ; performances production non évaluées | ⚠️ Conforme avec limite documentée |
| 3 | Outils | embeddings Mistral, LangChain, FAISS CPU et tests Python | ✅ Conforme |
| 4 — RAG et évaluation | Prérequis | retrieval réutilisé et testé, 25 cas avec références sources | ⚠️ Conforme avec limite documentée |
| 4 | Résultat attendu | pipeline complet codé ; génération réelle non validée | ❌ Non conforme |
| 4 | Recommandations | questions variées, abstention, ville/période ; aucune mémoire superflue | ✅ Conforme |
| 4 | Points de vigilance | aucune métrique inventée, limites API tracées ; annotations à faire approuver humainement | ⚠️ Conforme avec limite documentée |
| 4 | Outils | LangChain + Mistral, grille sémantique quatre axes préparée, tests Python | ⚠️ Conforme avec limite documentée |

## Interprétation du 429

Un GET modèles HTTP 200 prouve que le compte authentifié expose le modèle testé.
Le POST minimal direct HTTP 429 code 1300 exclut FAISS, la taille du contexte RAG
et LangChain comme causes nécessaires de ce refus. Il ne permet pas de distinguer
une limite par seconde, par minute, un quota ou une restriction de workspace.
Le propriétaire doit consulter les limites/usage Mistral pour cette distinction.
Aucun changement de plan, paiement ou création de clé n'a été effectué.

Une seconde erreur de certificat TLS a été observée dans le client LangChain de
cet environnement. Le test minimal LangChain corrigé retourne lui aussi HTTP 429.
La configuration utilise désormais des clients HTTP vérifiant
les certificats système. Elle ne désactive pas TLS. Les rapports de génération
indiquent le résultat réellement obtenu après correction.

## Évaluation choisie

La grille sémantique reprend faithfulness, answer relevancy, context precision et
context recall. Elle emploie un juge Mistral et exige une justification et un score
borné pour chaque axe. Ce n'est pas une exécution de Ragas : cette alternative
limite les appels et conserve les dépendances existantes. Aucune incompatibilité
Ragas n'est affirmée. Les scores de juge doivent être revus humainement.
Les métriques UID déterministes ne remplacent pas ces quatre mesures sémantiques.
Les labels ciblés ne sont pas exhaustifs : leur précision peut sous-estimer les
événements pertinents non annotés. Les villes/périodes sont fournies explicitement
au script ; cette évaluation ne mesure pas leur extraction automatique du langage.
