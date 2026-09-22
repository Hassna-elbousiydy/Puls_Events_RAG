# Soutenance — Puls-Events RAG

**Projet :** Développez un assistant pour la recommandation d’événements culturels
**Formation :** Data Engineer — OpenClassrooms
**Candidate :** Hassna EL-BOUSIYDY
**Durée cible :** 15 minutes
**Démo :** environnement local Docker

---

# Slide 1 — Puls-Events RAG

## Développer un assistant de recommandation d’événements culturels

**Objectif**

Construire un Proof of Concept capable de recommander des événements culturels à partir des données OpenAgenda en utilisant une architecture RAG.

**Technologies principales**

- Python
- Pandas
- LangChain
- Mistral AI
- FAISS
- Docker
- Pytest

**Périmètre**

Pays de la Loire.

### À dire à l’oral

« L’objectif du projet est de démontrer qu’il est possible de construire un assistant de recommandation culturelle basé sur les données OpenAgenda.
J’ai développé un pipeline complet allant de l’acquisition des données jusqu’à la génération et à l’évaluation des réponses. »

---

# Slide 2 — Besoin métier

Puls-Events souhaite permettre à un utilisateur de poser des questions naturelles comme :

- Où voir une exposition à Laval ?
- Quels concerts sont disponibles à Nantes ?
- Quel événement est prévu à Angers à une date précise ?
- Recommande-moi une sortie culturelle.

## Problème

Un LLM utilisé seul peut :

- inventer un événement ;
- donner une mauvaise date ;
- inventer un lieu ;
- utiliser des informations obsolètes.

## Solution retenue

Utiliser un système **RAG — Retrieval-Augmented Generation**.

### À dire à l’oral

« Le LLM n’est pas utilisé comme base de connaissances.
Le système recherche d’abord les événements dans notre corpus, puis transmet uniquement le contexte pertinent au modèle pour générer la réponse. »

---

# Slide 3 — Architecture générale

```text
OpenAgenda
    ↓
Acquisition API
    ↓
Prétraitement
    ↓
Chunking
    ↓
mistral-embed
    ↓
FAISS
    ↓
Question utilisateur
    ↓
Retrieval
    ↓
Contexte pertinent
    ↓
LangChain
    ↓
Chat Mistral
    ↓
Réponse
```

## Rôle des briques

- **OpenAgenda** : source de données
- **Pandas** : nettoyage
- **Mistral Embed** : vectorisation
- **FAISS** : recherche
- **LangChain** : orchestration
- **Mistral Chat** : génération

### À dire à l’oral

« LangChain orchestre les différentes briques mais ne remplace ni FAISS ni Mistral.
FAISS retrouve l’information.
Le modèle Mistral produit ensuite la réponse en langage naturel. »

---

# Slide 4 — Acquisition et préparation des données

## Source

Dataset public :

**Événements publics OpenAgenda**

Accès automatisé via l’API publique du dataset OpenAgenda exposée par OpenDataSoft.

## Dataset initial

```text
24 601 lignes
56 colonnes
```

## Prétraitement

- Pays de la Loire
- événements culturels
- événements non annulés
- contrôle des dates
- historique récent d’un an
- événements futurs
- déduplication des UID
- contrôle des informations nécessaires

### À dire à l’oral

« J’ai commencé avec le CSV pour construire le pipeline, puis j’ai automatisé l’acquisition avec l’API publique du dataset.
Le prétraitement permet surtout d’éviter d’indexer des événements expirés, annulés ou hors périmètre. »

---

# Slide 5 — Chunking et embeddings

## Chunking

```text
RecursiveCharacterTextSplitter
chunk_size = 1500
chunk_overlap = 200
```

Chaque chunk conserve :

- UID
- titre
- ville
- lieu
- dates
- région
- URL

## Embeddings

```text
Modèle : mistral-embed
Dimension : 1024
```

### Pourquoi un overlap ?

Pour ne pas perdre une information placée entre deux morceaux de texte.

### À dire à l’oral

« Un embedding est une représentation numérique du sens du texte.
La question et les descriptions des événements sont placées dans le même espace vectoriel, ce qui permet une recherche sémantique. »

---

# Slide 6 — Pourquoi FAISS ?

## Configuration

```text
FAISS CPU
IndexFlatL2
```

## Avantages pour ce POC

- open source
- local
- simple
- exact
- aucune infrastructure cloud supplémentaire
- volume compatible avec une recherche exhaustive

## Principe

Une distance L2 plus faible signifie une plus grande proximité entre deux vecteurs.

### Alternatives possibles

- Pinecone
- Weaviate
- Milvus
- FAISS IVF
- FAISS HNSW

### À dire à l’oral

« J’ai choisi IndexFlatL2 parce que le volume reste raisonnable.
Il compare exactement les vecteurs, ce qui donne une baseline simple et fiable.
Pour plusieurs millions de vecteurs, je choisirais plutôt un index approximatif ou une base vectorielle distribuée. »

---

# Slide 7 — Snapshot final

## Date de référence

```text
21 septembre 2026
```

## Vectorstore

| Indicateur | Valeur |
|---|---:|
| Chunks indexés | 14 547 |
| Événements uniques | 12 763 |
| Embedding | mistral-embed |
| Dimension | 1024 |
| Index | IndexFlatL2 |
| Événements expirés retirés | 549 |

## Optimisation

Les vecteurs des textes inchangés ont été réutilisés.

```text
Nouveaux appels embeddings : 0
```

### À dire à l’oral

« Le refresh temporel évite de recalculer tous les embeddings lorsqu’on retire simplement des événements expirés. »

---

# Slide 8 — Pipeline RAG

## Retrieval

Le retriever :

1. transforme la question en embedding ;
2. cherche dans FAISS ;
3. applique les filtres structurés éventuels ;
4. regroupe les chunks par UID ;
5. déduplique les événements ;
6. construit le contexte.

## Génération

Le modèle final est :

```text
ministral-8b-2512
temperature = 0
```

Le prompt interdit notamment :

- l’invention d’événements ;
- l’invention de dates ;
- l’invention d’URLs ;
- l’utilisation d’informations absentes du contexte.

### À dire à l’oral

« Les filtres et la recherche vectorielle sont complémentaires.
Le vectoriel permet de comprendre le sens de la question, alors qu’un filtre de ville ou de date garantit une contrainte métier précise. »

---

# Slide 9 — Évaluation du POC

## Benchmark

```text
25 questions-réponses
25 / 25 validées humainement
```

Types de cas :

- événement précis
- ville
- période
- plusieurs événements
- requêtes ambiguës
- hors périmètre

## Métriques retrieval

- UID Precision
- UID Recall
- Reciprocal Rank / MRR

## Métriques génération

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

### À dire à l’oral

« Les métriques sémantiques sont calculées avec un juge Mistral utilisant une grille explicite.
Ce n’est pas une exécution directe de Ragas.
Je les utilise donc avec les métriques déterministes et la revue humaine. »

---

# Slide 10 — Résultats

## Retrieval

| Métrique | Score |
|---|---:|
| UID Precision | 0.2212 |
| UID Recall | **1.0000** |
| MRR | **0.9545** |

## Génération

| Métrique | Score |
|---|---:|
| Faithfulness | **0.9160** |
| Answer relevancy | **0.8980** |
| Context precision | **0.8620** |
| Context recall | **0.7740** |

### Interprétation

**UID Recall = 1**

Tous les UID attendus ont été retrouvés sur les cas où une référence UID existe.

**MRR = 0.9545**

Le bon événement apparaît généralement très haut dans les résultats.

**Faithfulness = 0.916**

Les réponses sont globalement bien fondées sur le contexte.

### À dire à l’oral

« La précision UID paraît faible parce que le système renvoie plusieurs candidats alors que la référence attend souvent un seul événement.
Elle ne signifie donc pas que seulement 22 % des réponses sont correctes. »

---

# Slide 11 — Qualité, tests et limites

## Tests

Commande finale :

```powershell
docker compose run --rm -e PULS_REFERENCE_DATE=2026-09-21 rag pytest -q
```

Résultat :

```text
74 passed
1 warning non bloquant
```

## Cas q08

Une ambiguïté dans la ground truth a été détectée.

Plusieurs occurrences de :

```text
Cartophote / La Traversée Photographique
```

étaient présentes.

Après clarification de la date :

```text
UID recall = 1
MRR = 1
Faithfulness = 1
Context precision = 0.95
```

## Enseignement

La qualité du benchmark est aussi importante que celle du modèle.

### À dire à l’oral

« Le cas q08 m’a permis de montrer qu’un mauvais benchmark peut faire croire que le RAG est mauvais.
Il faut donc toujours analyser les erreurs avant de modifier le modèle. »

---

# Slide 12 — Démonstration live

## Exemple 1 — Cas normal

```powershell
docker compose run --rm rag python -m scripts.demo "Où voir Plants and People à Nantes ?" --city Nantes
```

Objectif :

- montrer le retrieval ;
- montrer les sources ;
- montrer la réponse générée.

## Exemple 2 — Ville + date

```powershell
docker compose run --rm rag python -m scripts.demo "Quand rencontrer Abigail Assor ?" --city Angers --start-date 2026-09-15 --end-date 2026-09-15
```

Objectif :

montrer l’utilisation des filtres structurés.

## Exemple 3 — Hors périmètre

```powershell
docker compose run --rm rag python -m scripts.demo "Quels concerts à Paris ?" --city Paris
```

Objectif :

montrer que le système ne doit pas inventer de résultat.

### À dire à l’oral

« Je vais maintenant exécuter directement le pipeline depuis l’environnement local afin de montrer une recommandation complète. »

---

# Slide 13 — Limites et passage en production

## Limites actuelles

- pas de mémoire conversationnelle ;
- pas d’interface Web ;
- dépendance à Mistral ;
- extraction automatique des filtres limitée ;
- qualité dépendante des données OpenAgenda ;
- `IndexFlatL2` non adapté aux très grands volumes ;
- variabilité possible du LLM et du LLM juge.

## Pour une version de production

- acquisition planifiée ;
- snapshots versionnés ;
- secrets sécurisés ;
- monitoring ;
- collecte du feedback ;
- tests de non-régression ;
- suivi du drift ;
- base vectorielle scalable si nécessaire.

---

# Slide 14 — Conclusion

## Résultat

Le POC valide une chaîne complète :

```text
OpenAgenda
→ Prétraitement
→ Embeddings
→ FAISS
→ Retrieval
→ LangChain
→ Mistral
→ Évaluation
```

## Chiffres clés

```text
12 763 événements uniques
14 547 chunks
25 / 25 cas évalués
74 tests réussis
UID Recall = 1.0000
MRR = 0.9545
Faithfulness = 0.9160
```

## Conclusion

Le POC démontre la faisabilité d’un assistant de recommandation culturelle fondé sur une base documentaire actualisable.

La prochaine étape serait de transformer ce POC en service de production avec automatisation, monitoring et interface utilisateur.

### À dire à l’oral

« Le projet valide donc la faisabilité technique du RAG pour Puls-Events.
Les prochaines étapes ne concernent plus la preuve de concept elle-même mais son industrialisation. »

---

# Questions probables de l’évaluateur

## Comment FAISS optimise-t-il la recherche dans une base vectorielle ?

FAISS est spécialisé dans le stockage et la recherche de vecteurs numériques.

Dans mon POC, `IndexFlatL2` réalise une recherche exacte en calculant les distances L2 entre la requête et les vecteurs indexés.

Pour des volumes beaucoup plus grands, FAISS propose des index comme IVF ou HNSW qui réduisent le nombre de comparaisons au prix d’une recherche approximative.

---

## Quelles sont les limites de FAISS avec beaucoup de données ?

`IndexFlatL2` doit comparer la requête avec tous les vecteurs.

Quand le volume devient très important :

- temps de recherche plus élevé ;
- consommation mémoire importante ;
- gestion distribuée plus complexe ;
- haute disponibilité non fournie directement.

Pour une production à grande échelle, je comparerais FAISS IVF/HNSW ou une solution distribuée comme Milvus, Weaviate ou Pinecone.

---

## Pourquoi utiliser LangChain ?

LangChain facilite l’orchestration entre :

- le retriever ;
- le contexte ;
- le prompt ;
- le modèle de langage.

Il permet également de séparer les composants du pipeline et de rendre le code plus maintenable.

Dans ce projet, LangChain n’est ni la base vectorielle ni le modèle.

---

## Comment garantir la qualité des données ?

J’utilise plusieurs contrôles :

- validation du périmètre géographique ;
- contrôle temporel ;
- suppression des événements annulés ;
- déduplication des UID ;
- contrôle des valeurs essentielles ;
- tests automatisés ;
- validation humaine du benchmark ;
- traçabilité par UID ;
- snapshots reproductibles.

---

## Comment détecter le drift ?

Je suivrais dans le temps :

- distribution des villes ;
- catégories d’événements ;
- volume quotidien ;
- taux de champs manquants ;
- performances du benchmark ;
- taux de réponses sans résultat ;
- feedback utilisateur.

Une baisse régulière des métriques ou un changement important dans les données peut indiquer un drift.

---

## Quels KPI suivre en production ?

### Données

- fraîcheur du dataset ;
- taux de données manquantes ;
- nombre d’événements indexés ;
- taux d’échec du pipeline.

### Retrieval

- Recall@k ;
- Precision@k ;
- MRR ;
- taux de requêtes sans résultat.

### Génération

- faithfulness ;
- answer relevancy ;
- taux d’hallucination détecté.

### Technique

- latence ;
- taux d’erreur API ;
- disponibilité ;
- coût par requête.

### Utilisateur

- satisfaction ;
- taux de clic ;
- feedback positif/négatif.

---

# Planning oral conseillé

```text
Slide 1   : 30 sec
Slide 2   : 1 min
Slide 3   : 1 min 30
Slide 4   : 1 min
Slide 5   : 1 min
Slide 6   : 1 min
Slide 7   : 45 sec
Slide 8   : 1 min 30
Slide 9   : 1 min
Slide 10  : 1 min 30
Slide 11  : 1 min
Slide 12  : 2 à 3 min de démo
Slide 13  : 1 min
Slide 14  : 30 sec
```

Durée cible totale :

```text
environ 15 minutes
```
