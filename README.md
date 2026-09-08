# Puls-Events RAG

## Présentation

Ce projet est un Proof of Concept (POC) de système RAG
(**Retrieval-Augmented Generation**) développé pour l'entreprise **Puls-Events**.

Puls-Events propose une plateforme permettant aux utilisateurs de découvrir
des événements culturels. L'objectif de ce projet est d'étudier la faisabilité
d'un assistant intelligent capable de recommander des événements pertinents
à partir de données issues d'**OpenAgenda**.

Le système RAG combinera :

- **LangChain** pour orchestrer les différentes briques du pipeline ;
- **Mistral AI** pour les embeddings et la génération de réponses ;
- **FAISS** pour l'indexation et la recherche vectorielle ;
- **Python** pour le traitement des données et le développement du système ;
- **Docker** pour disposer d'un environnement reproductible.

---

## Objectifs du projet

Le système final devra permettre de :

1. récupérer et charger les événements issus d'OpenAgenda ;
2. nettoyer et structurer les données ;
3. filtrer les événements selon un périmètre géographique défini ;
4. conserver uniquement les événements respectant les contraintes temporelles
   de la mission ;
5. construire un texte exploitable pour chaque événement ;
6. découper les textes en chunks lorsque cela est nécessaire ;
7. générer les embeddings des contenus ;
8. créer un index vectoriel avec FAISS ;
9. enregistrer les métadonnées associées aux événements ;
10. effectuer des recherches par similarité sémantique ;
11. utiliser LangChain pour orchestrer le système RAG ;
12. utiliser Mistral pour générer les réponses de l'assistant ;
13. permettre la reconstruction de la base vectorielle à la demande ;
14. tester automatiquement la qualité et la conformité des données ;
15. évaluer le système à partir d'un jeu de questions/réponses annoté.

---

## Architecture générale prévue

Le pipeline suivra globalement le fonctionnement suivant :

```text
OpenAgenda
    |
    v
Données brutes
    |
    v
Pré-processing / nettoyage / filtrage
    |
    v
Données structurées
    |
    v
Chunking
    |
    v
Embeddings Mistral
    |
    v
Index vectoriel FAISS
    |
    v
Recherche sémantique
    |
    v
Contexte récupéré
    |
    v
LangChain + Mistral
    |
    v
Réponse / recommandation utilisateur
```

---

## Contraintes principales de la mission

Le POC doit respecter plusieurs contraintes :

- les données doivent provenir d'OpenAgenda ;
- un périmètre géographique doit être sélectionné ;
- les événements utilisés doivent respecter la contrainte de récence
  demandée dans la mission ;
- la base vectorielle doit pouvoir être reconstruite à la demande ;
- FAISS doit être utilisé pour l'indexation vectorielle ;
- LangChain doit être utilisé dans le système RAG ;
- Mistral doit être utilisé pour les modèles ;
- le projet doit être versionné avec Git ;
- les dépendances doivent être documentées ;
- des tests Python doivent contrôler les données utilisées ;
- un jeu de questions/réponses annoté devra être créé pour l'évaluation.

Le périmètre géographique et les règles exactes de préparation des données
seront documentés pendant l'étape 2.

---

## Environnement de développement

L'environnement actuellement utilisé est basé sur :

- **Python 3.12.14**
- **LangChain 1.4.0**
- **LangChain Community 0.4.2**
- **LangChain Mistral 1.1.6**
- **Mistral SDK 2.9.4**
- **FAISS CPU 1.15.0**
- **NumPy 2.5.3**
- **Docker**
- **Docker Compose**
- **Git**

Un environnement virtuel Python est créé à l'intérieur du conteneur Docker
afin d'isoler les dépendances du projet.

FAISS est installé dans sa version **CPU**, conformément aux contraintes
de la mission.

---

## Structure du projet

```text
04_Puls_Events_RAG/
|
|-- data/
|   |-- raw/
|   |   `-- evenements-publics-openagenda.csv
|   |
|   |-- interim/
|   |   `-- .gitkeep
|   |
|   `-- processed/
|       `-- .gitkeep
|
|-- docs/
|
|-- reports/
|   `-- .gitkeep
|
|-- scripts/
|   |-- check_environment.py
|   `-- check_mistral_access.py
|
|-- src/
|
|-- tests/
|
|-- vectorstore/
|   `-- .gitkeep
|
|-- .dockerignore
|-- .env
|-- .env.example
|-- .gitignore
|-- compose.yaml
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

### Description des dossiers

#### `data/raw/`

Contient les données brutes récupérées depuis OpenAgenda.

Le fichier actuellement utilisé est :

```text
data/raw/evenements-publics-openagenda.csv
```

Les données brutes ne sont pas versionnées dans Git.

#### `data/interim/`

Contiendra les données obtenues pendant les étapes intermédiaires de
pré-processing.

#### `data/processed/`

Contiendra les données finales nettoyées et structurées, prêtes à être
vectorisées et indexées.

#### `scripts/`

Contient les scripts exécutables du projet.

Actuellement :

- `check_environment.py` : vérifie les bibliothèques Python et le
  fonctionnement de FAISS CPU ;
- `check_mistral_access.py` : vérifie l'accès réel à l'API Mistral.

D'autres scripts seront ajoutés pour le pré-processing, la vectorisation et
la reconstruction de l'index FAISS.

#### `src/`

Contiendra le code source principal du système RAG.

#### `tests/`

Contiendra les tests unitaires et les contrôles automatiques du projet.

Les tests devront notamment vérifier que les événements utilisés respectent
le périmètre géographique et temporel choisi.

#### `vectorstore/`

Contiendra les fichiers générés pour l'index vectoriel FAISS.

L'index étant reconstructible à partir des données et des scripts, les fichiers
générés dans ce dossier ne seront pas versionnés dans Git.

#### `reports/`

Contiendra les résultats d'évaluation et les fichiers générés pendant
l'analyse du POC.

#### `docs/`

Contiendra la documentation complémentaire du projet.

---

## Données

Les données OpenAgenda sont actuellement stockées localement dans :

```text
data/raw/evenements-publics-openagenda.csv
```

Le fichier brut est volontairement exclu du dépôt Git afin d'éviter de
versionner un fichier de données volumineux.

La reproductibilité du projet reposera sur :

- les scripts de traitement ;
- les règles de filtrage documentées ;
- les dépendances ;
- le pipeline de vectorisation ;
- la possibilité de reconstruire l'index FAISS.

Le nettoyage et la validation détaillée des données seront réalisés pendant
l'étape 2.

---

## Installation

### Prérequis

Avant de lancer le projet, les outils suivants doivent être installés :

- Git ;
- Docker Desktop ;
- Docker Compose.

Docker doit fonctionner avec des conteneurs Linux.

Pour vérifier l'installation :

```powershell
git --version
docker --version
docker compose version
docker info --format '{{.OSType}}'
```

La dernière commande doit retourner :

```text
linux
```

---

## Configuration de Mistral

Le projet utilise l'API Mistral.

Un modèle du fichier de configuration est fourni :

```text
.env.example
```

Créer le fichier local `.env` avec :

```powershell
Copy-Item .env.example .env
```

Le contenu du fichier doit suivre cette structure :

```dotenv
MISTRAL_API_KEY=votre_cle_api_mistral
```

La véritable clé API ne doit jamais être enregistrée dans Git.

Le fichier `.env` est donc ajouté au `.gitignore`.

---

## Gestion des dépendances

Les dépendances Python du projet sont enregistrées dans :

```text
requirements.txt
```

Les principales dépendances sont :

```text
langchain
langchain-community
langchain-mistralai
mistralai
faiss-cpu
numpy
python-dotenv
```

Les versions utilisées sont fixées ou contraintes afin de garantir la
reproductibilité de l'environnement.

---

## Construction de l'environnement Docker

Depuis la racine du projet :

```powershell
docker compose build
```

Cette commande :

1. utilise Python 3.12 ;
2. crée un environnement virtuel Python dans le conteneur ;
3. installe les dépendances de `requirements.txt` ;
4. vérifie leur compatibilité avec `pip check`.

---

## Vérification de l'environnement

Pour vérifier que l'environnement est correctement configuré :

```powershell
docker compose run --rm rag
```

Le script contrôle notamment :

- Python ;
- LangChain ;
- LangChain Community ;
- l'intégration LangChain/Mistral ;
- le SDK Mistral ;
- NumPy ;
- FAISS CPU ;
- la création d'un petit index FAISS ;
- une recherche de similarité dans cet index.

Résultat obtenu :

```text
============================================================
PULS-EVENTS RAG - VERIFICATION ENVIRONNEMENT
============================================================
Python              : 3.12.14
LangChain           : 1.4.0
LangChain Community : 0.4.2
LangChain Mistral   : 1.1.6
Mistral SDK         : 2.9.4
FAISS CPU           : 1.15.0
NumPy               : 2.5.3

Test FAISS CPU       : OK
Imports LangChain    : OK
Integration Mistral  : OK

ENVIRONNEMENT LOCAL : OK
```

---

## Vérification de l'accès à l'API Mistral

Après avoir configuré la clé dans `.env`, exécuter :

```powershell
docker compose run --rm rag python scripts/check_mistral_access.py
```

Résultat obtenu :

```text
============================================================
PULS-EVENTS RAG - VERIFICATION API MISTRAL
============================================================
Modeles accessibles : 46

ACCES MISTRAL : OK
```

Cette vérification confirme que :

- la clé API est correctement chargée ;
- l'authentification auprès de Mistral fonctionne ;
- l'environnement Docker peut communiquer avec l'API Mistral.

---

## Reproductibilité

L'environnement peut être reconstruit à partir des fichiers :

```text
Dockerfile
compose.yaml
requirements.txt
.env.example
```

Pour reconstruire l'environnement sur une nouvelle machine :

```powershell
git clone <URL_DU_DEPOT>
cd 04_Puls_Events_RAG
Copy-Item .env.example .env
```

Ajouter ensuite une clé Mistral valide dans `.env`, puis :

```powershell
docker compose build
docker compose run --rm rag
```

---

## Sécurité

Les éléments sensibles ou générés ne doivent pas être versionnés.

Sont notamment exclus de Git :

```text
.env
data/raw/*.csv
data/interim/*
data/processed/*
vectorstore/*
reports/*
```

Les fichiers `.gitkeep` permettent néanmoins de conserver la structure des
dossiers vides dans le dépôt.

---

## Tests

Le projet intégrera progressivement plusieurs niveaux de tests.

### Tests de l'environnement

Déjà disponibles :

```text
scripts/check_environment.py
scripts/check_mistral_access.py
```

### Tests des données

Pendant l'étape de pré-processing, des tests Python devront vérifier notamment :

- la conformité du périmètre géographique ;
- la conformité de la période ;
- l'absence ou la gestion correcte des données manquantes ;
- la cohérence des données utilisées pour construire la base vectorielle.

### Tests du système RAG

Des tests seront ensuite ajoutés pour vérifier :

- la construction de l'index FAISS ;
- la récupération des documents ;
- la pertinence des résultats ;
- la génération des réponses ;
- la qualité globale du système RAG.

---

## Reconstruction de la base vectorielle

Une contrainte importante du projet est de pouvoir reconstruire la base
vectorielle à la demande.

Le pipeline final devra donc permettre d'exécuter successivement :

```text
Données OpenAgenda
        |
        v
Pré-processing
        |
        v
Données propres
        |
        v
Chunking
        |
        v
Embeddings
        |
        v
Construction de l'index FAISS
```

Les scripts correspondants seront développés au cours des prochaines étapes.

---

## Évaluation du système RAG

Un jeu de données de test contenant des couples :

```text
question / réponse annotée
```

sera créé pour évaluer la qualité du système.

Les réponses générées par le RAG pourront être comparées aux réponses
annotées afin de mesurer la qualité du POC et d'identifier les axes
d'amélioration.

---

## État d'avancement

### Étape 1 - Préparation de l'environnement

- [x] Git installé
- [x] Docker installé
- [x] Docker Compose installé
- [x] Conteneurs Linux activés
- [x] Python 3.12 opérationnel
- [x] Environnement virtuel configuré dans Docker
- [x] LangChain installé
- [x] FAISS CPU installé
- [x] SDK Mistral installé
- [x] Intégration LangChain/Mistral installée
- [x] Imports Python vérifiés
- [x] Recherche FAISS testée
- [x] Accès réel à l'API Mistral testé
- [x] Gestion des dépendances configurée
- [x] Environnement reproductible avec Docker
- [x] Documentation de l'environnement créée

**Statut : étape 1 techniquement fonctionnelle.**

### Étape 2 - Pré-processing des données OpenAgenda

À venir.

Les prochaines tâches concerneront notamment :

- l'analyse du fichier OpenAgenda ;
- le choix et la validation du périmètre géographique ;
- le filtrage temporel ;
- le nettoyage des données ;
- la sélection des colonnes utiles ;
- la création du jeu de données propre ;
- l'ajout des tests unitaires correspondants.

---

## Technologies

| Technologie | Utilisation |
|---|---|
| Python | Traitement des données et développement |
| Pandas | Pré-processing des données |
| LangChain | Orchestration du système RAG |
| Mistral AI | Embeddings et génération |
| FAISS CPU | Recherche vectorielle |
| Docker | Reproductibilité de l'environnement |
| Git | Versionnement du projet |

---

## Auteur

**Hassna EL-BOUSIYDY**

Projet réalisé dans le cadre de la formation **Data Engineer**.

---

## Statut du projet

🚧 **En cours de développement**

Étape actuelle :

```text
Étape 1 - Préparation de l'environnement
```

Prochaine étape :

```text
Étape 2 - Pré-processing des données OpenAgenda
```