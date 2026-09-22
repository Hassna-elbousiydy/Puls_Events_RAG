# Jeu de référence Puls-Events

25 cas rédigés à partir du snapshot OpenAgenda utilisé par le projet.

Aucune réponse générée par Mistral n'est utilisée comme vérité terrain.
Les réponses de référence proviennent des informations structurées des
événements OpenAgenda : UID, titre, ville, lieu, dates et URL.

## Composition du jeu

Le jeu contient 25 questions couvrant plusieurs scénarios :

- recherche d'un événement précis ;
- recherche par ville ;
- exposition, concert, cinéma, patrimoine et autres thèmes culturels ;
- recherche par période ;
- plusieurs événements possibles ;
- absence de résultat dans le périmètre Pays de la Loire ;
- question utilisateur ambiguë.

Les cas avec événements de référence contiennent les UID OpenAgenda
attendus ainsi que leurs métadonnées sources.

Les cas d'absence ou d'ambiguïté sont évalués séparément afin de ne pas
leur attribuer artificiellement des métriques de retrieval classiques.

## Relecture humaine

Date de relecture : 2026-09-22

Nombre de cas relus : 25/25.

Méthode de relecture :

Chaque cas a été relu manuellement par l'auteure du projet.

Pour chaque question, la relecture a porté sur :

- la formulation de la question ;
- la réponse de référence ;
- le ou les UID attendus ;
- la ville ;
- le titre de l'événement ;
- les dates ;
- le lieu ;
- l'URL OpenAgenda ;
- la cohérence globale entre la question et la réponse attendue.

Pour les cas sans événement attendu, la cohérence de la réponse
d'abstention ou de clarification a également été vérifiée.

Les cas effectivement validés portent :

`"human_reviewed": true`

ainsi que la date et la méthode de relecture.

## Limites

Pour les questions ouvertes, les listes d'événements pertinents ne sont
pas nécessairement exhaustives. Une faible précision de retrieval peut
donc refléter l'existence d'autres événements pertinents qui n'ont pas
été inclus dans la référence annotée.

Les réponses annotées servent de vérité terrain pour comparer le sens
et les informations des réponses produites par le système RAG.
