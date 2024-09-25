# Multilayer Perceptron Project

## Overview

Ce projet est une introduction aux réseaux de neurones artificiels à travers l'implémentation d'un perceptron multicouche (MLP) pour classer le cancer du sein en tant que **malin** ou **bénin**, basé sur le jeu de données **Wisconsin Breast Cancer**.

## Structure du projet

Le projet nécessite de construire et d'entraîner un réseau de neurones à partir de zéro, sans utiliser de bibliothèques externes qui gèrent les réseaux de neurones. L'implémentation inclut :

1. **Traitement des données** : 
   - Le jeu de données est fourni et contient des caractéristiques qui décrivent les caractéristiques des noyaux cellulaires des échantillons de tissu mammaire.
   - **Prétraitez les données** avant l'entraînement et divisez-les en ensembles d'entraînement et de validation.

2. **Structure du réseau de neurones** :
   - Vous devez implémenter un réseau de neurones feedforward avec au moins **deux couches cachées**.
   - Chaque couche doit être constituée de perceptrons avec des **fonctions d'activation** personnalisables.
   - La couche de sortie utilisera la **fonction softmax** pour la classification binaire (malin vs bénin).

3. **Entraînement** :
   - Utilisez la **rétropropagation** et la **descente de gradient** pour entraîner le modèle.
   - Affichez les métriques d'entraînement et de validation à chaque époque pour suivre les progrès.

4. **Évaluation** :
   - Évaluez le modèle en utilisant la **perte d'entropie croisée binaire**.
   - Implémentez des courbes d'apprentissage pour visualiser la **perte** et la **précision** pendant l'entraînement.

## Programmes obligatoires

Vous devrez créer trois programmes principaux :

1. **Programme de séparation des données** : 
   - Ce programme divisera le jeu de données en deux parties : une pour l'entraînement et une pour la validation.
   - Pour exécuter le programme :  
     ```bash
     python load.py
     ```

2. **Programme d'entraînement** :
   - Entraînez le MLP en utilisant la rétropropagation et la descente de gradient.
   - **Sauvegardez le modèle** (topologie du réseau et poids) après l'entraînement.
   - Pour exécuter le programme :  
     ```bash
     python train.py
     ```

3. **Programme de prédiction** :
   - Chargez le modèle sauvegardé, effectuez des prédictions et évaluez sa précision.
   - Pour exécuter le programme :  
     ```bash
     python predict.py
     ```

## Soumission

Soumettez les éléments suivants :
- **Code source** pour les programmes de séparation des données, d'entraînement et de prédiction.
- **Un Makefile** pour la compilation (si applicable).
- Toute documentation pertinente expliquant la conception et le fonctionnement de votre projet.

Assurez-vous que le dépôt de votre projet est bien organisé et respecte les directives de soumission.
