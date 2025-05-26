# Générateur de Fantômes de Shepp-Logan 3D Aléatoires

## Table des Matières
1.  [Introduction](#introduction)
2.  [Prérequis](#prérequis)
3.  [Structure du Code](#structure-du-code)
4.  [Fonctionnalités Clés](#fonctionnalités-clés)
5.  [Description des Fonctions Principales](#description-des-fonctions-principales)
    *   [5.1 `shepp_logan_3d`](#51-shepp_logan_3d)
    *   [5.2 `generate_randomized_ellipsoid_parameters`](#52-generate_randomized_ellipsoid_parameters)
    *   [5.3 `generate_dataset`](#53-generate_dataset)
    *   [5.4 `display_slices_mpl`](#54-display_slices_mpl)
    *   [5.5 `visualize_with_vedo`](#55-visualize_with_vedo)
6.  [Utilisation (Exemple)](#utilisation-exemple)
7.  [Paramètres de Randomisation Clés](#paramètres-de-randomisation-clés)
8.  [Sortie Attendue](#sortie-attendue)
9.  [Personnalisation](#personnalisation)

## 1. Introduction

Ce script Python est conçu pour générer un dataset de fantômes de Shepp-Logan 3D. Les fantômes sont créés en superposant plusieurs ellipsoïdes d'intensités variables. Le script permet une randomisation significative des paramètres des ellipsoïdes (taille, position, orientation, intensité) tout en garantissant que les ellipsoïdes internes restent contenus à l'intérieur de l'ellipsoïde externe principal. Il fournit également des outils de visualisation 2D (coupes) et 3D.

## 2. Prérequis

Assurez-vous d'avoir Python 3 installé, ainsi que les bibliothèques suivantes :

*   **NumPy**: Pour les opérations numériques et la manipulation de tableaux.
    ```bash
    pip install numpy
    ```
*   **SciPy**: Utilisé pour les transformations de rotation.
    ```bash
    pip install scipy
    ```
*   **Matplotlib**: Pour la visualisation 2D des coupes (optionnel mais recommandé pour une vérification rapide).
    ```bash
    pip install matplotlib
    ```
*   **Vedo**: Pour la visualisation 3D interactive (optionnel mais fortement recommandé pour une inspection détaillée).
    ```bash
    pip install vedo
    ```
*   Les modules standards `json`, `random`, `os`, et `pathlib` sont également utilisés.

## 3. Structure du Code

Le code est organisé en plusieurs fonctions principales :
*   Fonctions de génération de fantôme :
    *   `shepp_logan_3d`: Crée un unique fantôme 3D à partir d'une liste de paramètres d'ellipsoïdes.
    *   `generate_randomized_ellipsoid_parameters`: Génère aléatoirement les paramètres pour un ensemble d'ellipsoïdes, en assurant la cohérence et le confinement.
    *   `generate_dataset`: Orchestre la création de multiples fantômes et les enregistre sur disque.
*   Fonctions de visualisation :
    *   `display_slices_mpl`: Affiche des coupes 2D orthogonales du fantôme avec Matplotlib.
    *   `visualize_with_vedo`: Permet une visualisation 3D interactive du fantôme avec Vedo.
*   Bloc principal `if __name__ == "__main__":`: Fournit un exemple d'utilisation pour générer un petit dataset et visualiser certains des résultats.

## 4. Fonctionnalités Clés

*   **Génération de fantômes 3D de Shepp-Logan** de taille spécifiée (par défaut 128x128x128).
*   **Randomisation des paramètres des ellipsoïdes** :
    *   Nombre d'ellipsoïdes (configurable).
    *   Intensité de chaque ellipsoïde.
    *   Taille (demi-axes) de chaque ellipsoïde.
    *   Position (centre) de chaque ellipsoïde.
    *   Orientation (rotation) de chaque ellipsoïde.
*   **Confinement Garanti**: Mécanisme robuste pour s'assurer que tous les ellipsoïdes internes sont contenus à l'intérieur de l'ellipsoïde externe principal, avec une marge de sécurité.
*   **Placement Centralisé**: Les ellipsoïdes internes sont placés préférentiellement autour du centre de l'ellipsoïde englobant.
*   **Génération de Dataset**: Capacité à générer un grand nombre de fantômes uniques et à sauvegarder :
    *   Les données volumiques du fantôme au format `.npy`.
    *   Les paramètres des ellipsoïdes utilisés pour générer chaque fantôme au format `.json`.
*   **Visualisation**:
    *   Coupes 2D (axiale, sagittale, coronale) via Matplotlib.
    *   Rendu 3D interactif (isosurfaces, rendu volumique, coupes dynamiques) via Vedo.

## 5. Description des Fonctions Principales

### 5.1 `shepp_logan_3d`

```python
def shepp_logan_3d(size=128, ellipsoid_params_list=None):
    # ... code ...
