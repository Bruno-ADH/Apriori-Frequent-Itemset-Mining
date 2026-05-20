# TP3 - Apriori Frequent Itemset Mining

## Objectif

Ce TP implemente l'algorithme **Apriori** pour extraire des itemsets frequents
et produire des regles d'association a partir de transactions.

Le cas d'application choisi est un panier d'achat. Chaque ligne du dataset
represente une transaction et contient les produits achetes ensemble.

## Dataset

Le fichier utilise est :

```text
data/market_basket_transactions.csv
```

Format :

```csv
transaction_id,items
T001,"bread;milk;butter"
T002,"bread;milk;eggs"
```

Les produits sont separes par `;`.

## Structure du dossier

```text
TP3/
├── data/
│   └── market_basket_transactions.csv
├── apriori_fim.py
├── requirements.txt
├── .gitignore
└── README.md
```

Les resultats generes sont places dans :

```text
outputs/
```

Ce dossier est ignore par Git.

## Installation

Depuis le dossier `TP3` :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloque l'activation :

```powershell
.\.venv\Scripts\activate.bat
```

## Execution

Execution avec les seuils par defaut :

```powershell
python apriori_fim.py
```

Execution avec des seuils personnalises :

```powershell
python apriori_fim.py --min-support 0.25 --min-confidence 0.70
```

Parametres :

- `--min-support` : proportion minimale de transactions contenant un itemset.
- `--min-confidence` : probabilite minimale du consequent sachant l'antecedent.
- `--data` : chemin vers le dataset CSV.
- `--output-dir` : dossier de sortie.

## Sorties

Le script genere :

```text
outputs/frequent_itemsets.csv
outputs/association_rules.csv
```

`frequent_itemsets.csv` contient :

- `items` : itemset frequent ;
- `size` : taille de l'itemset ;
- `support_count` : nombre de transactions contenant l'itemset ;
- `support` : proportion de transactions contenant l'itemset.

`association_rules.csv` contient :

- `antecedent` : partie gauche de la regle ;
- `consequent` : partie droite de la regle ;
- `support` : support de l'union antecedent + consequent ;
- `confidence` : probabilite du consequent sachant l'antecedent ;
- `lift` : force de l'association par rapport a l'independance.

## Rappels theoriques

Le support mesure la frequence d'apparition :

```text
support(A) = nombre de transactions contenant A / nombre total de transactions
```

La confiance d'une regle `A -> B` mesure :

```text
confidence(A -> B) = support(A union B) / support(A)
```

Le lift mesure si l'association est plus forte que le hasard :

```text
lift(A -> B) = confidence(A -> B) / support(B)
```

Interpretation du lift :

- `lift > 1` : association positive ;
- `lift = 1` : independance ;
- `lift < 1` : association negative.

## Dependances

Le TP utilise uniquement :

- `pandas`

L'algorithme Apriori est implemente directement en Python afin de montrer les
etapes de generation, filtrage et production des regles.
