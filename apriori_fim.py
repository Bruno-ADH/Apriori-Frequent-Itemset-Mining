"""
TP3 - Apriori pour l'extraction d'itemsets frequents (FIM)

Usage:
    python apriori_fim.py
    python apriori_fim.py --min-support 0.25 --min-confidence 0.60
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections import Counter
from typing import Iterable

import pandas as pd


def load_transactions(path: str, separator: str = ";") -> list[frozenset[str]]:
    data = pd.read_csv(path)
    if "items" not in data.columns:
        raise ValueError("Le fichier CSV doit contenir une colonne 'items'.")

    transactions: list[frozenset[str]] = []
    for raw_items in data["items"].dropna():
        items = [item.strip().lower() for item in str(raw_items).split(separator)]
        cleaned_items = frozenset(item for item in items if item)
        if cleaned_items:
            transactions.append(cleaned_items)

    if not transactions:
        raise ValueError("Aucune transaction valide trouvee dans le dataset.")
    return transactions


def support_count(itemset: frozenset[str], transactions: list[frozenset[str]]) -> int:
    return sum(1 for transaction in transactions if itemset.issubset(transaction))


def generate_candidates(previous_itemsets: set[frozenset[str]], size: int) -> set[frozenset[str]]:
    candidates: set[frozenset[str]] = set()
    previous_list = sorted(previous_itemsets, key=lambda itemset: sorted(itemset))

    for left, right in itertools.combinations(previous_list, 2):
        candidate = left.union(right)
        if len(candidate) != size:
            continue

        # Propriete Apriori: tous les sous-ensembles de taille k-1 doivent etre frequents.
        subsets = itertools.combinations(candidate, size - 1)
        if all(frozenset(subset) in previous_itemsets for subset in subsets):
            candidates.add(candidate)

    return candidates


def apriori(
    transactions: list[frozenset[str]],
    min_support: float,
) -> dict[frozenset[str], tuple[int, float]]:
    transaction_count = len(transactions)
    min_count = max(1, int(transaction_count * min_support + 0.999999))

    item_counter: Counter[str] = Counter()
    for transaction in transactions:
        item_counter.update(transaction)

    current_itemsets = {
        frozenset([item])
        for item, count in item_counter.items()
        if count >= min_count
    }

    frequent_itemsets: dict[frozenset[str], tuple[int, float]] = {}
    for itemset in current_itemsets:
        count = item_counter[next(iter(itemset))]
        frequent_itemsets[itemset] = (count, count / transaction_count)

    size = 2
    while current_itemsets:
        candidates = generate_candidates(current_itemsets, size)
        next_itemsets: set[frozenset[str]] = set()

        for candidate in candidates:
            count = support_count(candidate, transactions)
            if count >= min_count:
                next_itemsets.add(candidate)
                frequent_itemsets[candidate] = (count, count / transaction_count)

        current_itemsets = next_itemsets
        size += 1

    return frequent_itemsets


def powerset_non_empty_proper(itemset: frozenset[str]) -> Iterable[frozenset[str]]:
    items = list(itemset)
    for size in range(1, len(items)):
        for subset in itertools.combinations(items, size):
            yield frozenset(subset)


def generate_rules(
    frequent_itemsets: dict[frozenset[str], tuple[int, float]],
    min_confidence: float,
) -> pd.DataFrame:
    rules: list[dict[str, object]] = []

    for itemset, (_, itemset_support) in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        for antecedent in powerset_non_empty_proper(itemset):
            consequent = itemset.difference(antecedent)
            antecedent_support = frequent_itemsets[antecedent][1]
            consequent_support = frequent_itemsets[consequent][1]

            confidence = itemset_support / antecedent_support
            lift = confidence / consequent_support if consequent_support else 0.0

            if confidence >= min_confidence:
                rules.append(
                    {
                        "antecedent": ", ".join(sorted(antecedent)),
                        "consequent": ", ".join(sorted(consequent)),
                        "support": round(itemset_support, 4),
                        "confidence": round(confidence, 4),
                        "lift": round(lift, 4),
                    }
                )

    columns = ["antecedent", "consequent", "support", "confidence", "lift"]
    if not rules:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rules).sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False],
    )


def itemsets_to_dataframe(
    frequent_itemsets: dict[frozenset[str], tuple[int, float]]
) -> pd.DataFrame:
    rows = [
        {
            "items": ", ".join(sorted(itemset)),
            "size": len(itemset),
            "support_count": count,
            "support": round(support, 4),
        }
        for itemset, (count, support) in frequent_itemsets.items()
    ]
    return pd.DataFrame(rows).sort_values(
        by=["size", "support", "items"],
        ascending=[True, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TP3 - Apriori Frequent Itemset Mining")
    parser.add_argument("--data", default="data/market_basket_transactions.csv", help="Chemin du CSV")
    parser.add_argument("--min-support", type=float, default=0.20, help="Support minimal entre 0 et 1")
    parser.add_argument("--min-confidence", type=float, default=0.60, help="Confiance minimale entre 0 et 1")
    parser.add_argument("--output-dir", default="outputs", help="Dossier de sortie")
    args = parser.parse_args()

    if not 0 < args.min_support <= 1:
        raise ValueError("--min-support doit etre dans l'intervalle ]0, 1].")
    if not 0 < args.min_confidence <= 1:
        raise ValueError("--min-confidence doit etre dans l'intervalle ]0, 1].")

    os.makedirs(args.output_dir, exist_ok=True)

    transactions = load_transactions(args.data)
    frequent_itemsets = apriori(transactions, args.min_support)
    itemsets_df = itemsets_to_dataframe(frequent_itemsets)
    rules_df = generate_rules(frequent_itemsets, args.min_confidence)

    itemsets_path = os.path.join(args.output_dir, "frequent_itemsets.csv")
    rules_path = os.path.join(args.output_dir, "association_rules.csv")

    itemsets_df.to_csv(itemsets_path, index=False)
    rules_df.to_csv(rules_path, index=False)

    print("TP3 - Apriori Frequent Itemset Mining")
    print(f"Dataset: {args.data}")
    print(f"Nombre de transactions: {len(transactions)}")
    print(f"Support minimal: {args.min_support:.2f}")
    print(f"Confiance minimale: {args.min_confidence:.2f}")
    print(f"Itemsets frequents trouves: {len(itemsets_df)}")
    print(f"Regles d'association trouvees: {len(rules_df)}")

    print("\nTop itemsets frequents:")
    print(itemsets_df.head(15).to_string(index=False))

    print("\nTop regles d'association:")
    if rules_df.empty:
        print("Aucune regle ne respecte le seuil de confiance.")
    else:
        print(rules_df.head(15).to_string(index=False))

    print(f"\nItemsets sauvegardes: {itemsets_path}")
    print(f"Regles sauvegardees: {rules_path}")


if __name__ == "__main__":
    main()
