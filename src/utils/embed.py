"""
Frequent classification itemsets + t-SNE on rule covers (Mushroom dataset).

Requirements:
    pip install kaggle pandas scikit-learn mlxtend matplotlib

You also need Kaggle API credentials (~/.kaggle/kaggle.json) if you want
the automatic download to work, OR just skip the download part and put
'mushrooms.csv' in the working directory.
"""

import os
import zipfile

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from mlxtend.frequent_patterns import apriori


# ---------------------------------------------------------------------
# 1. (Optional) Download a simple categorical dataset from Kaggle
#    We'll use: uciml/mushroom-classification
# ---------------------------------------------------------------------
def download_mushroom_dataset():
    """
    Download the Mushroom Classification dataset from Kaggle
    and extract mushrooms.csv into the current directory.

    If download fails (no Kaggle API), you can just manually
    put mushrooms.csv next to this script.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[WARN] kaggle package not installed. Skipping download.")
        return

    csv_path = "mushrooms.csv"
    if os.path.exists(csv_path):
        print(f"[INFO] {csv_path} already exists, skipping download.")
        return

    print("[INFO] Downloading mushroom dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()

    # dataset: uciml/mushroom-classification
    api.dataset_download_files(
        "uciml/mushroom-classification",
        path=".",
        quiet=False
    )

    # Find and unzip the downloaded file (typically mushroom-classification.zip)
    for fname in os.listdir("."):
        if fname.endswith(".zip") and "mushroom" in fname.lower():
            print(f"[INFO] Extracting {fname} ...")
            with zipfile.ZipFile(fname, "r") as zf:
                zf.extractall(".")
            os.remove(fname)
            break

    if not os.path.exists(csv_path):
        print("[WARN] mushrooms.csv was not found after extraction. "
              "Check the zip contents or download manually.")


# ---------------------------------------------------------------------
# 2. Load and prepare the data (class column + one-hot of other features)
# ---------------------------------------------------------------------
def load_and_prepare_data(csv_path="mushrooms.csv"):
    """
    Load the mushroom dataset, separate the class column ('p', 'e'),
    and one-hot encode all *other* categorical features.

    Returns
    -------
    df : pandas.DataFrame
        Original dataset.
    encoded_all : pandas.DataFrame
        One-hot encoded features (no class column), shape (n_samples, n_items).
    y : pandas.Series
        Class labels ('p' or 'e'), length n_samples.
    """
    if not os.path.exists(csv_path):
        download_mushroom_dataset()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Download failed or file missing. "
            "Place the CSV in the working directory."
        )

    df = pd.read_csv(csv_path)

    if "class" not in df.columns:
        raise ValueError("Expected a 'class' column in mushrooms.csv.")

    y = df["class"].astype("category")
    X = df.drop(columns=["class"]).astype("category")

    # One-hot encode all non-class features
    encoded_all = pd.get_dummies(X)
    encoded_all = encoded_all.astype(bool)

    return df, encoded_all, y


# ---------------------------------------------------------------------
# 3. Mine frequent itemsets separately for each class (classification rules)
# ---------------------------------------------------------------------
def mine_classification_itemsets(encoded_all, y, min_support=0.2):
    """
    Mine frequent itemsets *conditionally* on each class value.

    For each class c ∈ {p, e}, we:
      - Restrict to rows with y == c
      - Mine frequent itemsets on that subset
      - Each itemset becomes a classification rule: itemset => class=c

    We return a DataFrame of "rules", each with:
      - 'class'                  : target class label ('p' or 'e')
      - 'antecedents'           : frozenset of dummy feature names
      - 'support_conditional'   : support within class subset (P(X | class))
      - 'support_global'        : support on full dataset (P(X ∧ class))
      - 'count'                 : number of samples in full dataset with X ∧ class

    Parameters
    ----------
    encoded_all : DataFrame
        One-hot encoded features for all samples.
    y : Series
        Class labels.
    min_support : float
        Minimum support within each class subset.

    Returns
    -------
    rules : DataFrame
    """
    all_rules = []
    n_total = len(y)

    for cls in sorted(y.unique()):
        mask_cls = (y == cls)
        encoded_cls = encoded_all[mask_cls]
        n_cls = mask_cls.sum()

        print(f"[INFO] Mining frequent itemsets for class={cls} "
              f"on {n_cls} samples (min_support={min_support})...")

        # apriori support is fraction w.r.t. encoded_cls
        freq = apriori(
            encoded_cls,
            min_support=min_support,
            use_colnames=True
        )

        # Rename & compute global stats
        freq = freq.rename(columns={
            "support": "support_conditional",
            "itemsets": "antecedents"
        })

        # Count within class; use integer rounding
        freq["count_in_class"] = (freq["support_conditional"] * n_cls).round().astype(int)

        # Global support of (X ∧ class=c) is count_in_class / n_total
        freq["support_global"] = freq["count_in_class"] / n_total

        # The classification rule label
        freq["class"] = cls

        # Optional: total number of items in the antecedent
        freq["len_antecedent"] = freq["antecedents"].apply(len)

        # Reorder columns
        freq = freq[
            ["class", "antecedents", "len_antecedent",
             "support_conditional", "support_global", "count_in_class"]
        ]

        print(f"[INFO] Found {len(freq)} frequent itemsets for class={cls}.")
        all_rules.append(freq)

    if not all_rules:
        return pd.DataFrame()

    rules = pd.concat(all_rules, ignore_index=True)
    print(f"[INFO] Total frequent classification itemsets: {len(rules)}")
    return rules


# ---------------------------------------------------------------------
# 4. Compute rule covers and build a cover matrix
# ---------------------------------------------------------------------
def compute_rule_covers(rules, encoded_all):
    """
    Compute the cover of each classification rule antecedent, over the
    FULL dataset (all classes).

    For each rule (itemset => class=c), the cover is:
      { i : all items in 'antecedents' are present in sample i }

    Parameters
    ----------
    rules : DataFrame
        Output of mine_classification_itemsets.
    encoded_all : DataFrame
        One-hot encoded features of shape (n_samples, n_items).

    Returns
    -------
    cover_matrix : np.ndarray (n_rules, n_samples)
        Binary matrix where cover_matrix[k, i] = 1 if sample i is in the
        cover of rule k, 0 otherwise.
    """
    n_samples = encoded_all.shape[0]
    n_rules = len(rules)
    print(f"[INFO] Computing covers for {n_rules} rules over {n_samples} samples...")
    cover_matrix = np.zeros((n_rules, n_samples), dtype=np.uint8)
    columns = encoded_all.columns

    for k, row in rules.iterrows():
        antecedent_items = list(row["antecedents"])  # dummy column names

        mask = np.ones(n_samples, dtype=bool)
        for item in antecedent_items:
            if item not in columns:
                # Should not happen, but be robust
                print(f"[WARN] Antecedent item {item} not found in encoded columns.")
                continue
            mask &= encoded_all[item].values

        cover_matrix[k, mask] = 1

    # Remove duplicate covers (keep first occurrence of each unique cover).
    # This also updates the `rules` DataFrame in-place to match the returned matrix.
    if n_rules > 1:
        # np.unique with axis=0 returns the indices of the first occurrences
        unique_rows, unique_idx = np.unique(
            np.ascontiguousarray(cover_matrix), axis=0, return_index=True
        )
        if len(unique_idx) < n_rules:
            keep_idx = np.sort(unique_idx)
            drop_idx = np.setdiff1d(np.arange(n_rules), keep_idx)

            print(f"[INFO] Found {len(drop_idx)} duplicate covers, removing duplicates...")

            # Drop duplicate rules from the DataFrame (in-place) and reindex so indices match rows
            try:
                rules.drop(index=drop_idx, inplace=True)
                rules.reset_index(drop=True, inplace=True)
            except Exception:
                # Fallback: if indices are not the default 0..n-1, rebuild the DataFrame
                rules = rules.iloc[keep_idx].reset_index(drop=True)

            cover_matrix = cover_matrix[keep_idx]

    return cover_matrix


# ---------------------------------------------------------------------
# 5. Run t-SNE on the cover vectors (each rule = one point)
# ---------------------------------------------------------------------
def tsne_on_covers(cover_matrix, n_components=3, random_state=0, perplexity=30.0):
    """
    Run t-SNE on the rule covers.

    Each rule is represented by its cover vector (length n_samples),
    i.e. a binary vector telling which samples the rule covers.

    Parameters
    ----------
    cover_matrix : np.ndarray (n_rules, n_samples)
    n_components : int
    random_state : int
    perplexity : float

    Returns
    -------
    embedding : np.ndarray (n_rules, n_components)
        Low-dimensional t-SNE embedding of the rules.
    """
    n_rules = cover_matrix.shape[0]
    print(f"[INFO] Running t-SNE on {n_rules} rules (dim={cover_matrix.shape[1]})...")

    X = cover_matrix.astype(np.float32)

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=min(perplexity, max(5.0, (n_rules - 1) / 3.0)),
        init="pca",
        learning_rate="auto"
    )
    embedding = tsne.fit_transform(X)
    return embedding


# ---------------------------------------------------------------------
# 6. Main script
# ---------------------------------------------------------------------
def main():
    # 1) Load data (df not used later, but kept if you want to inspect)
    df, encoded_all, y = load_and_prepare_data()

    # 2) Mine frequent classification itemsets per class (no confidence filtering)
    rules = mine_classification_itemsets(encoded_all, y, min_support=0.5)
    print("[INFO] Sample rules (classification itemsets):")
    print(rules.head())
    print(f"[INFO] Number of rules: {len(rules)}")

    if rules.empty:
        print("[WARN] No frequent itemsets found with this threshold.")
        return

    # 3) Compute covers of rule antecedents on the full dataset
    cover_matrix = compute_rule_covers(rules, encoded_all)

    # 4) t-SNE on covers
    embedding = tsne_on_covers(cover_matrix, n_components=3, random_state=42)

    # 5) Attach embedding to rules dataframe
    rules["tsne_0"] = embedding[:, 0]
    rules["tsne_1"] = embedding[:, 1]
    rules["tsne_2"] = embedding[:, 2]

    # 6) Save results
    rules.to_csv("classification_itemsets_with_tsne.csv", index=False)
    print("[INFO] Saved classification_itemsets_with_tsne.csv with t-SNE coordinates.")

    # Optional: quick 3D scatter plot of the rules
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Color by class if you want
        classes = rules["class"].values
        unique_classes = sorted(set(classes))
        # Map class -> int so that matplotlib can color them
        class_to_int = {c: i for i, c in enumerate(unique_classes)}
        colors = [class_to_int[c] for c in classes]

        scatter = ax.scatter(
            rules["tsne_0"], rules["tsne_1"], rules["tsne_2"],
            s=10, c=colors
        )
        ax.set_xlabel("t-SNE 0")
        ax.set_ylabel("t-SNE 1")
        ax.set_zlabel("t-SNE 2")
        ax.set_title("t-SNE embedding of frequent classification itemsets")

        # Legend
        handles = []
        for c in unique_classes:
            handles.append(
                plt.Line2D(
                    [0], [0], marker="o", linestyle="",
                    label=f"class={c}",
                    markerfacecolor="k"  # color not exact but legend is ok
                )
            )
        ax.legend(handles=handles)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[INFO] matplotlib not installed, skipping 3D plot.")


if __name__ == "__main__":
    main()

