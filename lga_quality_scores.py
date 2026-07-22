"""
Nigerian AVM Version 2 — Manual LGA Quality Scores
Adds a 1-10 prestige/quality score for each LGA as an independent location signal.
This is not biased by sample size like the LGA price encoding.
"""

# LGA quality scores — based on:
# - Property market prestige
# - Infrastructure quality
# - Security
# - Proximity to economic centres
# - General desirability

LGA_QUALITY_SCORES = {
    # Lagos — Ultra Premium (9-10)
    "Ikoyi":             10,
    "Victoria Island":   10,
    "Banana Island":     10,
    "Eko Atlantic":      10,
    "Lekki":             9,
    "Lekki Phase 1":     9,

    # Lagos — Premium (7-8)
    "Ikeja":             8,
    "Ikeja Gra":         8,
    "GRA Ikeja":         8,
    "Maryland":          7,
    "Magodo":            7,
    "Omole":             7,
    "Opebi":             7,
    "Allen":             7,
    "Oregun":            7,

    # Lagos — Mid-Upper (5-6)
    "Yaba":              6,
    "Gbagada":           6,
    "Surulere":          5,
    "Ojodu":             5,
    "Ojota":             5,
    "Ketu":              5,
    "Mile 12":           5,
    "Shomolu":           5,

    # Lagos — Mid-Market (3-4)
    "Ikorodu":           4,
    "Agege":             3,
    "Mushin":            3,
    "Oshodi":            3,
    "Isolo":             3,
    "Alimosho":          3,
    "Badagry":           3,

    # Abuja — Ultra Premium (9-10)
    "Maitama":           10,
    "Asokoro":           10,
    "Banana Island Abuja": 10,

    # Abuja — Premium (7-8)
    "Wuse 2":            9,
    "Jabi":              8,
    "Wuse":              7,
    "Garki":             7,
    "Guzape":            8,
    "Katampe":           7,

    # Abuja — Mid-Market (4-6)
    "Kubwa":             4,
    "Lugbe":             4,
    "Karu":              3,
    "Nyanya":            3,
    "Gwagwalada":        3,

    # Oyo (Ibadan)
    "Ibadan":            4,
    "GRA Ibadan":        7,
    "Bodija":            6,
    "Oluyole":           5,
    "Agodi":             6,

    # Default for unknown LGAs
    "DEFAULT":           4,
}


def get_lga_score(lga: str) -> int:
    """Get quality score for an LGA, with fuzzy matching."""
    if not lga:
        return LGA_QUALITY_SCORES["DEFAULT"]

    lga_clean = lga.strip().title()

    # Direct match
    if lga_clean in LGA_QUALITY_SCORES:
        return LGA_QUALITY_SCORES[lga_clean]

    # Partial match
    for key, score in LGA_QUALITY_SCORES.items():
        if key.lower() in lga_clean.lower() or lga_clean.lower() in key.lower():
            return score

    return LGA_QUALITY_SCORES["DEFAULT"]


def add_lga_scores_to_dataset(input_file: str, output_file: str):
    import pandas as pd

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} listings from {input_file}")

    df["lga_quality_score"] = df["lga"].apply(get_lga_score)

    # Show distribution
    print(f"\nLGA quality score distribution:")
    print(df["lga_quality_score"].value_counts().sort_index())

    print(f"\nSample LGA scores:")
    sample = df[["lga", "lga_quality_score"]].drop_duplicates().sort_values(
        "lga_quality_score", ascending=False
    ).head(20)
    print(sample.to_string(index=False))

    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    return df


if __name__ == "__main__":
    # Test the scoring
    test_lgas = ["Lekki", "Maitama", "Surulere", "Ikorodu", "Ikoyi", "Yaba", "Kubwa"]
    print("LGA Quality Score Test:")
    print("-" * 30)
    for lga in test_lgas:
        score = get_lga_score(lga)
        bar = "█" * score
        print(f"  {lga:<20} {score:2d}/10  {bar}")

    print("\nApplying to dataset...")
    add_lga_scores_to_dataset(
        "nigeria_property_raw_v2.csv",
        "nigeria_property_raw_v2_scored.csv"
    )

