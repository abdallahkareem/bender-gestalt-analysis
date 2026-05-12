import pandas as pd

df = pd.read_csv("bender_analysis_report.csv")

new_df = df.groupby("student").agg({
    "hu_distance": "mean",
    "hausdorff_distance": "mean",
    "area_ratio": "mean",
    "errors": lambda x : x.mode().iloc[0] if not x.mode().empty else None
})

new_df["score"] = 0.4*new_df["hu_distance"] + 0.3*new_df["hausdorff_distance"] + 0.3*new_df["area_ratio"]

new_df["score_category"] = pd.cut(new_df["score"], bins=3, labels=["Low", "Medium", "High"])

def map_disease(score_category, errors):
    errors = errors.lower()
    category = score_category.lower()
    diagnoses = []
    
    if "no significant errors" in errors:
        return "No Mental Illness"

    if "perseveration" in errors or ("integration failure" in errors and category == "high"):
        diagnoses.append("Brain-Damage")
    
    if "integration failure" in errors and category == "medium":
        diagnoses.append("Mental-retardation")
    
    if "distortion" in errors and "omission" in errors:
        diagnoses.append("Schizophrenia")

    if ("omission" in errors or "distortion" in errors) and len(diagnoses) == 0:
        diagnoses.append("Neurosis")
    elif "omission" in errors and ("brain-damage" in diagnoses or "schizophrenia" in diagnoses):
        diagnoses.append("Neurosis")

    if not diagnoses:
        return "Undetermined"
    
    return ", ".join(list(set(diagnoses)))
        

new_df["disease"] = new_df.apply(lambda row: map_disease(row["score_category"], row["errors"]), axis=1)
new_df.to_excel("disease_predictions.xlsx", columns=["disease"], index=True)