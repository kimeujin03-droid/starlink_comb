import pandas as pd
import numpy as np

DATA = "experiment4_annotations.csv"

df = pd.read_csv(DATA)

# -----------------------------
# 기본 정리
# -----------------------------
# 문자열 컬럼 정리
for col in ["comb_present", "starlink_like", "mask_issue", "urgent_followup"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": np.nan, "nan": np.nan})
        )

# 숫자 컬럼 정리
for col in ["decision_time_sec", "reviewer_confidence"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n[Columns]")
print(df.columns.tolist())

print("\n[Preview]")
print(df.head())

# -----------------------------
# agreement rate
# sample_id + condition 별로 annotator들끼리
# 같은 답을 했는지 계산
# -----------------------------
def agreement_rate_for_group(series: pd.Series) -> float:
    vals = series.dropna()
    if len(vals) <= 1:
        return np.nan
    # 모두 같은 답이면 1, 아니면 0
    return float(vals.nunique() == 1)

agreement_df = (
    df.groupby(["sample_id", "condition"])["comb_present"]
      .agg(agreement_rate_for_group)
      .reset_index(name="agreement")
)

print("\nAgreement table")
print(agreement_df.head(10))

print("\nMean agreement rate by condition")
print(agreement_df.groupby("condition")["agreement"].mean())

# -----------------------------
# annotation time
# -----------------------------
if "decision_time_sec" in df.columns:
    print("\nMean decision time")
    print(df.groupby("condition")["decision_time_sec"].mean())
else:
    print("\nNo decision_time_sec column found.")

# -----------------------------
# reviewer confidence
# -----------------------------
if "reviewer_confidence" in df.columns:
    print("\nReviewer confidence")
    print(df.groupby("condition")["reviewer_confidence"].mean())
else:
    print("\nNo reviewer_confidence column found.")

# -----------------------------
# ambiguous subset analysis
# 기존 gold label 기준
# -----------------------------
if "gold_annotation_label" in df.columns:
    amb_df = df[df["gold_annotation_label"].astype(str).str.lower() == "ambiguous"].copy()

    if len(amb_df) > 0:
        amb_agreement_df = (
            amb_df.groupby(["sample_id", "condition"])["comb_present"]
                  .agg(agreement_rate_for_group)
                  .reset_index(name="agreement")
        )

        print("\nAmbiguous-case agreement")
        print(amb_agreement_df.groupby("condition")["agreement"].mean())
    else:
        print("\nNo ambiguous cases found.")
else:
    print("\nNo gold_annotation_label column found.")

# -----------------------------
# optional: save summary tables
# -----------------------------
agreement_df.to_csv("exp4_agreement_table.csv", index=False)

summary_rows = []

if "decision_time_sec" in df.columns:
    time_summary = df.groupby("condition")["decision_time_sec"].mean()
    for cond, val in time_summary.items():
        summary_rows.append({
            "metric": "mean_decision_time_sec",
            "condition": cond,
            "value": val
        })

if "reviewer_confidence" in df.columns:
    conf_summary = df.groupby("condition")["reviewer_confidence"].mean()
    for cond, val in conf_summary.items():
        summary_rows.append({
            "metric": "mean_reviewer_confidence",
            "condition": cond,
            "value": val
        })

agr_summary = agreement_df.groupby("condition")["agreement"].mean()
for cond, val in agr_summary.items():
    summary_rows.append({
        "metric": "mean_agreement_rate",
        "condition": cond,
        "value": val
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("exp4_summary_metrics.csv", index=False)

print("\nSaved:")
print("- exp4_agreement_table.csv")
print("- exp4_summary_metrics.csv")