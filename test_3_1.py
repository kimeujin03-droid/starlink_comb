import pandas as pd

# =========================
# 설정
# =========================
INPUT_CSV = "experiment2_annotations.csv"
OUTPUT_CSV = "experiment4_annotations.csv"

ANNOTATORS = ["annotator1", "annotator2"]   # 필요하면 3명으로 늘리기
CONDITIONS = ["A", "B"]                     # A = raw only, B = raw + QA report

# =========================
# 기존 human annotation 불러오기
# =========================
df = pd.read_csv(INPUT_CSV)

# sample_id만 중복 제거해서 사용
samples = df[["sample_id"]].drop_duplicates().copy()

# sample metadata도 같이 붙이고 싶으면 이걸 사용
meta_cols = [c for c in ["file", "pol", "band_class", "benchmark_split", "annotation_label", "artifact_type", "notes"] if c in df.columns]
sample_meta = df[["sample_id"] + meta_cols].drop_duplicates().copy()

# =========================
# Experiment 4용 annotation sheet 만들기
# =========================
rows = []

for _, row in sample_meta.iterrows():
    for annotator in ANNOTATORS:
        for condition in CONDITIONS:
            rows.append({
                "sample_id": row["sample_id"],
                "file": row.get("file", ""),
                "pol": row.get("pol", ""),
                "band_class": row.get("band_class", ""),
                "benchmark_split": row.get("benchmark_split", ""),
                "gold_annotation_label": row.get("annotation_label", ""),   # 분석 때 참고용, 실험 중 annotator에게는 가릴 수도 있음
                "artifact_type": row.get("artifact_type", ""),
                "condition": condition,
                "annotator": annotator,

                # 아래부터 annotator가 채울 칸
                "comb_present": "",
                "starlink_like": "",
                "mask_issue": "",
                "urgent_followup": "",
                "decision_time_sec": "",
                "reviewer_confidence": "",
                "review_notes": ""
            })

exp4_df = pd.DataFrame(rows)

# =========================
# 저장
# =========================
exp4_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved: {OUTPUT_CSV}")
print(f"Total rows: {len(exp4_df)}")
print(exp4_df.head(10))