import pandas as pd, numpy as np

df = pd.read_csv("datasets/cirrhosis.csv")
df = df.dropna(subset=["Stage"])
for c in ["Cholesterol","Copper","Alk_Phos","SGOT","Tryglicerides","Platelets","Prothrombin","Bilirubin","Albumin"]:
    df[c] = df[c].fillna(df[c].median())

df["Age_yrs"] = df["Age"] / 365.25

print("Per-stage statistics:")
for stage in [1, 2, 3, 4]:
    s = df[df["Stage"] == stage]
    print(f"\n--- Stage {int(stage)} (n={len(s)}) ---")
    for col in ["Age_yrs","Bilirubin","Albumin","SGOT","Alk_Phos","Platelets","Prothrombin","Cholesterol","Copper","Tryglicerides"]:
        print(f"  {col:16s}: {s[col].mean():.2f} ± {s[col].std():.2f}  [{s[col].min():.1f} – {s[col].max():.1f}]")
    female_pct = 100 * (s["Sex"] == "F").mean()
    ascites_pct = 100 * (s["Ascites"] == "Y").mean()
    hepato_pct = 100 * (s["Hepatomegaly"] == "Y").mean()
    spider_pct = 100 * (s["Spiders"] == "Y").mean()
    edema_n = 100 * (s["Edema"] == "N").mean()
    edema_s = 100 * (s["Edema"] == "S").mean()
    edema_y = 100 * (s["Edema"] == "Y").mean()
    print(f"  Female={female_pct:.0f}%  Ascites={ascites_pct:.0f}%  Hepatomegaly={hepato_pct:.0f}%  Spiders={spider_pct:.0f}%")
    print(f"  Edema: N={edema_n:.0f}%  S={edema_s:.0f}%  Y={edema_y:.0f}%")
