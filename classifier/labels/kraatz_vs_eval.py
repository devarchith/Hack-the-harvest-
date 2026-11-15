"""Join the Kraatz (Zenodo 8087301) BARC planting/harvest records to the BARC evaluation polygons.
args: <dir with csvs> <eval gpkg> <kraatz shp> <out csv>"""
import sys, glob, os
import pandas as pd, geopandas as gpd
d, gpkg, shp, out = sys.argv[1:5]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
    kind = "hd" if "_hd_" in f else "pd"
    year = int(os.path.basename(f).split("_")[3])
    df = pd.read_csv(f)
    df = df[df["split"].str.strip() == "Whole Field"]
    df["cls"] = df["crop"].astype(str).str.split(" /").str[0].str.strip()
    df = df[df["cls"].isin(["Corn Field (FLDCRN)", "Corn Silage (CSIL)"])]
    df["label"] = df["cls"].map({"Corn Field (FLDCRN)": "Grain", "Corn Silage (CSIL)": "Silage"})
    df["date"] = pd.to_datetime(df[kind])
    g = df.groupby("idn").agg(label=("label", lambda s: "/".join(sorted(set(s)))), date=("date", "min"))
    for idn, r in g.iterrows():
        rows.append(dict(idn=idn, year=year, src=kind, label=r["label"], date=r["date"].date()))
lab = pd.DataFrame(rows)
wide = lab.pivot_table(index=["idn", "year"], columns="src", values="label", aggfunc="first").reset_index()
dates = lab[lab.src == "hd"].set_index(["idn", "year"])["date"].rename("harvest_date")
wide = wide.join(dates, on=["idn", "year"])
b = gpd.read_file(gpkg)
b["idn"] = b["FARM"].str.replace(" ", "") + "_" + b["FIELD"].astype(str)
k = gpd.read_file(shp)
wide["in_eval_gpkg"] = wide["idn"].isin(set(b["idn"]))
wide["in_kraatz_shp"] = wide["idn"].isin(set(k["id"]))
# 2021 agreement with the evaluation labels
ev = b[["idn", "Corn_2021", "Corn_2022", "Corn_2023", "ACRES"]]
w21 = wide[wide.year == 2021].merge(ev, on="idn", how="outer")
print("2021: Kraatz record vs eval Corn_2021")
print(pd.crosstab(w21["pd"].fillna("none"), w21["Corn_2021"].fillna("blank"), margins=True).to_string())
print(pd.crosstab(w21["hd"].fillna("none"), w21["Corn_2021"].fillna("blank"), margins=True).to_string())
dis = w21[(w21["Corn_2021"].notna()) & ((w21["pd"].fillna("none") != w21["Corn_2021"]) | (w21["hd"].fillna("none") != w21["Corn_2021"]))]
print("2021 eval-labelled fields where a record differs or is missing:"); print(dis[["idn", "pd", "hd", "harvest_date", "Corn_2021"]].to_string())
new = wide[wide.year <= 2020].copy()
new["label_final"] = new["hd"].fillna(new["pd"])
new["pd_hd_agree"] = (new["pd"] == new["hd"]) | new["hd"].isna() | new["pd"].isna()
print("\n2017-2020 field-years (harvest label if present, else planting label):")
print(pd.crosstab(new["year"], new["label_final"], margins=True).to_string())
print("planting vs harvest disagreements:"); print(new[~new["pd_hd_agree"]][["idn", "year", "pd", "hd", "harvest_date"]].to_string())
print("field-years on polygons in eval gpkg:", int(new["in_eval_gpkg"].sum()), "of", len(new), "; in Kraatz shp:", int(new["in_kraatz_shp"].sum()))
print("distinct fields 2017-2020:", new["idn"].nunique(), "; of which also eval-labelled 2021-23:",
      new["idn"].isin(set(b.loc[b[["Corn_2021","Corn_2022","Corn_2023"]].notna().any(axis=1), "idn"])).sum())
wide.sort_values(["year", "idn"]).to_csv(out, index=False)
print("wrote", out)
