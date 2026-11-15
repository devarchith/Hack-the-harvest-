"""Field-year corn labels from the BARC FarmLogic events; compare with the evaluation labels.
args: <events csv> <eval gpkg> <kraatz label csv> <out field-year csv>"""
import sys
import pandas as pd, geopandas as gpd
ev = pd.read_csv(sys.argv[1], parse_dates=["date"])
odd = ev[((ev.sec == "harvest") & (((ev.cls == "Grain") & (ev.machine != "Combine")) | ((ev.cls == "Silage") & (ev.machine != "Chopper"))))]
print("harvest records where crop and machine disagree:"); print(odd[["idn", "date", "cls", "machine", "split", "body"]].to_string())
w = ev[ev.split == "Whole Field"]
def agg(g):
    p = g[g.sec == "plant"]; h = g[g.sec == "harvest"]
    pc = "/".join(sorted(set(p.cls))) or None; hc = "/".join(sorted(set(h.cls))) or None
    return pd.Series(dict(plant=pc, harvest=hc, plant_date=p.date.min(), harvest_date=h.date.min(),
                          harvest_machine="/".join(sorted(set(h.machine))) or None))
fy = w.groupby(["idn", "year"]).apply(agg).reset_index()
fy = fy[~fy.plant.isin(["Sweet"]) | fy.harvest.notna()]
fy["label"] = fy.harvest.fillna(fy.plant)
b = gpd.read_file(sys.argv[2]); b["idn"] = b["FARM"].str.replace(" ", "") + "_" + b["FIELD"].astype(str)
long = b.melt(id_vars=["idn"], value_vars=["Corn_2021", "Corn_2022", "Corn_2023"], var_name="y", value_name="eval")
long["year"] = long.y.str[-4:].astype(int); long = long.dropna(subset=["eval"])
m = long.merge(fy, on=["idn", "year"], how="left")
print("\nEvaluation labels 2021-2023 vs farm reports (eval rows:", len(m), ")")
print("plant record:"); print(pd.crosstab(m["eval"], m.plant.fillna("no record")).to_string())
print("harvest record:"); print(pd.crosstab(m["eval"], m.harvest.fillna("no record")).to_string())
print("eval rows with no field report:", sorted(m.loc[m.plant.isna() & m.harvest.isna(), "idn"].unique()))
dis = m[(m.plant.notna() & (m.plant != m["eval"])) | (m.harvest.notna() & (m.harvest != m["eval"]))]
print("disagreements:"); print(dis[["idn", "year", "eval", "plant", "harvest", "harvest_machine", "harvest_date"]].to_string())
# planted silage, harvested grain (changed use)
ch = fy[(fy.plant.notna()) & (fy.harvest.notna()) & (fy.plant != fy.harvest)]
print("\nchanged use (plant != harvest):"); print(ch[["idn", "year", "plant", "harvest", "harvest_machine", "harvest_date"]].to_string())
k = pd.read_csv(sys.argv[3]); k = k[k.year <= 2020]
kk = k.merge(fy, on=["idn", "year"], how="left", suffixes=("_k", ""))
kk["lab_k"] = kk["hd"].fillna(kk["pd"])
print("\nZenodo 2017-2020 label vs field-report label (where both):"); print(pd.crosstab(kk.lab_k, kk.label.fillna("no report")).to_string())
fy["in_eval_gpkg"] = fy.idn.isin(set(b.idn))
fy.to_csv(sys.argv[4], index=False)
print("\nfield-years by year and final label (harvest if present, else plant):")
print(pd.crosstab(fy.year, fy.label, margins=True).to_string())
