"""Parse BARC FarmLogic field reports (BARCFarmMgmt.zip, Ag Data Commons 10.15482/USDA.ADC/28950746)
into corn plant and harvest events. args: <dir of .txt from the PDFs> <out events csv>"""
import sys, glob, os, re
import pandas as pd
DATE = re.compile(r"^\d{1,2}/\d{2}/\d{2}$")
SECTIONS = {"PLANT REPORT", "HARVEST REPORT", "HARVEST TICKET REPORT", "TILL REPORT", "NPK REPORT",
            "OTHER REPORT", "DAMAGE REPORT", "SUMMARY SPRAY REPORT"}
def corn_class(s):
    if "Corn, Silage" in s: return "Silage"
    if "Corn, Field" in s: return "Grain"
    if "Corn, Sweet" in s or "Sweet" in s and "Corn" in s: return "Sweet"
    return None
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.txt"))):
    idn = os.path.basename(f)[:-4]
    lines = [l.strip() for l in open(f, encoding="utf-8").read().splitlines()]
    sec = None; recs = []; cur = None
    for l in lines:
        if l in SECTIONS:
            if cur: recs.append(cur); cur = None
            sec = l; continue
        if sec in ("PLANT REPORT", "HARVEST REPORT") and DATE.match(l):
            if cur: recs.append(cur)
            cur = {"sec": sec, "date": l, "body": []}; continue
        if cur is not None:
            if l.startswith("Field Acres") or l.startswith("1-866") or l.startswith("FARM - "):
                recs.append(cur); cur = None; continue
            cur["body"].append(l)
    if cur: recs.append(cur)
    for r in recs:
        body = " | ".join(r["body"])
        c = corn_class(body)
        if c is None: continue
        split = "Whole Field" if "Whole Field" in body else "other"
        machine = ("Chopper" if "Chopper" in body else "Combine" if "Combine" in body else
                   "Planter/Drill" if r["sec"] == "PLANT REPORT" else "other")
        rows.append(dict(idn=idn, sec="plant" if r["sec"] == "PLANT REPORT" else "harvest",
                         date=pd.to_datetime(r["date"], format="%m/%d/%y"), cls=c, split=split,
                         machine=machine, body=body[:160]))
ev = pd.DataFrame(rows); ev["year"] = ev["date"].dt.year
ev.to_csv(sys.argv[2], index=False)
print(len(ev), "corn events;", ev.idn.nunique(), "fields")
print(pd.crosstab([ev.sec, ev.cls], ev.machine).to_string())
print(pd.crosstab(ev.year, [ev.sec, ev.cls]).to_string())
