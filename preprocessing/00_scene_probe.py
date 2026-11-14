# Scene-metadata probe over the BARC bounding box: no pixels, no labels.
import json, urllib.request, collections, datetime as dt
URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
BBOX = [-76.94, 39.00, -76.81, 39.05]
def search(coll, start, end):
    body = {"collections": [coll], "bbox": BBOX, "datetime": f"{start}/{end}", "limit": 250}
    out, url, data = [], URL, json.dumps(body).encode()
    while True:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        j = json.load(urllib.request.urlopen(req, timeout=60))
        out += j["features"]
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        if not nxt: return out
        url = nxt[0]["href"]; data = json.dumps(nxt[0].get("body", body)).encode() if nxt[0].get("method") == "POST" else None
for year in (2021, 2022, 2023):
    s2 = search("sentinel-2-l2a", f"{year}-08-01", f"{year}-11-30")
    ls = search("landsat-c2-l2", f"{year}-08-01", f"{year}-11-30")
    best = collections.defaultdict(lambda: 101.0); base = collections.Counter(); tiles = collections.Counter()
    for f in s2:
        p = f["properties"]; d = p["datetime"][:10]
        best[d] = min(best[d], p.get("eo:cloud_cover", 101)); base[p.get("s2:processing_baseline")] += 1; tiles[p.get("s2:mgrs_tile")] += 1
    for f in ls:
        p = f["properties"]; d = p["datetime"][:10]
        if p.get("platform") in ("landsat-8", "landsat-9"):
            best[d + "L"] = min(best[d + "L"], p.get("eo:cloud_cover", 101))
    clear = sorted(k for k, v in best.items() if v < 20)
    days = sorted({dt.date.fromisoformat(k[:10]) for k in clear})
    gaps = [(b - a).days for a, b in zip(days, days[1:])]
    print(f"{year}: S2 items {len(s2)}, Landsat8/9 items {sum(1 for f in ls if f['properties'].get('platform') in ('landsat-8','landsat-9'))}; "
          f"dates with <20% scene cloud (S2 or L8/9): {len(days)}; longest gap {max(gaps) if gaps else None} d")
    print("   clear dates:", ", ".join(d.strftime('%m-%d') for d in days))
    print("   S2 processing baselines:", dict(base), " tiles:", dict(tiles))
