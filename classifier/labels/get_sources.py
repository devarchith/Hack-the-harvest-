"""Download the two public BARC farm-record datasets and check them against the publishers' md5 checksums.

  python classifier/labels/get_sources.py /home/keeboh/work/labels

- Zenodo 10.5281/zenodo.8087301 (CC BY 4.0): BARC farm-database planting and harvest records, 2017-2021, and the
  field outlines of that study.
- Ag Data Commons 10.15482/USDA.ADC/28950746.v2, BARCFarmMgmt.zip (U.S. Public Domain): FarmLogic field reports (PDF)
  with planting and harvest records, 2014-2024.

Writes <dir>/zenodo/farmdb/*.csv, <dir>/zenodo/outlines/*.shp and <dir>/farm_reports/pdf/*.pdf. Files already present
with the right checksum are not downloaded again.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

ZENODO = "https://zenodo.org/records/8087301/files/"
FIGSHARE = "https://api.figshare.com/v2/articles/28950746"
FILES = [  # (published file name, md5 published by the repository, local zip, extract to)
    ("farm_database_annual_planting_harvesting_after_some_screening..zip", "65590c4f15b2179599bf676172ab36ba",
     "zenodo/farmdb.zip", "zenodo/farmdb"),
    ("barc_field_outlines_for_study.zip", "35fae5dc89622d483b11e4a370f26d9c", "zenodo/outlines.zip", "zenodo/outlines"),
    ("BARCFarmMgmt.zip", "1d1d01cf82d81afa2006ea5e342b28d3", "farm_reports/BARCFarmMgmt.zip", "farm_reports/pdf"),
]


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def url_for(name):
    if name != "BARCFarmMgmt.zip":
        return ZENODO + name + "?download=1"
    files = json.load(urlopen(FIGSHARE, timeout=60))["files"]
    return next(f["download_url"] for f in files if f["name"] == name)


def main():
    root = Path(sys.argv[1])
    for name, want, local, dest in FILES:
        zp = root / local
        zp.parent.mkdir(parents=True, exist_ok=True)
        if not zp.exists() or md5(zp) != want:
            with urlopen(url_for(name), timeout=300) as r:
                zp.write_bytes(r.read())
        got = md5(zp)
        if got != want:
            sys.exit(f"{name}: md5 {got}, expected {want}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(root / dest)
            n = len(z.namelist())
        print(f"{name}: md5 {got} matches the published checksum, {zp.stat().st_size:,} bytes, {n} files -> {dest}")


if __name__ == "__main__":
    main()
