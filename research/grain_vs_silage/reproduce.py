"""Rebuild every input and result in this folder with one command.

  python research/grain_vs_silage/reproduce.py              # whole pipeline, skips steps whose outputs exist
  python research/grain_vs_silage/reproduce.py --analysis-only   # only re-run analysis + figures

Steps (all outputs except results/ go to the git-ignored data/ folder):
  1. Download the Kraatz farm records and outlines from Zenodo 8087301 (CC-BY-4.0), md5-checked.
  2. Build the 2017-2020 training field-years (kraatz_fields.py).
  3. Pull Sentinel-2 + Landsat + Daymet for the 83 evaluation fields, 2021-2023 (preprocessing/loader.py).
  4. Pull the same for the 2017-2020 training fields (loader --fields).
  5. Run grain_silage.py (features + validation -> results/report.md) and make_figures.py.

Steps 3-4 read imagery from Microsoft Planetary Computer (no account needed). Each takes roughly 10-30 minutes
depending on network; lower --workers if many scene-days fail (Planetary Computer throttles bursts).
"""
import argparse, hashlib, json, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
ZENODO = "https://zenodo.org/api/records/8087301/files/{name}/content"
KRAATZ_FILES = {  # name: md5 from the Zenodo record
    "farm_database_annual_planting_harvesting_after_some_screening..zip": "65590c4f15b2179599bf676172ab36ba",
    "barc_field_outlines_for_study.zip": "35fae5dc89622d483b11e4a370f26d9c",
}
EVAL_GPKG = ROOT / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg"


def run(*args):
    print("\n>>", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force-pull", action="store_true", help="re-pull imagery even if the parquet files exist")
    a = ap.parse_args()

    if not a.analysis_only:
        kdir = DATA / "kraatz_zenodo_8087301"
        kdir.mkdir(parents=True, exist_ok=True)
        for name, want in KRAATZ_FILES.items():
            p = kdir / name
            if not p.exists() or md5(p) != want:
                print("downloading", name)
                urllib.request.urlretrieve(ZENODO.format(name=name), p)
            if md5(p) != want:
                sys.exit(f"md5 mismatch for {p}")
        run(HERE / "kraatz_fields.py", "--kraatz-dir", kdir, "--gpkg", EVAL_GPKG, "--out", DATA)

        pulls = [("barc", ["--gpkg", EVAL_GPKG, "--years", 2021, 2022, 2023]),
                 ("kraatz_2017_2020", ["--fields", DATA / "fields_kraatz_2017_2020.gpkg",
                                       "--years", 2017, 2018, 2019, 2020])]
        for tag, src in pulls:
            common = ["preprocessing/loader.py", *src, "--tag", tag, "--cell-m", 50000, "--workers", a.workers]
            if a.force_pull or not (DATA / f"series_{tag}.parquet").exists():
                run(*common, "--weather")
                prov = json.loads((DATA / f"series_{tag}.provenance.json").read_text())
                if prov["failures"]:
                    print(f"{len(prov['failures'])} scene-days failed; retrying them once")
                    run(*common, "--retry-failed")
            prov = json.loads((DATA / f"series_{tag}.provenance.json").read_text())
            if prov.get("failures"):
                print(f"WARNING: series_{tag} still has {len(prov['failures'])} failed scene-days; "
                      f"re-run: python preprocessing/loader.py {' '.join(map(str, common[1:]))} --retry-failed")

    run(HERE / "grain_silage.py")
    run(HERE / "make_figures.py")
    run(HERE / "aef_embeddings.py")  # first run downloads a 78 MB index and reads 14 tile windows (a few minutes)
    run(HERE / "model_comparison.py")
    print("\nDone. Read research/grain_vs_silage/results/report.md")


if __name__ == "__main__":
    main()
