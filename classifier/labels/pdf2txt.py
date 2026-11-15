"""Extract the text of each farm-report PDF. args: <dir of .pdf> <out dir for .txt>"""
import sys, glob, os, pypdf
src, dst = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(src, "*.pdf"))):
    r = pypdf.PdfReader(f)
    t = "\n".join(p.extract_text() or "" for p in r.pages)
    open(os.path.join(dst, os.path.basename(f)[:-4] + ".txt"), "w", encoding="utf-8").write(t)
    print(os.path.basename(f), len(r.pages), "pages", len(t), "chars")
