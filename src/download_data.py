"""Fetch the UCI zip into data/ and unzip (including a zip-inside-zip)."""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def already_there():
    for p in DATA.rglob("Inertial Signals"):
        if p.parent.name == "train":
            return True
    return False


def unzip_nested(zpath: Path):
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(DATA)
    for inner in list(DATA.rglob("*.zip")):
        if inner == zpath:
            continue
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if already_there():
        print("dataset already unzipped under data/")
        return
    zpath = DATA / "uci_har.zip"
    if not zpath.exists() or zpath.stat().st_size < 1_000_000:
        print("downloading", URL)
        req = Request(URL, headers={"User-Agent": "har-assignment"})
        with urlopen(req) as r:
            zpath.write_bytes(r.read())
        print("saved", zpath, "bytes", zpath.stat().st_size)
    print("unzipping")
    unzip_nested(zpath)
    if not already_there():
        raise SystemExit("unzip finished but train/Inertial Signals was not found")
    print("ok")


if __name__ == "__main__":
    main()
