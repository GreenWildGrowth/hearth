# scripts/download_geonames.py
from pathlib import Path
import subprocess
import zipfile

RAW_DIR = Path("data/raw/geonames")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    {
        "name": "allCountries.zip",
        "url": "http://download.geonames.org/export/dump/allCountries.zip",
        "dest": RAW_DIR / "allCountries.zip",
        "extract": True,
    },
    {
        "name": "countryInfo.txt",
        "url": "http://download.geonames.org/export/dump/countryInfo.txt",
        "dest": RAW_DIR / "countryInfo.txt",
        "extract": False,
    },
    {
        "name": "admin1CodesASCII.txt",
        "url": "http://download.geonames.org/export/dump/admin1CodesASCII.txt",
        "dest": RAW_DIR / "admin1CodesASCII.txt",
        "extract": False,
    },
]


def curl_download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest}")
        return

    cmd = [
        "curl",
        "-L",
        "-C", "-",
        "--retry", "10",
        "--retry-delay", "5",
        "--connect-timeout", "30",
        "-o", str(dest),
        url,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def unzip_file(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    for item in FILES:
        print(f"\n=== {item['name']} ===")
        curl_download(item["url"], item["dest"])
        if item["extract"]:
            marker = RAW_DIR / f".{item['name']}.extracted"
            if not marker.exists():
                unzip_file(item["dest"], RAW_DIR)
                marker.touch()

    print("\nDone.")


if __name__ == "__main__":
    main()