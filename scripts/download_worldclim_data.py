# scripts/download_worldclim_data.py
from pathlib import Path
import subprocess
import zipfile

DATA_DIR = Path("data/climate")
ARCHIVES_DIR = DATA_DIR / "_archives"
ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    {
        "name": "current_bio_10m",
        "url": "http://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip",
        "dest": ARCHIVES_DIR / "wc2.1_10m_bio.zip",
        "extract_dir": DATA_DIR / "current" / "wc2.1_10m_bio",
        "is_zip": True,
    },
    {
        "name": "future_bcc_csm2_mr_10m_ssp245_2041_2060",
        "url": "http://geodata.ucdavis.edu/cmip6/10m/BCC-CSM2-MR/ssp245/wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif",
        "dest": DATA_DIR / "future" / "BCC_CSM2_MR_ssp245_2041_2060" / "wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif",
        "extract_dir": None,
        "is_zip": False,
    },
    {
        "name": "future_cnrm_cm6_1_10m_ssp245_2041_2060",
        "url": "http://geodata.ucdavis.edu/cmip6/10m/CNRM-CM6-1/ssp245/wc2.1_10m_bioc_CNRM-CM6-1_ssp245_2041-2060.tif",
        "dest": DATA_DIR / "future" / "CNRM_CM6_1_ssp245_2041_2060" / "wc2.1_10m_bioc_CNRM-CM6-1_ssp245_2041-2060.tif",
        "extract_dir": None,
        "is_zip": False,
    },
    {
        "name": "future_ipsl_cm6a_lr_10m_ssp245_2041_2060",
        "url": "http://geodata.ucdavis.edu/cmip6/10m/IPSL-CM6A-LR/ssp245/wc2.1_10m_bioc_IPSL-CM6A-LR_ssp245_2041-2060.tif",
        "dest": DATA_DIR / "future" / "IPSL_CM6A_LR_ssp245_2041_2060" / "wc2.1_10m_bioc_IPSL-CM6A-LR_ssp245_2041-2060.tif",
        "extract_dir": None,
        "is_zip": False,
    },
]


def curl_download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] already exists: {dest}")
        return

    cmd = [
        "curl",
        "-L",
        "-C", "-",              # resume
        "--retry", "10",
        "--retry-delay", "5",
        "--connect-timeout", "30",
        "-o", str(dest),
        url,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def unzip_file(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    for item in DOWNLOADS:
        print(f"\n=== {item['name']} ===")
        curl_download(item["url"], item["dest"])

        if item["is_zip"]:
            marker = item["extract_dir"] / ".extracted_ok"
            if marker.exists():
                print(f"[skip] already extracted: {item['extract_dir']}")
                continue

            print(f"Extracting to: {item['extract_dir']}")
            unzip_file(item["dest"], item["extract_dir"])
            marker.touch()

    print("\nDone.")


if __name__ == "__main__":
    main()