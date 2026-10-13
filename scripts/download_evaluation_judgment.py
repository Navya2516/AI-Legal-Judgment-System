import requests
from pathlib import Path

ROOT_URL = "https://scr.sci.gov.in"
OUTPUT_DIR = Path("data/raw/evaluation_judgments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

path = "2024_2_722_742"
citation_year = "2024"
nc_display = "2024INSC131"

session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154.0 Safari/537.36"
}

print("Opening Supreme Court website...")
r = session.get(
    f"{ROOT_URL}/scrsearch/",
    headers=headers,
    verify=False,
    timeout=30
)

print("HTTP status:", r.status_code)

if r.status_code != 200:
    raise RuntimeError("Could not connect to Supreme Court website")

print("Session cookies:", session.cookies.get_dict())

payload = {
    "val": "51",
    "lang_flg": "",
    "path": path,
    "citation_year": citation_year,
    "fcourt_type": "3",
    "nc_display": nc_display,
    "ajax_req": "true",
}

print("\nRequesting PDF link...")

r = session.post(
    f"{ROOT_URL}/scrsearch/?p=pdf_search/openpdfcaptcha",
    data=payload,
    headers={
        **headers,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{ROOT_URL}/scrsearch/",
    },
    verify=False,
    timeout=60
)

print("PDF-link HTTP status:", r.status_code)
print("Response:")
print(r.text[:3000])
