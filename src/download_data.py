"""
download_data.py — Download historical daily streamflow data from USGS Water Data OGC API.

Station : USGS-01646500  (Potomac River at Little Falls, near Washington, D.C.)
Parameter: 00060          (Discharge / streamflow)
Statistic: 00003          (Daily mean)
Period   : 2021-01-01 to 2025-12-31  (~5 years)

API docs : https://api.waterdata.usgs.gov/docs/ogcapi/
Endpoint : https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items

Usage:
    python src/download_data.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
STATION_ID = "USGS-01646500"           # Potomac River at Little Falls
PARAMETER_CODE = "00060"                # Discharge (streamflow)
STATISTIC_ID = "00003"                  # Daily mean
DATE_START = "2021-01-01"
DATE_END = "2025-12-31"

BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items"
PAGE_LIMIT = 1000                       # max items per page (API default max)

# Output paths (relative to project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CSV_PATH = os.path.join(RAW_DIR, "streamflow.csv")
META_PATH = os.path.join(RAW_DIR, "metadata.json")


def build_first_url():
    """Build the initial API request URL with all query filters."""
    params = {
        "f": "json",
        "monitoring_location_id": STATION_ID,
        "parameter_code": PARAMETER_CODE,
        "statistic_id": STATISTIC_ID,
        "datetime": f"{DATE_START}/{DATE_END}",
        "limit": PAGE_LIMIT,
    }
    # Build URL string for logging
    req = requests.Request("GET", BASE_URL, params=params)
    prepared = req.prepare()
    return prepared.url


def fetch_all_pages():
    """
    Download every page of results from the USGS OGC API.
    Returns a list of feature dicts (one per daily observation).
    """
    params = {
        "f": "json",
        "monitoring_location_id": STATION_ID,
        "parameter_code": PARAMETER_CODE,
        "statistic_id": STATISTIC_ID,
        "datetime": f"{DATE_START}/{DATE_END}",
        "limit": PAGE_LIMIT,
    }

    all_features = []
    page = 1
    url = BASE_URL
    current_params = params

    while True:
        print(f"  Fetching page {page} …", end=" ", flush=True)
        response = requests.get(url, params=current_params, timeout=60)
        response.raise_for_status()
        data = response.json()

        features = data.get("features", [])
        matched = data.get("numberMatched", "?")
        returned = data.get("numberReturned", len(features))
        print(f"got {returned} records (total matched: {matched})")

        all_features.extend(features)

        if returned == 0:
            break

        # Find the "next" link for pagination
        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link["href"]
                break

        if next_url is None:
            break  # no more pages

        # The "next" URL is a full URL with params already embedded
        url = next_url
        current_params = None  # params are in the URL already
        page += 1

    return all_features


def features_to_dataframe(features):
    """
    Convert the list of GeoJSON features into a clean pandas DataFrame.
    Each feature has: properties (the data) and geometry (lat/lon).
    """
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])

        rows.append({
            "date": props.get("time"),
            "discharge_cfs": props.get("value"),
            "unit_of_measure": props.get("unit_of_measure"),
            "parameter_code": props.get("parameter_code"),
            "statistic_id": props.get("statistic_id"),
            "monitoring_location_id": props.get("monitoring_location_id"),
            "approval_status": str(props.get("approvals_status", [])),
            "qualifier": str(props.get("qualifier")),
            "longitude": coords[0] if coords else None,
            "latitude": coords[1] if len(coords) > 1 else None,
        })

    df = pd.DataFrame(rows)

    # Sort by date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Convert discharge to numeric (keep NaN for missing)
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")

    return df


def print_summary(df):
    """Print a human-readable summary of the downloaded data."""
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)

    print(f"  Station ID         : {STATION_ID}")
    print(f"  Parameter code     : {PARAMETER_CODE} (Discharge / streamflow)")
    print(f"  Statistic          : {STATISTIC_ID} (Daily mean)")

    if not df.empty:
        unit = df["unit_of_measure"].dropna().unique()
        print(f"  Unit of measure    : {', '.join(unit)}")
        print(f"  Date range         : {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Row count          : {len(df)}")
        print(f"  Column names       : {list(df.columns)}")
        print(f"  Missing values     :")
        for col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                print(f"    {col:30s}: {n_miss}")
        if df[df.columns].isna().sum().sum() == 0:
            print(f"    (none)")
        dup_dates = df["date"].duplicated().sum()
        print(f"  Duplicate dates    : {dup_dates}")
    else:
        print("  *** NO DATA RETURNED ***")

    print("=" * 60)


def save_metadata(source_url):
    """Save a small JSON file with the source URL and retrieval date."""
    meta = {
        "station_id": STATION_ID,
        "station_name": "Potomac River at Little Falls, near Washington, D.C.",
        "parameter_code": PARAMETER_CODE,
        "parameter_description": "Discharge (streamflow), daily mean",
        "statistic_id": STATISTIC_ID,
        "unit_of_measure": "ft3/s (cubic feet per second)",
        "date_range_requested": f"{DATE_START} to {DATE_END}",
        "source_url": source_url,
        "api_documentation": "https://api.waterdata.usgs.gov/docs/ogcapi/",
        "retrieval_date": datetime.now(timezone.utc).isoformat(),
        "output_file": "streamflow.csv",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to  : {META_PATH}")


def main():
    print("=" * 60)
    print("  HydroGuard AI — USGS Streamflow Data Downloader")
    print("=" * 60)
    print(f"\n  Station : {STATION_ID}")
    print(f"  Param   : {PARAMETER_CODE} (Discharge)")
    print(f"  Stat    : {STATISTIC_ID} (Daily mean)")
    print(f"  Period  : {DATE_START} to {DATE_END}")
    print()

    # Create output directory if needed
    os.makedirs(RAW_DIR, exist_ok=True)

    # Build and display the source URL
    source_url = build_first_url()
    print(f"  Source URL:\n  {source_url}\n")

    # Download all pages
    print("  Downloading from USGS OGC API …")
    try:
        features = fetch_all_pages()
    except requests.exceptions.RequestException as e:
        print(f"\n  ERROR: API request failed!\n  {e}")
        print("\n  Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try the URL in a browser:")
        print(f"     {source_url}")
        print("  3. Check API status: https://api.waterdata.usgs.gov/status/")
        sys.exit(1)

    if not features:
        print("\n  ERROR: No data returned from the API.")
        print("  The station may not have data for this parameter/date range.")
        sys.exit(1)

    print(f"\n  Total observations downloaded: {len(features)}")

    # Convert to DataFrame
    df = features_to_dataframe(features)

    # Save CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"  CSV saved to       : {CSV_PATH}")

    # Save metadata
    save_metadata(source_url)

    # Print summary
    print_summary(df)

    # Show first and last few rows as a sanity check
    print("\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print("\n  Last 5 rows:")
    print(df.tail().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
