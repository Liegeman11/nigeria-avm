"""
Nigerian AVM Version 2 — Mid-Market Lagos Scraper
Targets underrepresented areas: Surulere, Yaba, Gbagada, Ikorodu, Mushin
This fixes the Lekki vs Surulere location sensitivity weakness
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Target areas — CONFIRMED correct PropertyPro URL pattern via diagnostic:
# https://www.propertypro.ng/property-for-sale/in/lagos/<area>
TARGET_AREAS = {
    "surulere":  "https://www.propertypro.ng/property-for-sale/in/lagos/surulere",
    "yaba":      "https://www.propertypro.ng/property-for-sale/in/lagos/yaba",
    "gbagada":   "https://www.propertypro.ng/property-for-sale/in/lagos/gbagada",
    "ikorodu":   "https://www.propertypro.ng/property-for-sale/in/lagos/ikorodu",
    "mushin":    "https://www.propertypro.ng/property-for-sale/in/lagos/mushin",
    "agege":     "https://www.propertypro.ng/property-for-sale/in/lagos/agege",
    "shomolu":   "https://www.propertypro.ng/property-for-sale/in/lagos/shomolu",
    "maryland":  "https://www.propertypro.ng/property-for-sale/in/lagos/maryland",
}


def parse_price(price_str: str):
    cleaned = re.sub(r"[₦,\s]", "", str(price_str))
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_beds_baths(text: str):
    beds  = re.search(r"(\d+)\s*bed",  text, re.I)
    baths = re.search(r"(\d+)\s*bath", text, re.I)
    return (
        int(beds.group(1))  if beds  else None,
        int(baths.group(1)) if baths else None,
    )


def parse_date_added(text: str):
    m = re.search(r"Added\s+(.+)", text)
    return m.group(1).strip() if m else text.strip()


def parse_location_parts(location: str):
    parts = [p.strip() for p in location.split() if p.strip()]
    if len(parts) >= 3:
        state = parts[-1]
        lga   = parts[-2]
        area  = " ".join(parts[:-2])
    elif len(parts) == 2:
        state = lga = parts[-1]
        area  = parts[0]
    else:
        state = lga = area = location
    return area, lga, state


def extract_card(card) -> dict | None:
    content = card.select_one(".property-listing-content")
    if not content:
        return None

    title_el = content.select_one(".pl-title h3 a")
    title    = title_el.get_text(strip=True) if title_el else "N/A"
    href     = title_el["href"] if title_el and title_el.has_attr("href") else ""
    url      = f"https://www.propertypro.ng{href}" if href.startswith("/") else href

    loc_el   = content.select_one(".pl-title p")
    location = loc_el.get_text(strip=True) if loc_el else "N/A"
    area, lga, state = parse_location_parts(location)

    type_links = content.select(".pl-title h6 a")
    prop_type  = type_links[-1].get_text(strip=True) if type_links else "N/A"
    prop_type  = re.sub(r"\s+for\s+sale", "", prop_type, flags=re.I).strip()

    price_el  = content.select_one(".pl-price h3")
    price_raw = price_el.get_text(strip=True) if price_el else ""
    price_ngn = parse_price(price_raw)

    bedbath_el = content.select_one(".pl-price h6")
    bedbath    = bedbath_el.get_text(strip=True) if bedbath_el else ""
    bedrooms, bathrooms = parse_beds_baths(bedbath)

    if bedrooms is None:
        m = re.search(r"(\d+)\s*bed", title, re.I)
        bedrooms = int(m.group(1)) if m else None

    pid_el = content.select_one(".pl-price p")
    pid    = pid_el.get_text(strip=True).replace("PID :", "").strip() if pid_el else "N/A"

    agent_el = content.select_one(".pl-footer-left .flex-grow-1")
    agent    = agent_el.get_text(strip=True) if agent_el else "N/A"

    date_el    = content.select_one(".date-added")
    date_added = parse_date_added(date_el.get_text(strip=True)) if date_el else "N/A"

    badge_items = content.select(".pl-badge-left li")
    features    = [b.get_text(strip=True) for b in badge_items if b.get_text(strip=True)]

    return {
        "pid":        pid,
        "title":      title,
        "price_ngn":  price_ngn,
        "price_raw":  price_raw,
        "bedrooms":   bedrooms,
        "bathrooms":  bathrooms,
        "prop_type":  prop_type,
        "location":   location,
        "area":       area,
        "lga":        lga,
        "state":      state,
        "agent":      agent,
        "features":   "; ".join(features),
        "date_added": date_added,
        "url":        url,
        "source":     "v2_midmarket",
    }


def scrape_area(area_name: str, base_url: str, max_pages: int = 20):
    print(f"\nScraping {area_name.title()} (up to {max_pages} pages)...")
    all_listings = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  [!] Page {page} failed: {e}")
            break

        soup  = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("div.property-listing")

        if not cards:
            print(f"  [!] No cards on page {page} — stopping {area_name}")
            break

        page_listings = []
        for card in cards:
            row = extract_card(card)
            if row and row["price_ngn"]:
                page_listings.append(row)

        # VALIDATION — on page 1, check that results actually match the target area.
        # Catches silent URL-pattern failures immediately instead of after a full run.
        if page == 1 and page_listings:
            match_count = sum(
                1 for r in page_listings
                if area_name.lower() in r["location"].lower()
            )
            match_pct = match_count / len(page_listings) * 100
            print(f"  [validation] {match_count}/{len(page_listings)} "
                  f"({match_pct:.0f}%) listings actually mention '{area_name}'")
            if match_pct < 30:
                print(f"  [!] WARNING: URL pattern may be wrong for {area_name} — "
                      f"results don't match target area. Stopping early.")
                return all_listings + page_listings  # return what we have, don't waste time

        all_listings.extend(page_listings)
        print(f"  Page {page:2d}: {len(page_listings):2d} listings "
              f"(total: {len(all_listings)})")

        time.sleep(1.5 + (page % 3) * 0.3)

    return all_listings


def run_midmarket_scrape():
    all_results = []

    for area_name, base_url in TARGET_AREAS.items():
        area_listings = scrape_area(area_name, base_url, max_pages=20)
        all_results.extend(area_listings)
        print(f"  {area_name.title()} complete: {len(area_listings)} listings")
        time.sleep(3)  # pause between areas

    df_new = pd.DataFrame(all_results)
    df_new.drop_duplicates(subset=["pid"], inplace=True)

    # Load existing v1 data and combine
    df_v1 = pd.read_csv("nigeria_property_raw.csv")
    df_v1["source"] = "v1"

    df_combined = pd.concat([df_v1, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=["pid"], inplace=True)
    df_combined.to_csv("nigeria_property_raw_v2_midmarket.csv", index=False)

    print(f"\n{'='*55}")
    print(f"  MID-MARKET SCRAPE COMPLETE")
    print(f"{'='*55}")
    print(f"  New mid-market listings:  {len(df_new)}")
    print(f"  Original v1 listings:     {len(df_v1)}")
    print(f"  Combined total:           {len(df_combined)}")
    print(f"\n  LGA distribution (new listings):")
    print(df_new["lga"].value_counts().head(15))
    print(f"\n  Saved to nigeria_property_raw_v2_midmarket.csv")


if __name__ == "__main__":
    run_midmarket_scrape()

