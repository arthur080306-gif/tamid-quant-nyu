"""
Rangers Odds Scraper v4
- Uses JavaScript clicks to navigate pagination (not URL params)
- Tracks current date across all game rows in a group
- Stops when no new games are found on a page
"""

import re
import time
import csv
import os
from datetime import datetime
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

OUTPUT_DIR = "betting_backtest/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rangers_real_odds.csv")

SEASONS = [
    ("2021-22", "https://www.oddsportal.com/hockey/usa/nhl-2021-2022/results/"),
    ("2022-23", "https://www.oddsportal.com/hockey/usa/nhl-2022-2023/results/"),
    ("2023-24", "https://www.oddsportal.com/hockey/usa/nhl-2023-2024/results/"),
]


def get_max_pages(html):
    matches = re.findall(r'data-number="(\d+)"', html)
    return max(int(m) for m in matches) if matches else 1


def parse_rangers_games(html, season):
    soup = BeautifulSoup(html, "html.parser")
    games = []
    seen_ids = set()
    current_date = ""

    event_rows = soup.find_all("div", class_=re.compile(r"eventRow"))

    for row in event_rows:
        # Pick up date from header if present
        date_el = row.find(attrs={"data-testid": "date-header"})
        if date_el:
            raw = date_el.get_text(strip=True)
            m = re.match(r"(\d{2} \w+ \d{4})", raw)
            if m:
                try:
                    current_date = datetime.strptime(m.group(1), "%d %b %Y").strftime("%Y-%m-%d")
                except:
                    current_date = m.group(1)

        rangers_link = row.find("a", href=re.compile(r"new-york-rangers"))
        if not rangers_link:
            continue

        href = rangers_link.get("href", "")
        gid_m = re.search(r"-([a-zA-Z0-9]+)/$", href)
        if not gid_m:
            continue
        gid = gid_m.group(1)
        if gid in seen_ids:
            continue
        seen_ids.add(gid)

        slug = href.strip("/").split("/")[-1]
        teams = re.sub(r"-[a-zA-Z0-9]+$", "", slug)

        if teams.startswith("new-york-rangers-"):
            home_team = "New York Rangers"
            away_team = teams[len("new-york-rangers-"):].replace("-", " ").title()
            rangers_home = True
        else:
            away_team = "New York Rangers"
            home_team = re.sub(r"-new-york-rangers$", "", teams).replace("-", " ").title()
            rangers_home = False

        score_els = row.find_all("div", class_=re.compile(r"min-mt:!flex.*font-bold"))
        scores = [e.get_text(strip=True) for e in score_els if e.get_text(strip=True).isdigit()]
        home_score = scores[0] if scores else ""
        away_score = scores[1] if len(scores) > 1 else ""

        odd_els = row.find_all(attrs={"data-testid": re.compile(r"odd-container")})
        odds = []
        for el in odd_els:
            p = el.find("p")
            if p:
                txt = p.get_text(strip=True)
                if re.match(r"^[+-]?\d+$", txt):
                    odds.append(txt)

        home_ml = odds[0] if odds else ""
        draw_ml = odds[1] if len(odds) > 1 else ""
        away_ml = odds[2] if len(odds) > 2 else ""
        rangers_ml = home_ml if rangers_home else away_ml
        opponent_ml = away_ml if rangers_home else home_ml

        games.append({
            "season": season,
            "date": current_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "rangers_home": rangers_home,
            "home_ml": home_ml,
            "draw_ml": draw_ml,
            "away_ml": away_ml,
            "rangers_ml": rangers_ml,
            "opponent_ml": opponent_ml,
            "game_url": href,
        })

    return games, seen_ids


def scrape_season(driver, season_name, base_url):
    all_games = []
    all_ids = set()

    print(f"  Loading page 1 ...", end=" ", flush=True)
    driver.get(base_url)
    time.sleep(9)

    html = driver.page_source
    max_pages = get_max_pages(html)
    games, ids = parse_rangers_games(html, season_name)
    new_games = [g for g in games if re.search(r"-([a-zA-Z0-9]+)/$", g["game_url"]).group(1) not in all_ids]
    all_ids.update(ids)
    all_games.extend(new_games)
    print(f"{len(new_games)} Rangers games (max pages: {max_pages})")

    for page_num in range(2, max_pages + 1):
        print(f"  Loading page {page_num} ...", end=" ", flush=True)

        clicked = driver.execute_script(f"""
            let links = document.querySelectorAll('a.pagination-link');
            for (let l of links) {{
                if (l.getAttribute('data-number') === '{page_num}') {{
                    l.click();
                    return true;
                }}
            }}
            return false;
        """)

        if not clicked:
            print("link not found — stopping.")
            break

        time.sleep(9)
        html = driver.page_source
        games, ids = parse_rangers_games(html, season_name)
        new_games = [g for g in games if re.search(r"-([a-zA-Z0-9]+)/$", g["game_url"]).group(1) not in all_ids]
        all_ids.update(ids)
        all_games.extend(new_games)
        print(f"{len(new_games)} new Rangers games (total so far: {len(all_games)})")

        if len(new_games) == 0:
            print(f"  No new games on page {page_num} — stopping.")
            break

        time.sleep(3)

    return all_games


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🏒 Rangers Odds Scraper v4 — Starting")
    print("   Chrome will open — leave it alone\n")

    opts = uc.ChromeOptions()
    driver = uc.Chrome(options=opts)

    all_games = []

    try:
        for season_name, url in SEASONS:
            print(f"\n── Season {season_name} {'─'*40}")
            games = scrape_season(driver, season_name, url)
            all_games.extend(games)
            print(f"  ✅ Season total: {len(games)} unique Rangers games")
            time.sleep(5)
    finally:
        driver.quit()

    if not all_games:
        print("\n❌ No games found.")
        return

    fieldnames = ["season","date","home_team","away_team","home_score","away_score",
                  "rangers_home","home_ml","draw_ml","away_ml","rangers_ml","opponent_ml","game_url"]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_games)

    print(f"\n✅ Saved {len(all_games)} games → {OUTPUT_FILE}")
    print("\n" + "="*50)
    print("SCRAPE SUMMARY")
    print("="*50)
    by_season = {}
    for g in all_games:
        by_season[g["season"]] = by_season.get(g["season"], 0) + 1
    for s, n in sorted(by_season.items()):
        print(f"  {s}: {n} games")
    print(f"  TOTAL: {len(all_games)} games")
    print("="*50)


if __name__ == "__main__":
    main()
