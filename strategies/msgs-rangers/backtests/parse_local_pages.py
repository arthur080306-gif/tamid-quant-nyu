"""
Parse locally saved OddsPortal HTML pages for 2024-25 Rangers games
"""
import re, os
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

PAGES_DIR = "nhl_pages"
OUTPUT    = "betting_backtest/data/rangers_real_odds.csv"

existing = pd.read_csv(OUTPUT)
print(f"Existing games: {len(existing)}")

all_games = []
all_ids   = set()

page_files = sorted(
    [f for f in os.listdir(PAGES_DIR) if f.endswith(".html")],
    key=lambda x: int(re.search(r'\d+', x).group())
)
print(f"Found {len(page_files)} HTML files: {page_files}")

for fname in page_files:
    path = os.path.join(PAGES_DIR, fname)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    cur_date = ""
    page_games = 0

    for row in soup.find_all("div", class_=re.compile(r"eventRow")):
        # Pick up date header
        date_el = row.find(attrs={"data-testid": "date-header"})
        if date_el:
            m = re.match(r"(\d{2} \w+ \d{4})", date_el.get_text(strip=True))
            if m:
                try:
                    cur_date = datetime.strptime(m.group(1), "%d %b %Y").strftime("%Y-%m-%d")
                except:
                    cur_date = m.group(1)

        link = row.find("a", href=re.compile(r"new-york-rangers"))
        if not link:
            continue

        href = link.get("href", "")
        gid  = re.search(r"-([a-zA-Z0-9]+)/$", href)
        if not gid:
            continue
        gid = gid.group(1)
        if gid in all_ids:
            continue
        all_ids.add(gid)

        slug  = href.strip("/").split("/")[-1]
        teams = re.sub(r"-[a-zA-Z0-9]+$", "", slug)

        if teams.startswith("new-york-rangers-"):
            home = "New York Rangers"
            away = teams[17:].replace("-", " ").title()
            rh   = True
        else:
            away = "New York Rangers"
            home = re.sub(r"-new-york-rangers$", "", teams).replace("-", " ").title()
            rh   = False

        odd_els = row.find_all(attrs={"data-testid": re.compile(r"odd-container")})
        odds = []
        for el in odd_els:
            p = el.find("p")
            if p and re.match(r"^[+-]?\d+$", p.get_text(strip=True)):
                odds.append(p.get_text(strip=True))

        hml = odds[0] if odds else ""
        dml = odds[1] if len(odds) > 1 else ""
        aml = odds[2] if len(odds) > 2 else ""

        all_games.append({
            "season":      "2024-25",
            "date":        cur_date,
            "home_team":   home,
            "away_team":   away,
            "home_score":  "",
            "away_score":  "",
            "rangers_home": rh,
            "home_ml":     hml,
            "draw_ml":     dml,
            "away_ml":     aml,
            "rangers_ml":  hml if rh else aml,
            "opponent_ml": aml if rh else hml,
            "game_url":    href,
        })
        page_games += 1

    print(f"  {fname}: {page_games} Rangers games")

print(f"\n2024-25 total: {len(all_games)} Rangers games")

if all_games:
    new_df   = pd.DataFrame(all_games)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"Total games now: {len(combined)}")
    print(combined.groupby("season").size())
else:
    print("No Rangers games found — check that pages saved correctly")
    print("Try opening one of the HTML files in your browser to verify it loaded")
