"""
Data loading & enrichment utilities for the World Cup Standings dashboard.

The source file (team_standings.csv) only contains two columns: Standing and Team.
To make the requested analyses (correlation, regression, classification, clustering,
geolocation, ...) meaningful, this module enriches the raw data with:

  1. REAL, factual metadata: football confederation + capital city coordinates,
     used only for grouping and the map.
  2. SIMULATED performance statistics (points, goals, possession, etc.), generated
     with a fixed random seed and correlated with final standing. These are clearly
     labelled as simulated everywhere in the app -- they are NOT the real 2010 World
     Cup match statistics, since that data isn't in the source file.
"""

import numpy as np
import pandas as pd

# Real-world metadata: (Confederation, Continent, Capital latitude, Capital longitude)
COUNTRY_META = {
    "Spain":        ("UEFA",     "Europe",        40.4168, -3.7038),
    "Netherlands":  ("UEFA",     "Europe",        52.3676,  4.9041),
    "Germany":      ("UEFA",     "Europe",        52.5200, 13.4050),
    "Uruguay":      ("CONMEBOL", "South America", -34.9011, -56.1645),
    "Argentina":    ("CONMEBOL", "South America", -34.6037, -58.3816),
    "Brazil":       ("CONMEBOL", "South America", -15.8267, -47.9218),
    "Ghana":        ("CAF",      "Africa",          5.6037,  -0.1870),
    "Paraguay":     ("CONMEBOL", "South America", -25.2637, -57.5759),
    "Japan":        ("AFC",      "Asia",           35.6762, 139.6503),
    "Chile":        ("CONMEBOL", "South America", -33.4489, -70.6693),
    "Portugal":     ("UEFA",     "Europe",        38.7223,  -9.1393),
    "USA":          ("CONCACAF", "North America", 38.9072, -77.0369),
    "England":      ("UEFA",     "Europe",        51.5074,  -0.1278),
    "Mexico":       ("CONCACAF", "North America", 19.4326, -99.1332),
    "South Korea":  ("AFC",      "Asia",          37.5665, 126.9780),
    "Slovakia":     ("UEFA",     "Europe",        48.1486,  17.1077),
    "Ivory Coast":  ("CAF",      "Africa",          6.8276,  -5.2893),
    "Slovenia":     ("UEFA",     "Europe",        46.0569,  14.5058),
    "Switzerland":  ("UEFA",     "Europe",        46.9480,   7.4474),
    "South Africa": ("CAF",      "Africa",        -25.7461,  28.1881),
    "Australia":    ("AFC",      "Oceania",       -35.2809, 149.1300),
    "New Zealand":  ("OFC",      "Oceania",       -41.2865, 174.7762),
    "Serbia":       ("UEFA",     "Europe",        44.7866,  20.4489),
    "Denmark":      ("UEFA",     "Europe",        55.6761,  12.5683),
    "Greece":       ("UEFA",     "Europe",        37.9838,  23.7275),
    "Italy":        ("UEFA",     "Europe",        41.9028,  12.4964),
    "Nigeria":      ("CAF",      "Africa",          9.0765,   7.3986),
    "Algeria":      ("CAF",      "Africa",         36.7538,   3.0588),
    "France":       ("UEFA",     "Europe",        48.8566,   2.3522),
    "Honduras":     ("CONCACAF", "North America", 14.0723, -87.1921),
    "Cameroon":     ("CAF",      "Africa",          3.8480,  11.5021),
    "North Korea":  ("AFC",      "Asia",          39.0392, 125.7625),
}

RANDOM_SEED = 42


def load_raw(path="team_standings.csv"):
    return pd.read_csv(path)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach real metadata + simulated performance stats to the raw standings df."""
    df = df.copy()
    n = len(df)

    df["Confederation"] = df["Team"].map(lambda t: COUNTRY_META.get(t, ("Unknown",) * 4)[0])
    df["Continent"] = df["Team"].map(lambda t: COUNTRY_META.get(t, ("Unknown",) * 4)[1])
    df["Latitude"] = df["Team"].map(lambda t: COUNTRY_META.get(t, (None, None, np.nan, np.nan))[2])
    df["Longitude"] = df["Team"].map(lambda t: COUNTRY_META.get(t, (None, None, np.nan, np.nan))[3])

    rng = np.random.default_rng(RANDOM_SEED)

    # "base" performance score: 1.0 for the champion, ~0 for the last team
    base = (n - df["Standing"] + 1) / n

    goals_for = np.clip(np.round(base * 9 + rng.normal(0, 1.1, n)), 0, None)
    goals_against = np.clip(np.round((1 - base) * 7 + rng.normal(0, 1.1, n)), 0, None)
    points = np.clip(np.round(base * 9 + rng.normal(0, 0.8, n)), 0, 9)
    possession = np.clip(48 + base * 18 + rng.normal(0, 4, n), 30, 72)
    pass_accuracy = np.clip(66 + base * 22 + rng.normal(0, 3, n), 50, 96)
    shots_on_target = np.clip(np.round(3 + base * 6 + rng.normal(0, 1.2, n)), 0, None)

    df["Points_sim"] = points.astype(int)
    df["GoalsFor_sim"] = goals_for.astype(int)
    df["GoalsAgainst_sim"] = goals_against.astype(int)
    df["GoalDiff_sim"] = df["GoalsFor_sim"] - df["GoalsAgainst_sim"]
    df["Possession_sim"] = possession.round(1)
    df["PassAccuracy_sim"] = pass_accuracy.round(1)
    df["ShotsOnTarget_sim"] = shots_on_target.astype(int)

    df["Podium"] = (df["Standing"] <= 3).astype(int)
    df["Top10"] = (df["Standing"] <= 10).astype(int)

    return df


SIMULATED_COLUMNS = [
    "Points_sim", "GoalsFor_sim", "GoalsAgainst_sim", "GoalDiff_sim",
    "Possession_sim", "PassAccuracy_sim", "ShotsOnTarget_sim",
]

REAL_COLUMNS = ["Standing", "Team", "Confederation", "Continent", "Latitude", "Longitude"]
