"""Query Haryana, Gujarat, Uttar Pradesh from Overpass API."""
import os, json, time, requests
from collections import Counter

URLS = ["https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter"]
HEADERS = {"User-Agent": "SIH26162-Research/1.0", "Accept": "application/json"}

STATES = {
    "haryana": (74.45, 27.65, 77.60, 30.95),
    "gujarat": (68.15, 20.10, 73.95, 24.70),
    "uttar_pradesh": (77.10, 23.85, 84.70, 30.45),
}

def classify(tags):
    if tags.get("natural") == "volcano": return "volcano"
    if tags.get("industrial") == "refinery": return "refinery"
    if tags.get("power") == "plant": return "power_plant"
    if tags.get("industrial") in ("factory","manufacturing","works"): return "factory"
    if tags.get("industrial") in ("oil","gas"): return "oil_gas"
    if tags.get("man_made") == "works": return "factory"
    if tags.get("resource") in ("oil","gas"): return "oil_gas"
    if tags.get("landuse") == "industrial": return "industrial_zone"
    if tags.get("power") in ("station","substation"): return "power_infrastructure"
    return "other_industrial"

def post(query, timeout=45):
    for url in URLS:
        for attempt in range(2):
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
            except Exception as e:
                pass
            time.sleep(2)
    raise RuntimeError(f"Failed: {url}")

def get_state(name, bbox):
    w, s, e, n = bbox
    q = f'[out:json][timeout:45];(nwr["power"="plant"]({s},{w},{n},{e});nwr["industrial"="refinery"]({s},{w},{n},{e});nwr["industrial"="factory"]({s},{w},{n},{e});nwr["man_made"="works"]({s},{w},{n},{e});nwr["resource"="oil"]({s},{w},{n},{e});nwr["resource"="gas"]({s},{w},{n},{e});nwr["natural"="volcano"]({s},{w},{n},{e});nwr["landuse"="industrial"]({s},{w},{n},{e}););out center;'
    return [s for s in [{
        "id": e["id"], "osm_type": e["type"], "lat": (e.get("lat") if e["type"]=="node" else e["center"].get("lat")),
        "lon": (e.get("lon") if e["type"]=="node" else e["center"].get("lon")),
        "tags": e.get("tags",{}), "site_type": classify(e.get("tags",{})),
        "name": e.get("tags",{}).get("name",""), "state": name
    } for e in post(q).get("elements",[]) if (e.get("lat") if e["type"]=="node" else e.get("center",{}).get("lat"))]]


if __name__ == "__main__":
    path = "data/raw/osm_industrial_sites.json"
    existing = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: existing = json.load(f)
    print(f"Loaded {len(existing)} existing sites")
    seen = set((s["osm_type"],s["id"]) for s in existing)
    all_sites = list(existing)

    for i, (sn, bb) in enumerate(STATES.items(),1):
        try:
            sites = get_state(sn, bb)
            added = 0
            for s in sites:
                k = (s["osm_type"],s["id"])
                if k not in seen:
                    seen.add(k)
                    all_sites.append(s)
                    added += 1
            print(f"[{i}/3] {sn}: +{len(sites)} ({added} new, total {len(all_sites)})")
            with open(path, "w", encoding="utf-8") as f: json.dump(all_sites, f, indent=2)
        except Exception as e:
            print(f"[{i}/3] {sn}: FAILED {e}")
        time.sleep(2)

    print(f"\nTotal: {len(all_sites)} sites")
    for t,n in Counter(s["site_type"] for s in all_sites).most_common():
        print(f"  {t}: {n}")
