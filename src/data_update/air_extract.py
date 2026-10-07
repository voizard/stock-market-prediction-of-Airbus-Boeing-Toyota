"""Robust Airbus (EPA:AIR) data extraction.

Yahoo is IP-blocked; stockanalysis.com international history API is undocumented,
so we use two independent sources:
  1) the ~1y daily chart blob embedded in the EPA:AIR overview page
  2) the `changes` object (price1y / price5y) on the same page
"""
import json
import re
import urllib.request

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


html = get("https://stockanalysis.com/quote/epa/AIR/")

# --- chart blob: chart:{expiration:...,data:[{c:205.3,o:203.85,t:1759881600},...]} ---
m = re.search(r"chart:\{expiration:\d+,data:\[(.*?)\]\}", html, re.S)
points = []
if m:
    for mm in re.finditer(r"\{c:([0-9.]+)[^}]*?t:(\d+)\}", m.group(1)):
        points.append((int(mm.group(2)), float(mm.group(1))))
print("chart points:", len(points))
if points:
    print("first:", points[0], "last:", points[-1])

# --- changes object ---
mc = re.search(r"changes:\{([^}]*)\}", html)
changes = {}
if mc:
    for mm in re.finditer(r"(\w+):([0-9.]+)", mc.group(1)):
        changes[mm.group(1)] = float(mm.group(2))
print("changes:", json.dumps(changes, indent=2))

# --- current price / name ---
mp = re.search(r'"(?:price|lastPrice)":([0-9.]+)', html)
mt = re.search(r"<title>([^<]*)", html)
print("title:", mt.group(1) if mt else None)

json.dump({"points": points, "changes": changes}, open("/tmp/air_extract.json", "w"))
print("saved /tmp/air_extract.json")
