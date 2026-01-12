#!/usr/bin/env python3
"""
Fetch joker image URLs from Balatro wiki and save to image_assets.json
"""

import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path


def get_wiki_page(joker_name: str) -> str:
    """Fetch wiki page content for a joker."""
    # Convert joker name to wiki URL format
    wiki_name = joker_name.replace(" ", "_")

    # Some jokers have special page names
    if joker_name == "Joker":
        wiki_name = "Joker_(Joker)"

    url = f"https://balatrogame.fandom.com/wiki/{urllib.parse.quote(wiki_name)}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"  Error fetching {joker_name}: {e}")
        return ""


def extract_image_url(html: str, joker_name: str) -> str:
    """Extract the joker image URL from wiki page HTML."""
    # Look for the static.wikia.nocookie.net URL pattern
    # Pattern: https://static.wikia.nocookie.net/balatrogame/images/X/XX/Name.png

    wiki_name = joker_name.replace(" ", "_")

    # Try multiple patterns
    patterns = [
        rf'(https://static\.wikia\.nocookie\.net/balatrogame/images/[a-f0-9]/[a-f0-9]{{2}}/{re.escape(wiki_name)}\.png)',
        rf'(https://static\.wikia\.nocookie\.net/balatrogame/images/[a-f0-9]/[a-f0-9]{{2}}/{re.escape(wiki_name)}\.png/revision/latest[^"\']*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            url = match.group(1)
            # Clean up URL - remove revision params for cleaner URLs
            if '/revision/latest' in url:
                url = url.split('/revision/latest')[0]
            return url

    return ""


def main():
    # Load existing assets
    assets_path = Path(__file__).parent.parent / "balatro_sim" / "data" / "image_assets.json"
    with open(assets_path) as f:
        assets = json.load(f)

    # Load joker names
    jokers_path = Path(__file__).parent.parent / "jokers_parsed.json"
    with open(jokers_path) as f:
        jokers = json.load(f)

    print(f"Fetching image URLs for {len(jokers)} jokers...")

    success = 0
    failed = []

    for i, joker in enumerate(jokers):
        name = joker["name"]

        # Skip if we already have this joker
        if name in assets["jokers"]:
            print(f"[{i+1}/{len(jokers)}] {name}: cached")
            success += 1
            continue

        print(f"[{i+1}/{len(jokers)}] {name}...", end=" ")

        html = get_wiki_page(name)
        if html:
            url = extract_image_url(html, name)
            if url:
                assets["jokers"][name] = url
                print(f"OK")
                success += 1
            else:
                print(f"no image found")
                failed.append(name)
        else:
            failed.append(name)

        # Rate limit to be nice to the wiki
        time.sleep(0.5)

        # Save periodically
        if (i + 1) % 10 == 0:
            with open(assets_path, 'w') as f:
                json.dump(assets, f, indent=2)

    # Final save
    with open(assets_path, 'w') as f:
        json.dump(assets, f, indent=2)

    print(f"\nDone! {success}/{len(jokers)} jokers mapped.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
