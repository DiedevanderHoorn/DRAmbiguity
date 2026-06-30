from playwright.sync_api import sync_playwright
import time
import subprocess
import json
from collections import defaultdict

def save_for_drawing(
    X_embedded,
    y,
    X_embedded_new,
    new_y,
    split_to_parent,
    pred_labels = None
):


    if pred_labels is not None:
        correct = ( y == pred_labels)
    else:
        correct = [True for i in enumerate(range(len(y)))]

    symbols = [
        "cross" if i in split_to_parent else "circle"
        for i in range(len(X_embedded_new))
    ]

    dup_data = [
        {
            "x": float(x),
            "y": float(y_coord),
            "label": int(lbl),
            "pred_correct" : bool(correct[split_to_parent[i]]) if i in split_to_parent else bool(correct[i]),
            "symbol": symbols[i],
            "id": i,
        }
        for i, ((x, y_coord), lbl) in enumerate(zip(X_embedded_new, new_y))
    ]

    og_data = [
        {
            "x": float(x),
            "y": float(y_coord),
            "label": int(lbl),
            "pred_correct" : bool(correct[i]),
            "symbol": symbols[i],
            "id": i
        }
        for i, ((x, y_coord), lbl) in enumerate(zip(X_embedded, y))
    ]

    grouped = defaultdict(list)
    for child, parent in split_to_parent.items():
        grouped[parent].append(child)

    dup_edges = []
    for parent, nodes in grouped.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dup_edges.append({
                    "source": int(nodes[i]),
                    "target": int(nodes[j]),
                    "type": "duplicate"
                })

    data = {
        "original_embedding" : og_data,
        "disambiguated_embedding" : dup_data,
        "edges" : dup_edges,
    }

    file_path = f"data/data.json"
            
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
   

def render_images(fig):
    # start local server
    server = subprocess.Popen(
        ["python", "-m", "http.server", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(2)  # give server time to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page( 
                viewport={"width": 1200, "height": 600},
                device_scale_factor=3
            )

            page.goto("http://localhost:8000/render.html")

            # wait for plots to render
            page.wait_for_selector("#original svg")
            page.wait_for_selector("#disambiguated svg")

            svgs = page.query_selector_all("svg")

            if len(svgs) < 2:
                print("Error: expected 2 plots")
                return
            
            # save images
            svgs[0].screenshot(path=f"{fig}/original.png")
            svgs[1].screenshot(path=f"{fig}/disambiguated.png")
            print(f"Saved to figures in {fig}")

            browser.close()

    finally:
        server.terminate()
