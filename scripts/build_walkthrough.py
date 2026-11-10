"""Package recorded browser keyframes into a self-contained 90-second tour.

Usage: python scripts/build_walkthrough.py --frames /path/to/retail-demo
Presentation pacing is not a real-time performance recording.
"""

import argparse
import base64
import json
from pathlib import Path

STEPS = [
    (
        "01-workspace.jpg",
        "Define the review",
        "Real retail data, explicit metrics, and an immutable snapshot. The saved playbook can run again on a new period.",
    ),
    (
        "02-first-review.jpg",
        "Inspect the first period",
        "Displayed values come from fixed metric contracts, not numbers supplied by a language model.",
    ),
    (
        "03-computation.jpg",
        "Follow a computation reference",
        "The stored calculation exposes its source hash, query, parameters, denominator, and caveats.",
    ),
    (
        "04-refresh.jpg",
        "Refresh the same definitions",
        "A second snapshot is a fresh analysis. Historical outputs remain unchanged.",
    ),
    (
        "05-comparison.jpg",
        "Compare compatible periods",
        "Net sales rose from £309,792.40 to £388,700.17 in these historical records. The change is descriptive, not a causal effect.",
    ),
    (
        "06-portable-report.jpg",
        "Take the evidence with you",
        "The report retains exact currency values and working computation links. Live-model accuracy and participant outcomes remain deferred.",
    ),
]


def build(frames: Path, destination: Path):
    slides = [
        {
            "title": title,
            "caption": caption,
            "image": "data:image/jpeg;base64,"
            + base64.b64encode((frames / name).read_bytes()).decode(),
        }
        for name, title, caption in STEPS
    ]
    data = json.dumps(slides, ensure_ascii=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>90-second retail operations walkthrough</title><style>
body{font-family:system-ui;background:#101d2f;color:#f4f7fc;margin:0;padding:24px}main{max-width:1180px;margin:auto}
h1{font-size:28px}p{line-height:1.5;color:#c5d3e3}img{display:block;width:100%;border-radius:10px;border:1px solid #384a60;background:white}
button{padding:10px 16px;margin:8px 8px 8px 0;border:0;border-radius:5px;cursor:pointer}progress{width:100%;margin:16px 0;height:9px}
footer{font-size:12px;color:#aabed4;margin-top:12px}#caption{min-height:48px}
</style><main><h1>Auditable retail operations · 90-second tour</h1>
<p>Recorded browser states. Presentation pacing is not measured runtime. No live-model or participant results are implied.</p>
<button id="play">Play 90-second tour</button><button id="previous">Previous</button><button id="next">Next</button>
<span id="clock"></span><progress id="progress" max="90" value="0"></progress>
<h2 id="title"></h2><p id="caption"></p><img id="frame" alt="Recorded retail review interface">
<footer>Source data: Chen (2015), UCI Online Retail, CC BY 4.0 · Current contract/oracle checks: 140/140.
This tour is a browser-state presentation, not a video of real-time model execution.</footer></main>
<script>
const slides=SLIDES;let elapsed=0,playing=false;
function render(){const i=Math.min(5,Math.floor(elapsed/15));document.getElementById('frame').src=slides[i].image;
document.getElementById('title').textContent=(i+1)+' / 6 · '+slides[i].title;document.getElementById('caption').textContent=slides[i].caption;
document.getElementById('clock').textContent=elapsed+' / 90 seconds';document.getElementById('progress').value=elapsed;
document.getElementById('play').textContent=playing?'Pause':'Play 90-second tour';}
document.getElementById('play').onclick=()=>{if(elapsed>=90)elapsed=0;playing=!playing;render()};
document.getElementById('next').onclick=()=>{playing=false;elapsed=Math.min(75,(Math.floor(elapsed/15)+1)*15);render()};
document.getElementById('previous').onclick=()=>{playing=false;elapsed=Math.max(0,(Math.floor(elapsed/15)-1)*15);render()};
setInterval(()=>{if(playing){elapsed++;if(elapsed>=90)playing=false;render()}},1000);render();
</script></html>""".replace("SLIDES", data)
    destination.write_text(page)
    print(f"Recorded-state walkthrough: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("examples/retail-walkthrough.html"))
    args = parser.parse_args()
    build(args.frames, args.out)
