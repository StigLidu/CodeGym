import os

out_dir = "asset"
os.makedirs(out_dir, exist_ok=True)

path = os.path.join(out_dir, "codegym_brackets_courier.svg")

content = '''<svg width="1200" height="300" viewBox="0 0 1200 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g2" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="#61DAFB"/>
      <stop offset="50%" stop-color="#7C3AED"/>
      <stop offset="100%" stop-color="#22C55E"/>
    </linearGradient>
    <filter id="soft" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <g transform="scale(1,1) translate(0,30)">
    <text x="60" y="200" font-family="Courier New, monospace"
          font-weight="800" font-size="150" fill="url(#g2)" filter="url(#soft)">&lt;CodeGym/&gt;</text>
  </g>
</svg>'''

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

path
