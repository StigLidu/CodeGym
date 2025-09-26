# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import os

out_dir = "asset"
os.makedirs(out_dir, exist_ok=True)

path = os.path.join(out_dir, "codegym_logo_v2.svg")

content = '''<svg width="1000" height="160" viewBox="0 0 1000 160" xmlns="http://www.w3.org/2000/svg">
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
  <g transform="scale(1,1) translate(0,10)">
    <text x="50%" y="100" text-anchor="middle"
          font-family="Courier New, monospace"
          font-weight="800" font-size="150"
          fill="url(#g2)" filter="url(#soft)">
      &lt;CodeGym/&gt;
    </text>
  </g>
</svg>'''

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
