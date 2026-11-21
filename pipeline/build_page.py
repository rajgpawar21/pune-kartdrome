"""Assemble index.html: crop the satellite image, embed it with analysis.json into page_tpl.html.

Usage: python3 build_page.py <satellite.png> <geom.npz> <analysis.json> <page_tpl.html> <index.html>
"""
import sys, json, base64, numpy as np, cv2
sat, geom, analysis, tpl, out = sys.argv[1:6]
g = np.load(geom); C, N, h = g['C'], g['N'], g['h']
allp = np.vstack([C + N * h[:, None], C - N * h[:, None]]); pad = 60      # same crop as directions.py
x0, y0 = np.floor(allp.min(0) - pad).astype(int); x1, y1 = np.ceil(allp.max(0) + pad).astype(int)
x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, 1919); y1 = min(y1, 1079)
crop = cv2.imread(sat)[y0:y1, x0:x1]
ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 72])
img = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
page = open(tpl).read().replace('__DATA__', open(analysis).read()).replace('__IMG__', json.dumps(img))
head = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>\n'
open(out, 'w').write(head + page + '\n</body></html>\n')
print('wrote', out, len(page) // 1024, 'KB')
