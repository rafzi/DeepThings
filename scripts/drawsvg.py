import drawSvg as svg

IMGW = 1200
IMGH = 200
BORDER = 50
d = None

def drawCube(x, y, w, h, z):
  global d
  d.append(svg.Lines(x, y, x+w, y, x+w, y+h, x, y+h, close=True, fill="white", stroke="black"))
  d.append(svg.Lines(x+w, y, x+w+z, y+z, x+w+z, y+h+z, x+w, y+h, close=True, fill="#555555", stroke="black"))
  d.append(svg.Lines(x, y+h, x+w, y+h, x+w+z, y+h+z, x+z, y+h+z, close=True, fill="#cccccc", stroke="black"))

def drawArrow(x, y, w, hw, hh):
  global d
  d.append(svg.Lines(x, y, x+w-hw/2, y, stroke="black"))
  d.append(svg.Lines(x+w-hw, y-hh/2, x+w, y, x+w-hw, y+hh/2, close=True, fill="black"))

def drawLayer(x, w, z):
  lx = x
  ly = 0
  lz = z/2
  drawCube(lx, ly, w, w, lz)

def drawNetArrow(x, label, sz):
  global d
  drawArrow(x+15, -15, 25, 5, 8)
  d.append(svg.Text(label, sz, x+15+25/2, -15-10, center=0.5, fill="black"))

def getLabel(t):
  if t == 0:
    return "NCONV"
  elif t == 1:
    return "LOP"
  elif t == 2:
    return "LIP"
  elif t == 3:
    return "FUSE1"
  elif t == 4:
    return "FUSE2"
  else:
    return "UNKNOWN"

def drawNet(ws, zs, ts, filename, layerlabeloffset=0, scalemult=1.0, fontsz=14):
  global d
  if len(ws) != len(zs):
    raise "mismatch"
  if len(zs) != len(ts)+1:
    raise "mismatch2"

  ARROWW = 30
  slotw = IMGW/len(ws)

  zs = [i * 0.25 for i in zs]

  maxw = max(ws)
  maxz = max(zs)
  e = [mw + mz/2 for mw,mz in zip(ws, zs)]
  maxe = max(e)
  scale = slotw / maxe * 0.85 * scalemult

  ws = [i * scale for i in ws]
  zs = [i * scale for i in zs]

  d = svg.Drawing(IMGW+2*BORDER, IMGH, origin=(-BORDER,-BORDER))

  for l in range(0, len(ws)):
    drawLayer(l*slotw, ws[l], zs[l])
    if l != len(ws)-1:
      d.append(svg.Text(str(l+layerlabeloffset), fontsz, (l+1)*slotw-5, -15+6, center=0.5, fill="black"))
      drawNetArrow((l+1) * slotw - ARROWW, getLabel(ts[l]), fontsz)

  d.saveSvg(filename)


# Types: 0=notconv, 1=lop, 2=lip, 3=f1, 4=f2

# YOLOv2
drawNet([608, 608, 304, 304, 152, 152, 152, 152, 76, 76, 76, 76, 38, 38, 38, 38, 38, 38,  19,  19,   19,  19,   19,  19,   19,   19,   38,  38, 19,  19,   19,   19],
        [3, 32, 32, 64, 64, 128, 64, 128, 128, 256, 128, 256, 256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024, 512, 64, 256, 1280, 1024, 425],
        [1, 0, 1, 0, 3, 4, 1, 0, 3, 4, 1, 0, 3, 4, 3, 4, 1, 0, 3, 4, 3, 4, 1, 3, 4, 0, 2, 0, 0, 3, 4],
        "net_yolofull.svg")

# YOLOv2 l0 - l15
drawNet([608, 608, 304, 304, 152, 152, 152, 152, 76, 76, 76, 76, 38, 38, 38, 38, 38],
        [3, 32, 32, 64, 64, 128, 64, 128, 128, 256, 128, 256, 256, 512, 256, 512, 256],
        [1, 0, 1, 0, 3, 4, 1, 0, 3, 4, 1, 0, 3, 4, 3, 4],
        "net_yolo1.svg")

# YOLOv2 l15-l31
drawNet([38,  38,  19,  19,   19,  19,   19,  19,   19,   19,   38,  38, 19,  19,   19,   19],
        [256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024, 512, 64, 256, 1280, 1024, 425],
        [1, 0, 3, 4, 3, 4, 1, 3, 4, 0, 2, 0, 0, 3, 4],
        "net_yolo2.svg",
        16)

# VGG-16
drawNet([256, 224, 224, 224, 112, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14, 7],
        [3, 3, 64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
        [0, 3, 4, 0, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4, 0],
        "net_vgg.svg")

# Extraction GoogleNet
drawNet([224, 112, 56, 56, 28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 7],
        [3, 64, 64, 192, 192, 182, 256, 256, 512, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 1024, 512, 1024, 512, 1012, 1000],
        [1, 0, 1, 0, 2, 3, 4, 1, 0, 2, 3, 4, 3, 4, 3, 4, 3, 4, 1, 0, 2, 3, 4, 3, 4],
        "net_extract.svg")

# AlexNet
drawNet([227, 55, 27, 27, 13, 13, 13, 13, 6],
        [3, 96, 96, 256, 256, 384, 384, 256, 256],
        [1, 0, 1, 0, 1, 3, 4, 0],
        "net_alexnet.svg")

###

# YOLOv2 OWP
drawNet([38,   38,  38,  38,  38,  38,  19,  19,   19,  19,   19,  19,   19,   19,   38,  38, 19,  19,   19,   19],
        [256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024, 512, 64, 256, 1280, 1024, 425],
        [3, 4, 3, 4, 1, 0, 3, 4, 3, 4, 1, 3, 4, 0, 2, 0, 0, 3, 4],
        "owpnet_yolo.svg",
        13, 1.6, fontsz=17)

# AlexNet OWP
drawNet([27, 27, 13, 13, 13, 13],
        [96, 256, 256, 384, 384, 256],
        [1, 0, 1, 3, 4],
        "owpnet_alexnet.svg",
        3, 0.65, fontsz=17)

# VGG-16 OWP
drawNet([56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14],
        [128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
        [1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4],
        "owpnet_vgg.svg",
        8, fontsz=17)

# Extraction GoogleNet OWP
drawNet([28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 7],
        [192, 182, 256, 256, 512, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 1024, 512, 1024, 512, 1012, 1000],
        [2, 3, 4, 1, 0, 2, 3, 4, 3, 4, 3, 4, 3, 4, 1, 0, 2, 3, 4, 3, 4],
        "owpnet_extract.svg",
        5, 1.6, fontsz=17)

