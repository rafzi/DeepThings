import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1: YOLOv2, 2: AlexNet, 3: VGG-16, 4: GoogLeNet
model = 4
LINEPLOT = True

dfs = pd.read_excel("t.xlsx", sheet_name=None, header=None)
if model == 1:
  ms = "YOLOv2"
elif model == 2:
  ms = "AlexNet"
elif model == 3:
  ms = "VGG-16"
elif model == 4:
  ms = "GoogLeNet"
sh = dfs[ms]
print(sh)


labels = ["1", "2", "3", "4", "5", "6"]
x = np.arange(len(labels))

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots()
plt.subplots_adjust(top=0.95, right=0.95)

# Workaround for this: https://bugs.python.org/issue32790
def fmtFlt(f, digits):
  s = ("{:#." + str(digits) + "g}").format(f)
  sz = len(s) - 1
  if sz < digits:
    s += "0"
  if s[-1] == ".":
    s = s[:-1]
  return s

def autolabel(rects):
  """Attach a text label above each bar in *rects*, displaying its height."""
  for rect in rects:
    height = rect.get_height()
    ax.annotate(fmtFlt(height, 3),
                xy=(rect.get_x() + 1.2*rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', rotation=90, fontsize=9.5)

def addData(speed, fused):
  y = []
  lineindex = -4 + (speed)*(13+4)
  addindex = 1 if fused else 0
  for i in range(0, 6):
    y.append(sh[5*2 + addindex][lineindex] / sh[i*2 + addindex][lineindex])
  y = np.array(y)# / 1000
  y = np.flip(y)
  l = ("OWP @ " if fused else "LOP @ ") + \
    ("1 GBit/s" if speed == 1 else ("100 MBit/s" if speed == 2 else "10 MBit/s"))
  color = "C1" if fused else "C0"
  if LINEPLOT:
    color = "C3" if speed == 1 else ("C4" if speed == 2 else "C1")
    #line = "o" if speed == 1 else ("v" if speed == 2 else "s")
    line = "o" if fused else "s"
    line += "--" if fused else "-"
    ax.plot(x, y, line, label=l, color=color)
  else:
    barw = 0.15
    bars = 6
    i = 2 * (-speed+4-1) + int(fused)
    #patterns = ["\\\\", "//", "||", "--", "..", "OO"]
    patterns = ["\\\\", "\\\\", "//", "//", "..", ".."]
    g = ax.bar(x + barw/2 - bars/2*barw + i * barw, y, barw, label=l, color=color,
      hatch=patterns[i], alpha=0.99)
    #autolabel(g)

# 1: 1gbit, 2: 100mbit, 3: 10mbit
addData(1, True)
addData(1, False)
addData(2, True)
addData(2, False)
addData(3, True)
addData(3, False)


#plt.ylim(plt.ylim()*1.1)
ybot, ytop = plt.ylim()
plt.ylim(ybot, ytop*1.05)
ax.set_xlabel("Number of devices")
ax.set_ylabel("Run time speedup over one device")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig("plot_runtime.pdf")
plt.show()
