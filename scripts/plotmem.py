import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1: YOLOv2, 2: AlexNet, 3: VGG-16, 4: GoogLeNet
model = 4

dfs = pd.read_excel("t.xlsx", sheet_name=None, header=None)
if model == 1:
  ms = "YOLOv2"
elif model == 2:
  ms = "AlexNet"
elif model == 3:
  ms = "VGG-16"
elif model == 4:
  ms = "GoogLeNet"
sh = dfs["Memory"]
print(sh)


labels = ["1", "2", "3", "4", "5", "6"]
x = np.arange(len(labels))

fig, ax = plt.subplots()

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
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

def addData():
  y = []
  for i in range(0, 6):
    y.append(-sh[i + 1][model] + sh[1][model])
  y = np.array(y) / 1000
  g = ax.bar(x, y)

  autolabel(g)

addData()


#plt.ylim(plt.ylim()*1.1)
ybot, ytop = plt.ylim()
plt.ylim(ybot, ytop*1.05)
ax.set_xlabel("Number of devices")
ax.set_ylabel("Memory savings over 1 device [MB]")
ax.set_xticks(x)
ax.set_xticklabels(labels)

plt.savefig("plot_mem.pdf")
plt.show()
