from __future__ import print_function
from ortools.linear_solver import pywraplp

# Number of nodes.
N = 1
# Model to use.
model = "yolo"  # test, yolo, vgg, extract, alex

if model == "test":
  # Test model
  S = [100*100*3, 100*100*20, 50*50*20, 50*50*50, 25*25*50, 25*25*100, 12*12*100, 12*12*500]
  W = [3*3*20*3, 0, 3*3*50*20, 0, 3*3*100*50, 0, 3*3*500*100]
  CONV = [1, 1, 1, 1, 1, 1, 1]
elif model == "yolo":
  # YOLOv2
  S = [608*608*3, 608*608*32, 304*304*32, 304*304*64, 152*152*64, 152*152*128, 152*152*64, 152*152*128, 76*76*128, 76*76*256, 76*76*128, 76*76*256, 38*38*256, 38*38*512, 38*38*256, 38*38*512, 38*38*256, 38*38*512, 19*19*512, 19*19*1024, 19*19*512, 19*19*1024, 19*19*512, 19*19*1024, 19*19*1024, 19*19*1024, 38*38*512, 38*38*64, 19*19*256, 19*19*1280, 19*19*1024, 19*19*425, 0]
  W = [3*3*32*3, 0, 3*3*64*32, 0, 3*3*128*64, 1*1*64*128, 3*3*128*64, 0, 3*3*256*128, 1*1*128*256, 3*3*256*128, 0, 3*3*512*256, 1*1*256*512, 3*3*512*256, 1*1*256*512, 3*3*512*256, 0, 3*3*1024*512, 1*1*512*1024, 3*3*1024*512, 1*1*512*1024, 3*3*1024*512, 3*3*1024*1024, 3*3*1024*1024, 0, 1*1*64*512, 0, 0, 3*3*1024*1280, 1*1*425*1024, 0]
  CONV = [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0]
elif model == "vgg":
  # VGG-16
  S = [0, 224*224*3, 224*224*64, 224*224*64, 112*112*64, 112*112*128, 112*112*128, 56*56*128, 56*56*256, 56*56*256, 56*56*256, 28*28*256, 28*28*512, 28*28*512, 28*28*512, 14*14*512, 14*14*512, 14*14*512, 14*14*512, 7*7*512]
  W = [0, 3*3*64*3, 3*3*64*64, 0, 3*3*128*64, 3*3*128*128, 0, 3*3*256*128, 3*3*256*256, 3*3*256*256, 0, 3*3*512*256, 3*3*512*512, 3*3*512*512, 0, 3*3*512*512, 3*3*512*512, 3*3*512*512, 0]
  CONV = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
elif model == "extract":
  # Darknet Extraction
  S = [224*224*3, 112*112*64, 56*56*64, 56*56*192, 28*28*192, 28*28*128, 28*28*256, 28*28*256, 28*28*512, 14*14*512, 14*14*256, 14*14*512, 14*14*256, 14*14*512, 14*14*256, 14*14*512, 14*14*256, 14*14*512, 14*14*512, 14*14*1024, 7*7*1024, 7*7*512, 7*7*1024, 7*7*512, 7*7*1024, 7*7*1000]
  W = [7*7*64*3, 0, 3*3*192*64, 0, 1*1*128*192, 3*3*256*128, 1*1*256*256, 3*3*512*256, 0, 1*1*256*512, 3*3*512*256, 1*1*256*512, 3*3*512*256, 1*1*256*512, 3*3*512*256, 1*1*256*512, 3*3*512*256, 1*1*512*512, 3*3*1024*512, 0, 1*1*512*1024, 3*3*1024*512, 1*1*512*1024, 3*3*1024*512, 1*1*1000*1024]
  CONV = [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
elif model == "alex":
  # AlexNet
  S = [227*227*3, 55*55*96, 27*27*96, 27*27*256, 13*13*256, 13*13*384, 13*13*384, 13*13*256, 6*6*256]
  W = [11*11*96*3, 0, 5*5*256*96, 0, 3*3*384*256, 3*3*384*384, 3*3*256*384, 0]
  CONV = [1, 0, 1, 0, 1, 1, 1, 0]
else:
  print("unknown model")
  exit(1)

genId = 0

def addMaxFuncVar(solver, vars):
  global genId
  genId += 1

  M = 1e9

  # C = max(v_0, ..., v_N)
  C = solver.IntVar(0, solver.Infinity(), 'C')

  h = []
  for i, v in enumerate(vars):
    id = str(i) + "_" + str(genId)
    h.append(solver.BoolVar('h' + id))

    # C >= v_i
    ct1 = solver.Constraint(0, solver.Infinity(), 'ct1_gen' + id)
    ct1.SetCoefficient(v, -1)
    ct1.SetCoefficient(C, 1)

    # C <= v_i + (1-h_i)M
    ct2 = solver.Constraint(-M, solver.Infinity(), 'ct2_gen' + id)
    ct2.SetCoefficient(v, 1)
    ct2.SetCoefficient(h[i], -M)
    ct2.SetCoefficient(C, -1)

  # sum(h_i) = 1
  ct3 = solver.Constraint(1, 1, 'ct3_gen' + str(genId))
  for hi in h:
    ct3.SetCoefficient(hi, 1)

  return C


def forceTo(solver, v, value):
  global genId
  genId += 1

  ct = solver.Constraint(value, value, 'ct_gen' + str(genId))
  ct.SetCoefficient(v, 1)


def solve():
  solveType = pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
  solver = pywraplp.Solver("deepthings_solver", solveType)

  if len(S) != len(CONV)+1:
    print("length of S and CONV do not match")
    exit(1)

  if len(W) != len(CONV):
    print("length of W and CONV do not match")
    exit(1)

  L = len(CONV)
  # Layer input sizes.
  M = S[:-1]
  # Layer output sizes.
  K = S[1:]

  a = []
  b = []
  for l in range(0, L):
    print("W[", l, "]: ", W[l])
    a.append(solver.IntVar(0, solver.Infinity(), 'a' + str(l)))
    b.append(solver.BoolVar('b' + str(l)))

    if CONV[l]:
      # a[l] = b[l](M[l] + K[l]) + (1-b[l])(M[l] + K[l])/N
      DATA = (M[l] + K[l])/N
      cta = solver.Constraint(-DATA, -DATA, 'cta' + str(l))
      cta.SetCoefficient(a[l], -1)
      cta.SetCoefficient(b[l], M[l] + K[l] - DATA)
    else:
      forceTo(solver, a[l], 0)

    # b[l] >= b[l-1]
    if l != 0:
      ct1 = solver.Constraint(0, 1, 'ct1' + str(l))
      ct1.SetCoefficient(b[l], 1)
      ct1.SetCoefficient(b[l-1], -1)

  #forceTo(solver, b[12], 1)

  # max(a[l]) + sum(b[l]W[l]/N + (1-b[l])W[l])
  obj = solver.Objective()
  obj.SetMinimization()
  obj.SetCoefficient(addMaxFuncVar(solver, a), 1)
  for l in range(0, L):
    obj.SetCoefficient(b[l], W[l]/N - W[l])


  solver.Solve()
  F_n = obj.Value()
  for l in range(0, L):
    F_n += W[l]

  print("Total F_n:", F_n)
  for l in range(0, L):
    print(l, " / a[l] =", a[l].solution_value(), "/ b[l] =", b[l].solution_value())

  for l in range(0, L):
    if b[l].solution_value():
      print("Breakpoint to start OPFD should be at:", l)
      break

  max_a = 0
  w_tail = 0
  for l in range(0, L):
    if not CONV[l]:
      continue
    a_i = M[l] + K[l]
    if b[l].solution_value() == 0:
      a_i /= N
      w_tail += W[l] / N
    else:
      w_tail += W[l]
    max_a = a_i if a_i > max_a else max_a
    print("l:", l, "i/o:", (M[l]+K[l])/1000, "w:", W[l]/1000)
  print("max_a =", max_a)
  print("w_tail =", w_tail)
  print("total (single device) =", max_a + w_tail)



if __name__ == "__main__":
  solve()
