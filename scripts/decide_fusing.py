from __future__ import print_function
from ortools.linear_solver import pywraplp

# Number of nodes.
N = 6

# Paper example:
#S = [4, 8, 16, 4, 4]
#CONV = [1, 1, 1, 1]

# YOLOv2
S = [608*608*3, 608*608*32, 304*304*32, 304*304*64, 152*152*64, 152*152*128, 152*152*64, 152*152*128, 76*76*128, 76*76*256, 76*76*128, 76*76*256, 38*38*256, 38*38*512, 38*38*256, 38*38*512, 38*38*256, 38*38*512, 19*19*512, 19*19*1024, 19*19*512, 19*19*1024, 19*19*512, 19*19*1024, 19*19*1024, 19*19*1024, 38*38*512, 38*38*64, 19*19*256, 19*19*1280, 19*19*1024, 19*19*425, 0]
W = [3*3*32*3, 0, 3*3*64*32, 0, 3*3*128*64, 1*1*64*128, 3*3*128*64, 0, 3*3*256*128, 1*1*128*256, 3*3*256*128, 0, 3*3*512*256, 1*1*256*512, 3*3*512*256, 1*1*256*512, 3*3*512*256, 0, 3*3*1024*512, 1*1*512*1024, 3*3*1024*512, 1*1*512*1024, 3*3*1024*512, 3*3*1024*1024, 3*3*1024*1024, 0, 1*1*64*512, 0, 0, 3*3*1024*1280, 1*1*425*1024, 0]
CONV = [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0]

# VGG-16
#S = [0, 224*224*3, 224*224*64, 224*224*64, 112*112*64, 112*112*128, 112*112*128, 56*56*128, 56*56*256, 56*56*256, 56*56*256, 28*28*256, 28*28*512, 28*28*512, 28*28*512, 14*14*512, 14*14*512, 14*14*512, 14*14*512, 7*7*512]
#CONV = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]

# Darknet Extraction
#S = [224*224*3, 112*112*64, 56*56*64, 56*56*192, 28*28*192, 28*28*128, 28*28*256, 28*28*256, 28*28*512, 14*14*512, 14*14*256, 14*14*512, 14*14*256, 14*14*512, 14*14*256, 14*14*512, 14*14*256, 14*14*512, 14*14*512, 14*14*1024, 7*7*1024, 7*7*512, 7*7*1024, 7*7*512, 7*7*1024, 7*7*1000]
#CONV = [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]

# AlexNet
#S = [227*227*3, 55*55*96, 27*27*96, 27*27*256, 13*13*256, 13*13*384, 13*13*384, 13*13*256, 6*6*256]
#CONV = [1, 0, 1, 0, 1, 1, 1, 0]

def getC_LOP(l):
  return M[l] - 0

genId = 0

# Create a BoolVar that is constrained to be equal to x1*x2
def makeMultConstraint(solver, x1, x2, name):
  global genId
  genId += 1

  y = solver.BoolVar(name + str(genId))

  # y <= x1  <=>  0 <= x1 - y <= 1
  ct1 = solver.Constraint(0, 1, 'ct1_gen' + str(genId))
  ct1.SetCoefficient(x1, 1)
  ct1.SetCoefficient(y, -1)

  # y <= x2  <=>  0 <= x2 - y <= 1
  ct2 = solver.Constraint(0, 1, 'ct2_gen' + str(genId))
  ct2.SetCoefficient(x2, 1)
  ct2.SetCoefficient(y, -1)

  # y >= x1 + x2 - 1  <=>  -1 <= y - x1 - x2 <= 1
  ct3 = solver.Constraint(-1, 1, 'ct3_gen' + str(genId))
  ct3.SetCoefficient(y, 1)
  ct3.SetCoefficient(x1, -1)
  ct3.SetCoefficient(x2, -1)

  return y

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

  L = len(CONV)

  # Create variables.
  o = []
  i = []
  f = []
  s = []
  r = []
  oc = []
  fc = []
  for l in range(0, L):
    o.append(solver.BoolVar('o' + str(l)))
    i.append(solver.BoolVar('i' + str(l)))
    f.append(solver.BoolVar('f' + str(l)))
    s.append(solver.BoolVar('s' + str(l)))
    r.append(solver.BoolVar('r' + str(l)))

    notconv = solver.BoolVar('notconv' + str(l))
    forceTo(solver, notconv, not CONV[l])

    # All LOP.
    #forceTo(solver, o[l], CONV[l])

    # A layer can only be of one layer type.
    ct1 = solver.Constraint(1, 1, 'ct1' + str(l))
    ct1.SetCoefficient(o[l], 1)
    ct1.SetCoefficient(i[l], 1)
    ct1.SetCoefficient(f[l], 1)
    ct1.SetCoefficient(s[l], 1)
    ct1.SetCoefficient(notconv, 1)

    # After a first fused layer, the next layer must be the second fused one.
    if l != 0:
      ct2 = solver.Constraint(0, 0, 'ct2' + str(l))
      ct2.SetCoefficient(f[l-1], -1)
      ct2.SetCoefficient(s[l], 1)

  for l in range(0, L):
    # r[l] = o[l] * (o[l+1] + f[l+1])
    if l == L-1:
      forceTo(solver, r[l], 0)
    else:
      rt1 = makeMultConstraint(solver, o[l], o[l+1], 'rt1')
      rt2 = makeMultConstraint(solver, o[l], f[l+1], 'rt2')
      ct_rt = solver.Constraint(0, 0, 'ct_rt' + str(l))
      ct_rt.SetCoefficient(rt1, 1)
      ct_rt.SetCoefficient(rt2, 1)
      ct_rt.SetCoefficient(r[l], -1)

    # Multiplied coefficients for some LOP and FUSE1 terms.
    if l == 0:
      oc.append(solver.BoolVar('oc' + str(l)))
      fc.append(solver.BoolVar('fc' + str(l)))
      forceTo(solver, oc[l], 0)
      forceTo(solver, fc[l], 0)
    else:
      oc.append(makeMultConstraint(solver, o[l], r[l-1], 'oc'))
      fc.append(makeMultConstraint(solver, f[l], r[l-1], 'fc'))


  # Last layer cannot be first fused one.
  forceTo(solver, f[L-1], 0)

  # First layer cannot be second fused one.
  forceTo(solver, s[0], 0)

  # Test alternatives.
  '''
  forceTo(solver, o[0], 1)
  forceTo(solver, f[2], 1)
  forceTo(solver, s[3], 1)
  forceTo(solver, f[4], 1)
  forceTo(solver, s[5], 1)
  forceTo(solver, o[6], 1)
  forceTo(solver, f[7], 1)
  forceTo(solver, s[8], 1)
  forceTo(solver, i[10], 1)
  forceTo(solver, f[13], 1)
  forceTo(solver, s[14], 1)
  '''

  # Layer input sizes.
  M = S[:-1]
  # Layer output sizes.
  K = S[1:]

  obj = solver.Objective()
  obj.SetMinimization()
  for l in range(0, L):
    # LOP
    obj.SetCoefficient(o[l], M[l] * (N-1) + (K[l]*N - K[l]) / N)
    obj.SetCoefficient(oc[l], -(M[l] * (N-1)))
    obj.SetCoefficient(r[l], K[l] * (N-1) - ((K[l]*N - K[l]) / N))
    # LIP
    obj.SetCoefficient(i[l], (M[l]*N - M[l]) / N + K[l] * (N-1))
    # FUSE1
    obj.SetCoefficient(f[l], M[l] * (N-1))
    obj.SetCoefficient(fc[l], -(M[l] * (N-1)))
    # FUSE2
    obj.SetCoefficient(s[l], K[l] * (N-1))


  solver.Solve()
  print('Total C: ', obj.Value())
  print("Communication size:", "{:,}".format(4*int(obj.Value())), "Bytes")
  for l in range(0, L):
    def getLayerType():
      if o[l].solution_value():
        return "LOP" + (" with reuse" if r[l].solution_value() else "")
      elif i[l].solution_value():
        return "LIP"
      elif f[l].solution_value():
        return "FUSE1"
      elif s[l].solution_value():
        return "FUSE2"
      else:
        return "Not convolutional"
    print("Layer", l, "is:", getLayerType())


if __name__ == "__main__":
  solve()
