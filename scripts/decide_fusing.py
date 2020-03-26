from __future__ import print_function
from ortools.linear_solver import pywraplp

# Number of nodes.
N = 2
# Number of layers.
L = 4
# Layer input sizes.
M = [4, 8, 16, 4]
# Layer output sizes.
K = [8, 16, 4, 4]

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

  # Create variables.
  o = []
  i = []
  f = []
  s = []
  r = []
  oc1 = []
  fc = []
  for l in range(0, L):
    o.append(solver.BoolVar('o' + str(l)))
    i.append(solver.BoolVar('i' + str(l)))
    f.append(solver.BoolVar('f' + str(l)))
    s.append(solver.BoolVar('s' + str(l)))
    r.append(solver.BoolVar('r' + str(l)))
    oc1.append(solver.BoolVar('oc1' + str(l)))
    fc.append(solver.BoolVar('fc' + str(l)))

    # A layer can only be of one layer type.
    ct1 = solver.Constraint(1, 1, 'ct1' + str(l))
    ct1.SetCoefficient(o[l], 1)
    ct1.SetCoefficient(i[l], 1)
    ct1.SetCoefficient(f[l], 1)
    ct1.SetCoefficient(s[l], 1)

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
      forceTo(solver, oc1[l], 0)
      forceTo(solver, fc[l], 0)
    else:
      oc1[l] = makeMultConstraint(solver, o[l], r[l-1], 'oc1')
      fc[l] = makeMultConstraint(solver, f[l], r[l-1], 'fc')


  # Last layer cannot be first fused one.
  forceTo(solver, f[L-1], 0)

  # First layer cannot be second fused one.
  forceTo(solver, s[0], 0)

  obj = solver.Objective()
  obj.SetMinimization()
  for l in range(0, L):
    # LOP
    obj.SetCoefficient(o[l], M[l] * (N-1) + (K[l]*N - K[l]) / N)
    obj.SetCoefficient(oc1[l], -(M[l] * (N-1)))
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
  for l in range(0, L):
    print('---')
    print('o' + str(l) + ' = ' + str(o[l].solution_value()))
    print('oc1' + str(l) + ' = ' + str(oc1[l].solution_value()))
    print('i' + str(l) + ' = ' + str(i[l].solution_value()))
    print('f' + str(l) + ' = ' + str(f[l].solution_value()))
    print('fc' + str(l) + ' = ' + str(fc[l].solution_value()))
    print('s' + str(l) + ' = ' + str(s[l].solution_value()))
    print('r' + str(l) + ' = ' + str(r[l].solution_value()))


if __name__ == "__main__":
  solve()
