import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--user','ortools'])

from ortools.linear_solver import pywraplp


def LinearProgrammingExample():
    """Linear programming sample."""
    # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create the two variables and let them take on any non-negative value.
    x1 = solver.NumVar(0, solver.infinity(), 'x1')
    x2 = solver.NumVar(0, solver.infinity(), 'x2')

    print('Number of variables =', solver.NumVariables())

    # Constraint 0
    solver.Add(-x1 + x2 <= 11)

    # Constraint 1
    solver.Add(x1 + x2 <= 27)

    # Constraint 2
    solver.Add(2 * x1 + 5 * x2 <= 90)

    print('Number of constraints =', solver.NumConstraints())

    # Objective function: 4x1 + 6x2.
    solver.Maximize(4 * x1 + 6 * x2)

    # Solve the system.
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print('x1 =', x1.solution_value())
        print('x2 =', x2.solution_value())
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())


LinearProgrammingExample()
