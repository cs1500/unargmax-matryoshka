import cvxpy as cp
import numpy as np

def unargm_opt(j, W):
    """
    Solve the feasibility problem for jth label.

    Parameters:
    -----------
        - j: integer denoting label index
        - W: full unembedding matrix
    """
    n, d = W.shape; epsilon = 0.1
    z = cp.Variable(d)
    obj = cp.Minimize(0)
    constraints = [
        (W[i, :] - W[j, :]) @ z <= -epsilon
        for i in range(0, n) if i != j # list comprehension
    ]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return prob.status

def unargm(W=None, cat=False):
    """
    Given an unembedding matrix of n by d, check if there are any
    unargmaxable classes.

    Parameters:
    -----------
        - W: matrix of n by d
    """

    # --- parse parameters ---
    if W is None: # test
        W = np.array([
            [1, 2],
            [3, 4],
        ])
    n, d = W.shape

    # --- main optimisation loop
    result = []
    for i in range(0, n):
        wi = W[i, :]
        status = unargm_opt(j=i, W=W) # 1 for not maxable, 0 otherwise
        if status in ["optimal", "optimal_inaccurate"]:
            out = 0
        else:
            out = 1
        result.append(out)

    if cat:
        print("give result")
    return result

t = unargm()
print(t)