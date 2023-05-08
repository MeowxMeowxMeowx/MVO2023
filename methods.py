import numpy as np
from scipy.optimize import OptimizeResult

def grad_const(jac, x0=None, args=(), callback=None,
              options={}, **kwargs):
    """
    Minimization method

    Parameters
    ----------
    jac : callable f(x, *args)
        Gradient of objective function.
        
    x0 : array-like
        Initial guess. 
    
    args : tuple, optional
        Extra arguments passed to the `jac`.
    
    callback : callable f(x), optional
        Function called after each iteration.
    
    options : dict, optional
        A diactionary with solver options.
            maxiter : int
                Maximum number of iterations to perform.
            tol : float
                Tolerance for termination.
            stepsize : float
                Factor for calculatng step in one iteration.

    **kwargs : dict, optional
        Other parameters passed to `grad_const`.

    Raises
    ------
    ValueError
        if `x0` is not provided.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        success a Boolean flag indicating if the optimizer exited successfully,
        message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
    """
    
    if x0 is None:
        raise ValueError("Must provide initial guess `x0`!")

    maxiter = options.get("maxiter", 1000)
    tol = options.get("tol", 1e-6)
    stepsize = options.get("stepsize", 1)
    callback_step = options.get("callback_step", False)

    x = np.array(x0)

    for it in range(maxiter):
        s = -jac(x,*kwargs.get("vec"))
        x = x + stepsize*s

        if callback is not None:
          if callback_step:
            callback(x)
          else:
            callback(np.linalg.norm(jac(x,*kwargs.get("vec")))-tol)
        if np.linalg.norm(jac(x,*kwargs.get("vec"))) < tol:
            break
    
    
    success = np.linalg.norm(jac(x,*kwargs.get("vec"))) < tol

    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"
    
    return OptimizeResult(x=x,success=success,fun=jac(x,*kwargs.get("vec")),nit=it+1) 


def bisection(dfun, bounds=None, args=(), callback=None,
              options={}, **kwargs):
    """
    Minimization method

    Parameters
    ----------
    dfun : callable f(x, *args)
        Derivative of objective function to minimize.
    
    bounds : tuple of numeric sequence
        Two items corresponding to the optimization bounds.
    
    args : tuple, optional
        Extra arguments passed to the objective function.
    
    callback : callable f(x), optional
        Function called after each iteration.
    
    options : dict, optional
        A diactionary with solver options.
            maxiter : int
                Maximum number of iterations to perform.
            tol : float
                Tolerance for termination

    **kwargs : dict, optional
        Other parameters passed to `bisection`.

    Raises
    ------
    ValueError
        if `bounds` is not provided.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        success a Boolean flag indicating if the optimizer exited successfully,
        message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
    """
    
    maxiter = options.get("maxiter", 1000)
    if bounds is None:
      raise ValueError("Must provide `bounds` parameter!\n Tuple of numeric sequence: two items corresponding to the optimization bounds.")
    
    a, b = bounds
    tol = options.get("tol", 1e-8)
    maxiter = options.get("maxiter", 1000)
    # maxiter = min(maxiter, np.ceil(np.log2(b-a) - np.log2(tol)))

    for it in range(maxiter):
        pass
        c = (a+b)/2

        if callback is not None:
            midpoint = (a+b) / 2
            callback(midpoint)
        
        if b - a < tol:
            break
        tmp = dfun(c)
        if tmp>0:
          b = c
        else:
          a = c
    
    success = (b - a) < tol

    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"
    
    return OptimizeResult(x=(a+b)/2, success=success, 
                          message=msg, nit=it, njev=it, tol=tol) 



def cauchy(jac, x0=None, args=(), callback=None,
              options={}, **kwargs):
    """
    Minimization method

    Parameters
    ----------
    jac : callable f(x, *args)
        Gradient of objective function.
        
    x0 : array-like
        Initial guess. 
    
    args : tuple, optional
        Extra arguments passed to the `jac`.
    
    callback : callable f(x), optional
        Function called after each iteration.
    
    options : dict, optional
        A diactionary with solver options.
            maxiter : int
                Maximum number of iterations to perform.
            tol : float
                Tolerance for termination
            max_stepsize : float
                Maximum stap size allowed.

    **kwargs : dict, optional
        Other parameters passed to `cauchy`.

    Raises
    ------
    ValueError
        if `x0` is not provided.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        success a Boolean flag indicating if the optimizer exited successfully,
        message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
    """
    if x0 is None:
        raise ValueError("Must provide initial guess `x0`!")


    maxiter = options.get("maxiter", 1000)
    tol = options.get("tol", 1e-8)
    max_stepsize = options.get("stepsize", 1)
    #f = args
    x = np.array(x0)
    l = max_stepsize

    for it in range(maxiter):
        s = -jac(x,*kwargs.get("vec"))
        fi = lambda l : jac(x+l*s,*kwargs.get("vec"))@s
        bis = bisection(fi,(0,max_stepsize),(0,max_stepsize))
        stepsize = bis.x
        x = x + stepsize*s
        if callback is not None:
            callback(np.linalg.norm(jac(x,*kwargs.get("vec")))-tol)
        
        if np.linalg.norm(jac(x,*kwargs.get("vec"))) < tol:
            break
    
    
    success = np.linalg.norm(jac(x,*kwargs.get("vec"))) < tol

    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"
    
    return OptimizeResult(x=x,success=success,fun=jac(x,*kwargs.get("vec")),nit=it+1) 



def backtracking_line_search(fun, jac, x0, args=(), s=None, max_stepsize=1,
                             rho=0.1, c=0.5, options={}):
    """
    Implements the backtracking line search algorithm to find the step size alpha
    for gradient descent.

    Parameters:
        f : callable
            Objective function.
        g : callable
            Gradient function.
        x0 : array-like
            Current point.
        s : array-like, optional
            Search direction. If not provided, gradient direction is used.
        max_stepsize : numeric, optional.
            Initial step size. Default is 1.
        rho : numeric, optional
            Shrinkage factor. Nonnegative number, less than one. Default is 0.1.
        c : numeric, optional
            Armijo condition parameter. Default is 0.05.
        options : dict, optional
            A diactionary with solver options.
                maxiter : int
                    Maximum number of iterations to perform.

    Returns
    -------
        stepsize : float
            Step size that satisfies the Armijo condition.
    """

    assert 0 < rho < 1, "Rho parameter should be from range (0, 1)!"

    maxiter = options.get("maxiter", 100)

    x = np.array(x0)
    if s is None:
        s = -jac(x, *args)
        df_s = -s@s
    else:
        df_s = jac(x, *args) @ s
    
    stepsize = max_stepsize

    fx = fun(x, *args[:-1])
    
    
    for it in range(maxiter):
        if fun(x + stepsize*s, *args[:-1]) > fx + c*stepsize*df_s:
            stepsize *= rho
        else:
            return stepsize
    
    return stepsize


def backtrack(jac, x0=None, args=(), callback=None,
              options={}, **kwargs):
    """
    Minimization method

    Parameters
    ----------
    jac : callable f(x, *args)
        Gradient of objective function.
        
    x0 : array-like
        Initial guess. 
    
    args : tuple, optional
        Extra arguments passed to the `jac`.
    
    callback : callable f(x), optional
        Function called after each iteration.
    
    options : dict, optional
        A diactionary with solver options.
            maxiter : int
                Maximum number of iterations to perform.
            tol : float
                Tolerance for termination

    **kwargs : dict, optional
        Other parameters passed to `cauchy`.

    Raises
    ------
    ValueError
        if `x0` is not provided.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        success a Boolean flag indicating if the optimizer exited successfully,
        message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
    """
    
    if x0 is None:
        raise ValueError("Must provide initial guess `x0`!")

    maxiter = options.get("maxiter", 1000)
    tol = options.get("tol", 1e-8)
    stepsize = options.get("stepsize", 1e-5)

    maxiter_step_search = options.get("maxiter_step_search", 100)
    rho = options.get("rho", 0.5)
    c = options.get("c", 1e-4)

    x = np.array(x0)

    for it in range(maxiter):
        s = -jac(x,*kwargs.get("vec"))
        l = backtracking_line_search(*args,jac,x, rho = rho, c = c,max_stepsize=maxiter_step_search,args=kwargs.get("vec"))
        x = x + s*l
        if callback is not None:
            callback(np.linalg.norm(jac(x,*kwargs.get("vec")))-tol)
        
        if np.linalg.norm(jac(x,*kwargs.get("vec")))<tol: 
            break
    
    
    success = np.linalg.norm(jac(x,*kwargs.get("vec"))) < tol  

    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"
    
    return OptimizeResult(x=x,success=success,fun=jac(x,*kwargs.get("vec")),nit=it+1) 