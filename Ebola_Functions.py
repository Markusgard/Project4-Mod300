import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
Beta = 0.06
Population_in_Dirdal = 683
Population_in_Sokndal = 3305
Starting_Zombies = 1
Timesteps = np.linspace(0, 200)

class Town:
    """
    Represents a town affected by a zombie outbreak.
    
    This class stores the population, infection rate, and other model parameters
    used in the SZ (Susceptible–Zombie) model and its extensions. Upon creation,
    it initializes parameters and can be used to generate analytical or numerical
    results for the given location.

    Attributes
    ----------
    Name : str
        Name of the town (e.g., "Dirdal" or "Sokndal").
    S0 : int
        Initial number of susceptible individuals (total population at t=0).
    Z0 : int
        Initial number of zombies.
    beta : float
        Infection rate constant (per hour).
    T : np.ndarray
        Array of time steps for the simulation.
    eps : float
        Small tolerance parameter (not actively used in this version).
    alpha : float
        Constant background rate of zombie killing (per hour).
    sigma : float
        Transition rate from exposed to zombie (1/incubation period).
    w_t : float
        Time-dependent attack rate (set to 0 unless counterattacks are modeled).
    a : float
        Attack strength scaling factor for ω(t) when used.
    E0 : float
        Initial number of exposed individuals.
    R0 : float
        Basic reproduction number approximation (β / (α + ω)).
    attacks : list[int]
        List of times (in hours) when counterattacks occur, for SEZR models.
    """

    def __init__(self, location, population, beta=0.06, Timesteps=np.linspace(0, 200),
                 Starting_Zombies=1, eps=1, E0=0, alpha=0.02, w_t=0, sigma=1/24, attackstrenth=20,landa = 1/7,gamma=1/7):
        """
        Initialize a Town instance with parameters for the SZ or SEZR model.
        Not all parameters are needed for this first task, but to avoid unnecessary code duplication, this initialization class will be used for all towns modeled in the project.
        Parameters
        ----------
        location : str
            Name of the town (used for labeling plots).
        population : int
            Total initial population (susceptible individuals at t=0).
        beta : float, optional
            Infection rate constant (default = 0.06).
        Timesteps : np.ndarray, optional
            Array of simulation times (default = np.linspace(0, 200)).
        Starting_Zombies : int, optional
            Initial number of zombies (default = 1).
        eps : float, optional
            Small tolerance value (default = 1).
        E0 : int, optional
            Initial number of exposed individuals (default = 0).
        alpha : float, optional
            Constant kill rate for zombies (default = 0.02).
        w_t : float, optional
            Time-dependent attack rate (default = 0, meaning no counterattacks).
        sigma : float, optional
            Incubation rate (1 / incubation period) (default = 1/24).
        attackstrenth : float, optional
            Multiplier for attack strength in SEZR models (default = 20).
        """
        self.Name = location
        self.S0 = population
        self.Z0 = Starting_Zombies
        self.beta = beta
        self.T = Timesteps
        self.eps = eps
        self.alpha = alpha
        self.sigma = sigma
        self.w_t = w_t
        self.a = attackstrenth * self.beta
        self.E0 = E0
        self.R0 = beta / (alpha + w_t)
        self.init_conditions()
        self.landa = landa
        self.gamma = gamma

    def init_conditions(self):
        """
        Initialize starting vectors and parameters for later use in model equations.
        
        Sets up arrays for initial populations and default attack timings.
        """
        self.b0 = np.array([self.Z0, self.S0 - self.Z0])
        self.c0 = np.array([self.Z0, self.S0 - self.Z0])
        self.k0 = np.array([self.S0, self.E0, self.Z0, self.R0])
        self.t0 = np.array([1/24, 0.02])
        self.attacks = [100, 124, 148, 172, 196]


# --- Instantiate towns ---
Dirdal = Town('Dirdal', Population_in_Dirdal)
Sokndal = Town('Sokndal', Population_in_Sokndal)

# Code 1: Definition of the SZ analytical solution
def SZ_solution(town, t=None):
    """
    Compute the analytical solution to the two-compartment SZ model.

    This function uses the analytical expressions:
        S(t) = (S0 + Z0)(S0/Z0) * exp(-βt) / [1 + (S0/Z0) * exp(-βt)]
        Z(t) = (S0 + Z0) / [1 + (S0/Z0) * exp(-βt)]

    Parameters
    ----------
    town : Town
        Instance of the Town class containing model parameters.
    t : np.ndarray, optional
        Array of time steps (if None, uses town.T).

    Returns
    -------
    S : np.ndarray
        Array of susceptible individuals at each time step.
    Z : np.ndarray
        Array of zombie individuals at each time step.
    """
    if t is None:
        t = town.T

    S = town.S0 - town.Z0

    CHP = (S * (S / town.Z0) * np.exp(-town.beta * t)) / (1 + (S / town.Z0) * np.exp(-town.beta * t))
    CZP = (S + town.Z0) / (1 + (S / town.Z0) * np.exp(-town.beta * t))
    return np.array(CHP), np.array(CZP)


def Calculating_fraction_of_zombies(town, CZP):
    """
    Compute the fraction of the total population that are zombies.

    Parameters
    ----------
    town : Town
        Town object containing population and parameters.
    CZP : np.ndarray
        Array of zombie counts from the SZ model.

    Returns
    -------
    Fraction_of_zombies : np.ndarray
        The proportion of zombies relative to the total population.
    """
    Fraction_of_zombies = CZP / town.S0
    return Fraction_of_zombies


def plot(town, Z):
    """
    Plot the fraction of zombies in a given town over time.

    Parameters
    ----------
    town : Town
        Instance of the Town class containing time and metadata.
    Z : np.ndarray
        Array of zombie fractions (Z/N) to plot.
    """
    plt.plot(town.T, Z, label=town.Name)



# --- Task 2: Critical points and long-term behavior (using Town class setup) ---
import numpy as np
import matplotlib.pyplot as plt

def dZdt(Z, town):
    """
    Compute the rate of change of zombies (dZ/dt) in a town using the SZ model.

    The SZ model assumes that zombie population growth depends on a transmission
    rate `beta` and saturates as it approaches the total susceptible population `S0`.

    Parameters
    ----------
    Z : float or np.ndarray
        Current zombie population. Can be a scalar or an array for vectorized computations.
    town : object
        Town object containing model parameters:
        - `town.beta` : float, infection/transmission rate
        - `town.S0` : float, total initial susceptible population

    Returns
    -------
    dZ : float or np.ndarray
        Rate of change of the zombie population (dZ/dt) according to the SZ model.

    Notes
    -----
    The model equation used is:
        dZ/dt = beta * Z * (1 - Z / S0)
    which represents logistic-like growth of zombies constrained by the susceptible population.

    """
    return town.beta * Z * (1 - Z / town.S0)

# --- Phase plot for both towns ---
Z_vals_d = np.linspace(0, Dirdal.S0, 300)
Z_vals_s = np.linspace(0, Sokndal.S0, 300)

dZ_d = dZdt(Z_vals_d, Dirdal)
dZ_s = dZdt(Z_vals_s, Sokndal)


# --- Numerical check of long-term behavior using analytical solution ---
_, Z_d = SZ_solution(Dirdal)
_, Z_s = SZ_solution(Sokndal)

# print("Long-term (steady-state) results:")
# print(f"  Dirdal:  Z(t→∞) ≈ {Z_d[-1]:.1f} ≈ N = {Dirdal.S0}")
# print(f"  Sokndal: Z(t→∞) ≈ {Z_s[-1]:.1f} ≈ N = {Sokndal.S0}")

# print("\nInterpretation:")
# print("  • Z* = 0  → Unstable equilibrium (any infection grows).")
# print("  • Z* = N  → Stable equilibrium (entire population zombified).")
# print("  ⇒ For any β > 0 and Z₀ > 0, Z(t) → N and S(t) → 0 as t → ∞.")


# Exercise 2 — General ODE Solver + SZ model tests
# Implements forward Euler, RK2, RK4 for dy/dt = f(y, t).
# Applies to SZ-model: dS/dt = -β S Z / N, dZ/dt = +β S Z / N.
# Compares against analytical solution and studies step-size effects.

import pandas as pd

def step(t, c_old, dt, f, method, *args, **kwargs):
    """
    Compute a single time step for an ODE using a specified integration method.

    Parameters
    ----------
    t : float
        Current time.
    c_old : array-like
        Current value of the dependent variable(s) at time t.
    dt : float
        Time step size.
    f : callable
        Function representing the ODE system. Should have signature `f(t, c, *args, **kwargs)` and return the derivative dc/dt.
    method : str
        Integration method to use. Options are:
        - 'Euler' : Forward Euler method
        - 'RK2'   : Second-order Runge-Kutta (midpoint) method
        - 'RK4'   : Classical fourth-order Runge-Kutta method
    *args : 
        Additional positional arguments passed to `f`.
    **kwargs : 
        Additional keyword arguments passed to `f`.

    Returns
    -------
    numpy.ndarray
        Estimated increment `c_new - c_old` for this time step.

    Raises
    ------
    ValueError
        If `method` is not one of 'Euler', 'RK2', or 'RK4'.

    """
    if method == 'Euler':
        return dt*f(t, c_old, *args, **kwargs)
    elif method == 'RK2':
        k1 = np.array(dt*f(t, c_old, *args, **kwargs))
        return dt*f(t+dt*0.5, c_old + 0.5*k1, *args, **kwargs)
    elif method == 'RK4':
        k1 = np.array(f(t, c_old, *args, **kwargs))
        k2 = np.array(f(t + dt * 0.5, c_old + 0.5 * dt * k1, *args, **kwargs))
        k3 = np.array(f(t + dt * 0.5, c_old + 0.5 * dt * k2, *args, **kwargs))
        k4 = np.array(f(t + dt, c_old + dt * k3, *args, **kwargs))
        return (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    else:
        raise ValueError('Method ot implemented') 


def ode_solver_adaptiv(town, f, method, *args, **kwargs):
    """
    Adaptive ODE solver for a system of equations using Euler, RK2, or RK4 methods.

    This function integrates an ODE system defined by `f` over the time interval specified 
    in `town.T` using an adaptive time step to achieve a desired accuracy `town.eps`.

    Parameters
    ----------
    town : object
        An object containing the ODE initial conditions and parameters:
        - `town.c0` : array-like, initial state vector at t = town.T[0]
        - `town.T` : array-like, time points [t0, t1, ..., t_final]
        - `town.eps` : float, desired accuracy for adaptive step sizing
    f : callable
        Function representing the right-hand side of the ODE system. Should have signature
        `f(t, c, *args, **kwargs)` and return the derivatives dc/dt.
    method : str
        Integration method to use. Options are:
        - 'Euler' : Forward Euler method
        - 'RK2'   : Second-order Runge-Kutta (midpoint) method
        - 'RK4'   : Classical fourth-order Runge-Kutta method
    *args : 
        Additional positional arguments passed to `f`.
    **kwargs : 
        Additional keyword arguments passed to `f`.

    Returns
    -------
    t : numpy.ndarray
        Array of time points where the solution was computed.
    c : numpy.ndarray
        Array of solution values at each time point. Shape is (len(t), len(c0)).

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    """

    c = [town.c0]
    t = [town.T[0]]
    dt_old = 1e-1
    if method == 'Euler':
        p = 1
    elif method == 'RK2':
        p = 2
    elif method == 'RK4':
        p = 4
    else:
        
        assert ValueError('Method not implemented')
    while t[-1] < town.T[-1]:
        c_old = c[-1]
        eps_calc = 10*town.eps #just to enter while loop
        while eps_calc > town.eps:
            dt = dt_old
            c_long = c_old + step(t[-1], c_old, dt, f, method, *args, **kwargs)
            c_half = c_old + step(t[-1], c_old, 0.5*dt, f, method, *args, **kwargs)
            c_two_half = c_half + step(t[-1]+0.5*dt, c_half, 0.5*dt, f, method, *args, **kwargs)
            eps_calc = np.linalg.norm((c_long-c_two_half)/(2**p-1))
            eps_calc = np.ceil(eps_calc * 1e5) / 1e5
            dt_old = dt*(town.eps/eps_calc)**(1/(p+1))
        c.append(c_two_half)
        t.append(t[-1]+dt)
    return np.array(t), np.array(c)

def rhs_z(t,c,town):
    return np.array([town.beta*c[0]*(1-c[0]/town.S0), -town.beta*c[1]*(1-c[1]/town.S0)])

def Excercize_2_task_1(town):

    t, Z = ode_solver_adaptiv(town,  rhs_z,'RK2',town)
    S_d, Z_d = SZ_solution(town,t)


Excercize_2_task_1(Dirdal)
Excercize_2_task_1(Sokndal)

# -------------------------------
# Part 2: Step-size experiments for Dirdal (Euler, RK2, RK4)
# -------------------------------
dt_values = [0.05, 0.1, 0.5, 1.0, 2.0]
methods = ["euler", "rk2", "rk4"]

def rhs_SDZR(t, c, K, town):
    """
    Compute the derivatives for the SEZR (Susceptible-Exposed-Zombie-Removed) model.

    Parameters
    ----------
    t : float
        Current time (unused in this autonomous system, but required for ODE solver interface).
    c : array-like
        Current state vector [S, E, Z, R]:
        - S : Susceptible population
        - E : Exposed population
        - Z : Zombie population
        - R : Removed population
    K : array-like
        Model parameters [beta, sigma, alpha, w_t]:
        - beta : transmission rate (S → E)
        - sigma : progression rate from E → Z
        - alpha : removal rate
        - w_t : additional removal rate or control term
    town : object
        Town object containing additional model parameters:
        - S0 : total initial susceptible population

    Returns
    -------
    numpy.ndarray
        Derivatives [dS/dt, dE/dt, dZ/dt, dR/dt] at the current time.
    
    Notes
    -----
    The SEZR model used here includes interactions between Susceptible and Zombies
    and tracks the flow from Exposed to Zombies and Removed.
    """
    S, E, Z, R = c
    beta, sigma, alpha, w_t = K
    SZ = S * Z / town.S0

    d_S = -beta * SZ
    d_E = beta * SZ - sigma * E
    d_Z = sigma * E - (alpha + w_t) * SZ
    d_R = (alpha + w_t) * SZ

    return np.array([d_S, d_E, d_Z, d_R])


def estimate_lambda(hours):
    """
    Estimate a decay/removal rate (lambda) based on a given timescale.

    Parameters
    ----------
    hours : float
        Time duration in hours used to compute the decay rate.

    Returns
    -------
    float
        Estimated lambda value, assuming 60% of the population remains after `hours`.
    
    Notes
    -----
    The formula used is derived from the exponential decay relationship:
        fraction_remaining = exp(-lambda * hours)
    Solving for lambda gives:
        lambda = -ln(0.6) / hours
    """
    return - (1.0 / hours) * np.log(0.6)


def task_3(town):
    """
    Simulate the SEZR epidemic dynamics for a given town and plot results.

    Parameters
    ----------
    town : object
        Town object containing initial conditions and parameters:
        - k0 : initial state vector [S, E, Z, R]
        - beta, sigma, alpha, w_t : model parameters
        - S0 : total initial susceptible population
        - Name : string name of the town

    Returns
    -------
    None
        This function prints a lambda estimate and produces a matplotlib plot of the 
        populations over time: Susceptible, Exposed, Zombies, and Removed.

    Notes
    -----
    - The ODE system is solved using `ode_solver_adaptiv` with the RK4 method.
    - Time interval is fixed from 0 to 250 days.
    - Lambda is estimated based on the town's Name:
        - "Dirdal" → 48 hours
        - "Sokndal" → 72 hours
        - others → 1 hour
    - Populations are plotted on a single figure with labels and grid.
    """
    town.c0 = town.k0
    town.T = np.linspace(0,250)
    K = np.array([town.beta, town.sigma, town.alpha, town.w_t])
    N = town.S0
    t, Z = ode_solver_adaptiv(town, rhs_SDZR, 'RK4', K, town)  # pass parameters to RHS
    if town.Name == "Dirdal":
        hour = 48
    elif town.Name == ("Sokndal"):
        hour = 72
    else:
        hour = 1

    # print(f"Lambda estimate at {hour} for the town of {town.Name} is {estimate_lambda(hour)}")



task_3(Sokndal)
task_3(Dirdal)
#___ denne mp in i ode_solver_adaptiv ___ lingje 1 er der men må legge til den andre
#            eps_calc = np.linalg.norm((c_long-c_two_half)/(2**p-1))
#            eps_calc = np.ceil(eps_calc * 1e5) / 1e5





def rhs_SDZR_attacks(t,c,town):

    """
    Compute the derivatives for an SEZR (Susceptible-Exposed-Zombie-Removed) model 
    including scheduled zombie attacks.

    The model includes an additional attack term `w_t` that increases the removal rate
    based on Gaussian-shaped attack events at specified times.

    Parameters
    ----------
    t : float
        Current time.
    c : array-like
        Current state vector [S, E, Z, R]:
        - S : Susceptible population
        - E : Exposed population
        - Z : Zombie population
        - R : Removed population
    town : object
        Town object containing model parameters and attack schedule:
        - beta : float, transmission rate
        - sigma : float, progression rate from E → Z
        - alpha : float, removal rate
        - S0 : float, total initial susceptible population
        - a : float, amplitude factor for attacks
        - attacks : list of floats, times at which attacks occur

    Returns
    -------
    numpy.ndarray
        Array of derivatives [dS/dt, dE/dt, dZ/dt, dR/dt] at the current time.

    Notes
    -----
    - Attack events are modeled as Gaussian pulses added to the removal term:
        w_t = a * sum(exp(-0.5 * (t - ti)**2) for ti in attacks)
    - If Z < 1, E < 1, and t > 100, the transmission and progression rates
      (beta and sigma) are set to 0 to simulate the end of the outbreak.
    - Supports vector-valued populations (S, E, Z, R) and handles scheduled attacks.
    """


    S, E, Z, R = c
    beta, sigma, alfa, N, a = town.beta, town.sigma, town.alpha,town.S0,town.a
    w_t = a*np.sum([np.exp(-.5*(t-ti)**2) for ti in town.attacks])

    SZ = S*Z/N

    if Z<1 and E<1 and t >100:
        beta = 0
        sigma = 0
    d_S = -(beta*SZ)
    d_E = beta*SZ-sigma*E
    d_Z = sigma*E-(alfa+w_t)*SZ
    d_R = (alfa+w_t)*SZ


    return np.array([d_S,d_E,d_Z,d_R])



def Counter_Attacks(town):
    """
    Simulate and plot the progression of an SDZR (Susceptible–Exposed–Zombie–Removed)
    model for a given town, including the effects of human counterattacks.

    Parameters
    ----------
    town : object
        A town object containing the following attributes:
        - Name (str): Name of the town.
        - beta (float): Infection rate (zombie-to-human transmission).
        - sigma (float): Exposure rate.
        - alpha (float): Removal rate (zombies being destroyed).
        - w_t (float): Attack strength or defense parameter.
        - k0 (float): Initial condition for c0.
        - Other model-specific attributes required by `rhs_SDZR_attacks` and `ode_solver_adaptiv`.

    Effects
    -------
    - Solves the SDZR system using adaptive Runge–Kutta (RK4) method.
    - Plots the number of Exposed, Zombies, and Removed individuals over time.
    - Prints model parameters and the final number of humans left.
    """
    town.c0 = town.k0#Defines the correct standardarray for the solver in this task
    town.eps = 0.001#Stepsize
    K = np.array([town.beta, town.sigma, town.alpha, town.w_t])
    t,Z = ode_solver_adaptiv(town,  rhs_SDZR_attacks,'RK4',town)#Uses Runge-Kutta, sends it to be solved using adaptive solver
    # print(f"in {town.Name} Beta is {town.beta} and alpha is {town.alpha}")
Counter_Attacks(Dirdal)#Runs with initial values for dirdal
Counter_Attacks(Sokndal)#And Sokndal

def Counter_Attacks_Compared(town):
    """
    Compare the SDZR model with and without human attacks for a given town.

    Parameters
    ----------
    town : object
        A town object containing model parameters as described in the function `Counter_Attacks`.

    Effects
    -------
    - Solves and plots the zombie population over time for two cases:
        1. Without human counterattacks (`rhs_SDZR`)
        2. With human counterattacks (`rhs_SDZR_attacks`)
    - Prints the final number of humans left in both scenarios.
    """
    town.c0 = town.k0
    town.eps = 0.001
    town.T=np.linspace(0, 250)

    K = np.array([town.beta, town.sigma, town.alpha, town.w_t])
    t_2,Z_2 = ode_solver_adaptiv(town,  rhs_SDZR_attacks,'RK4', town)#Solves for both attacks and no attack
    t_1,Z_1= ode_solver_adaptiv(town, rhs_SDZR, 'RK4', K, town)  # pass parameters to RHS
    # print("Humans left, humans dont attack attacks", np.floor(Z_1[-1,0]))
    # print("Humans left, humans attack", np.floor(Z_2[-1,0]))

Counter_Attacks_Compared(Sokndal)
Counter_Attacks_Compared(Dirdal)

def read_file(file_path):
    x = [0]
    y_total = [0]
    y = [0]
    l = 0
    with open(file_path, "r") as f:
        next(f) #hopper over første lingje
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                l = l + 1
                days = int(parts[1])
                cases = int(parts[2])
                x.append(days)
                y_total.append(y_total[l-1]+cases)
                y.append(cases)
    return np.array([x,y_total,y])

def beta_f(t,beta,landa):

    """
    Zombie attack function
    Input
    -----
    t: float, time
    beta: float, array of Konstant beta0
    landa: float, array of Konstant : landa
    Output
    ------
    c : array, fusjen of beta_f(t)
    """

    return beta*np.exp(-1*landa*t)

def rhs_SDZR_EBOLA(t,c, K):

    """
    Zombie attack function
    Input
    -----
    t: float, time
    c: float, array of S, E, Z, R
    K: float, array of Konstant : beta, sigma, gamma, N,landa
    Output
    ------
    c : array, array of d_S,d_E,d_Z,d_R
    """
    

    S, E, Z, R = c
    beta_0,sigma, gamma, N ,landa= K
    beta = beta_f(t,beta_0,landa)
    SZ = S*Z/N

    d_S = -(beta*SZ)
    d_E = beta*SZ-sigma*E
    d_Z = sigma*E-gamma*Z
    d_R = gamma*Z


    return np.array([d_S,d_E,d_Z,d_R])

def Task_Ebola(town,beta=0.255,sigma=1/9.7,gamma=1/7, landa = 0.002):
    c0= np.array([town.S0,town.E0,town.Z0,town.R0])
    town.c0 = c0

    file_path = "data/ebola_cases_guinea.dat"
    x_f,_,y_f = read_file(file_path)
    t,Z = ode_solver_adaptiv(town, rhs_SDZR_EBOLA,'RK4', np.array([town.beta,town.sigma,town.gamma,town.S0,town.landa]))
    if __name__ == "__main__":
        plt.plot(t,Z[:,2],'-*', label="modelering")
        plt.plot(x_f,y_f,'*', label="fusjon")

        plt.legend()
        plt.grid()
        plt.show()

guinea = Town("guinea", 10e7,beta=0.255,sigma=1/9.7,gamma=1/7,landa=0.002, eps=0.001, Timesteps=np.linspace(0, 700))
Task_Ebola(guinea)
