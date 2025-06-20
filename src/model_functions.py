"""
Function definitions to calculate space-time yield factors and bioreactor
purchase cost factors for seed train bioprocesses made up of sequences of
black-box bioreactors. 

- Cell growth is modeled as exponential with a constant doubling time.
- Exponential growth doublings per bioreactor are constant across a seed train's 
expansion steps and each expansion step starts with the same initial cell density 
and ends with the same final cell density, which is why both bioreactor volume and 
biomass increase by the same factor each step.
- The same turnaround time is required for each expansion step in a seed train.
- 8760 operating hours are assumed per year.
- Bioreactor costs are governed by a power function and the same relationship applies
to all bioreactors in a seed train.
- Fractional values of expansion to seed train volume ratios and expansion to 
maximum individual bioreactor volumes are allowed.
"""

# %% Imports
import numpy as np
import math

# %% Cell expansion in seed train of black-box bioreactors

def STY_f(ind=2, **kwargs):
    """
    Calculate space-time yield factors.

    Function to calculate the space-time yield factor based on mass, volume, 
    and time calculated from the 'expansion()' function. The units of 
    space-time yield factors are per year (assuming 8760 operating hours in a 
    year).

    Parameters
    ----------
    ind : int, default 2
        Determines which index to use, allowing for the selection between:
            - 0: calculates factor for only Nth expansion step
            - 1: calculates factor for seed train excluding Nth step 
            - 2: calculates factor for entire seed train, including Nth step.
    **kwargs : 
        These parameters will be passed to the 'expansion()' function and 
        define the bioprocess or range of bioprocesses modeled.

    Returns
    -------
    numpy.ndarray
        Array of space-time yield factor(s) for the bioprocess(es) modeled.
        Array shape is (len(doubs_per_reactor), len(doubling_t)) where both
        doubs_per_reactor and doubling_t are 'expansion()' keyword arguments.
    """
    # Obtain mass, volume, and time from the 'expansion()' function
    mass, volume, time = expansion(**kwargs)

    # Index based on ind value
    index = [slice(-1, None), slice(None, -1), slice(None)][ind]

    # Compute the space-time yield factor using the mass, time, and volume
    # Assumes 8760 operating hours per year
    return 8760 * sum(mass[index])/(time*np.sum(volume[index], axis=0))


def vol_percent(**kwargs):
    """
    Calculate the final bioreactor volume percentage of total seed train.

    Function to calculate the final expansion step bioreactor volume as a
    percentage of the total seed train volume. 

    Parameters
    ----------
    **kwargs : 
        These parameters will be passed to the 'expansion()' function and 
        define the bioprocess or range of bioprocesses modeled.

    Returns
    -------
    numpy.ndarray
        Array of volume percentages for the bioprocess(es) modeled. 
        Array shape is (len(doubs_per_reactor), len(doubling_t)) where both
        doubs_per_reactor and doubling_t are 'expansion()' keyword arguments.
    """
    # Obtain volume from the 'expansion()' function
    volume = expansion(**kwargs)[1]

    # Return the final expansion bioreactor percentage of seed train
    return 100 * volume[-1]/np.sum(volume, axis=0)


def cost_vol(R=0.6, S=1.0, doubs_per_reactor=np.linspace(1, 9, 5), **kwargs):
    """
    Calculate bioreactor purchase cost factors.

    Function to calculate the seed train bioreactor purchase cost factor based 
    on volume calculated from the 'expansion()' function and accounting for
    scalability constraints and economies of scale. Bioreactor purchase cost 
    factors are unitless. 

    Parameters
    ----------
    R : float, default 0.6
        Equipment purchase cost and capacity power law relationship exponent. 
    S : float, default 1.0
        Scalability constraint or the ratio of the final expansion step
        volume required to the maximum individual bioreactor volume.
    doubs_per_reactor : numpy.ndarray, default np.linspace(1, 9, 5)
        Array of possible exponential growth doublings per expansion step. This
        array will also be passed to the 'expansion()' function. 
    **kwargs : 
        These parameters will be passed to the 'expansion()' function and 
        define the bioprocess or range of bioprocesses modeled.

    Returns
    -------
    numpy.ndarray
        Array of bioreactor purchase cost factor(s) for the bioprocess(es) 
        modeled. Array shape is (len(doubs_per_reactor), len(doubling_t))
        where both doubs_per_reactor and doubling_t are 'expansion()' keyword 
        arguments.
    """
    # Update kwargs with the 'doubs_per_reactor' array
    kwargs = kwargs|{'doubs_per_reactor': doubs_per_reactor}

    # Obtain volume from the 'expansion()' function
    # volume is array with shape: (N, len(doubs_per_reactor), len(doubling_t))
    volume = expansion(**kwargs)[1]

    # Number of steps (if any and excluding Nth) requiring scale-out
    step = np.minimum(np.floor(math.log2(S) / doubs_per_reactor),
                      volume.shape[0]-1).astype(int)

    return (volume[-1] + np.array([[np.sum(volume[:j, k, 0])
                                    + np.sum(S**(R-1) * volume[j:-1, k, 0]**R)]
                                   for k, j in enumerate(step)])
            ) / np.sum(volume, 0)


def expansion(N=5, 
              max_util=True,
              doubs_per_reactor = np.linspace(1, 9, 5),
              doubling_t = np.linspace(10, 30, 5), 
              draw_fill_count=0,  
              draw_fill_frac=0.5, 
              days_ss = 0.0,
              turnaround = 25.0,
              ):
    """
    Calculate mass produced, bioreactor volume, and process cycle time.

    Function to model cell expansion in a seed train, calculating mass produced, 
    bioreactor volume, and process cycle time. Function parameters define the 
    bioprocess or range of bioprocesses modeled.

    Parameters
    ----------
    N : int, default 5
        Number of discrete sequential, batch expansion steps in seed train.
    max_util : bool, default True
        If True, maximize reactor utilization.
    doubs_per_reactor : numpy.ndarray, default np.linspace(1, 9, 5)
        Array of possible exponential growth doublings per expansion step.
    doubling_t : numpy.ndarray, default is np.linspace(10, 30, 5)
        Array of possible cell doubling times [h].
    draw_fill_count : int, default 0
        Number of draw-and-fill harvests.
    draw_fill_frac : float, default 0.5
        Draw-and-fill draw down fraction.
    days_ss : float, default 0.0
        Length of operation during which there is continuous cell harvest [d].
    turnaround : float, default 25.0
        Total bioreactor turnaround time [h].

    Returns
    -------
    mass : list of numpy.ndarray 
        First array in list is mass produced in seed train excluding the Nth 
        step and the second array is mass produced in only the Nth step. Array 
        shapes are (len(doubs_per_reactor), len(doubling_t)).
    volume : numpy.ndarray
        Bioreactor volumes required for each discrete sequential expansion step. 
        Array shape is (N, len(doubs_per_reactor), len(doubling_t))
    bottleneck : numpy.ndarray
        Process cycle time based on scheduling bottleneck and whether or not 
        utilization is maximized. Array shape is (len(doubs_per_reactor), 
        len(doubling_t)).
    """
    # Create meshgrid for doubling times and doublings per reactor
    # shape (exponential doublings, doubling time)
    td, d = np.meshgrid(doubling_t.astype(np.float64),
                        doubs_per_reactor.astype(np.float64))

    # mass relative to X_N
    mass = [1/(2**d) - 1/(2**(N*d)),
            1 - 1/(2**d) + (days_ss*24*math.log(2)/td
                            + draw_fill_frac*draw_fill_count)]

    # Calculate bottleneck time and Nth step bioreactor to rest of seed train
    # ratio based on utilization mode
    if max_util:
        bottleneck = td*d + turnaround
        expansion_2_seedtrain_ratio = (d*td + days_ss*24 + turnaround
                                       + td*math.log(1/(1-draw_fill_frac), 2)
                                       *draw_fill_count) / bottleneck
    else:
        bottleneck = (d*td + days_ss*24 + turnaround
                      + td*math.log(1/(1-draw_fill_frac), 2)*draw_fill_count)
        expansion_2_seedtrain_ratio = np.ones(d.shape)

    # Calculate seed train volume relative to V_N
    volume = np.array([(1.0/(2.0**d))**x for x in range(1,N)]
                      + [expansion_2_seedtrain_ratio])

    return mass, volume, bottleneck
