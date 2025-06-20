"""
Cultured Meat Expansion Phase Modeling - Interactive Streamlit Application

This application provides an interactive interface for modeling and analyzing 
bioprocess designs for cultured meat production, comparing batch, semi-continuous,
and continuous operations with customizable parameters.
"""
import time
start = time.time()

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Import custom functions
import sys
sys.path.append('src')
from model_functions import STY_f, cost_vol, expansion
from plotting_utils import bar_plots, contour_plots, waterfall_plots

# =============================================================================
# Application Configuration
# =============================================================================

st.set_page_config(
    page_title="Cultured Meat Expansion Phase Modeling",
    layout="wide"
)

# =============================================================================
# Helper Functions
# =============================================================================


def get_factors(dicts):
    sty = STY_f(N=dicts['N'],
                doubs_per_reactor=np.array([dicts['d']]),
                doubling_t=np.array([dicts['t_d']]),
                draw_fill_count=dicts['i'],
                draw_fill_frac=dicts['f'],
                turnaround=dicts['t_{TAR}'],
                days_ss=dicts['t_{CH}'],
                max_util=max_util).flatten()
    cost = cost_vol(R=dicts['R'],
                    S=dicts['S'],
                    N=dicts['N'],
                    doubs_per_reactor=np.array([dicts['d']]),
                    doubling_t=np.array([dicts['t_d']]),
                    draw_fill_count=dicts['i'],
                    draw_fill_frac=dicts['f'],
                    turnaround=dicts['t_{TAR}'],
                    days_ss=dicts['t_{CH}'],
                    max_util=max_util).flatten()
    return sty[0], cost[0]


def waterfall_data(dicts, ratio):
    C_min = (batch_params['V_{max}'] / dicts['V_{max}']) ** (batch_params['R'] - 1)
    X_max = batch_params['X_{max}'] / dicts['X_{max}']
    Ctot_C = batch_params['C_{tot}/C'] / dicts['C_{tot}/C']
    waterfall_data = [ratio-1,
                      ratio*C_min-1,
                      ratio*C_min/X_max-1, 
                      ratio*C_min/X_max*Ctot_C-1
                      ]
    return waterfall_data


def calc_summary(dicts):
    mass, volume, time = expansion(
        N=dicts['N'],
        doubs_per_reactor=np.array([dicts['d']]),
        doubling_t=np.array([dicts['t_d']]),
        draw_fill_count=dicts['i'],
        draw_fill_frac=dicts['f'],
        turnaround=dicts['t_{TAR}'],
        days_ss=dicts['t_{CH}'],
        max_util=max_util
    )

    biomass = [1/(2**dicts['d'])**(dicts['N']-x) for x in range(dicts['N']+1)]
    biomass[-1] += (dicts['t_{CH}'] * 24 * np.log(2) / dicts['t_d'] +
                    dicts['f'] * dicts['i']) 
    volume[:-1] = volume[:-1][::-1]

    cost = np.array([(dicts['S']/(2**(dicts['d']*(dicts['N']-i))))**(dicts['R']-1) 
                     if dicts['S']<=2**(dicts['d']*(dicts['N']-i)) else 1 
                     for i in range(1, dicts['N']+1)])
    
    bioreactors = [rf"$1$ X ${x:.4f} \cdot V_N$" if y>1 
                   else f"${dicts['S']/(2**(dicts['d']*(dicts['N']-(i+1)))):.1f}$ X $V_{{max}}$ in parallel" 
                   if dicts['S']>1 else "$1$ X $V_N$" 
                   for i, (x, y) in enumerate(zip(volume.flatten(), cost))]
    
    if volume[-1] > 1:
        bioreactors[-1] = f"${volume.flatten()[-1]:.4f}$ staggered sets of {bioreactors[-1]}"

    df = pd.DataFrame(
        {'Step': [i+1 for i in range(dicts['N'])],
         "Biomass In/$X_N$": biomass[:-1],
         "Biomass Out/$X_N$": biomass[1:],
         "Required Volume/$V_N$": volume.flatten(),
         "Volumetric Purchase Cost/$C_{min}$": cost,
         "Individual Bioreactors": bioreactors}
    )

    df = df.set_index('Step')

    return df


# =============================================================================
# Application Header and Description
# =============================================================================
st.title("Cultured Meat Expansion Phase Modeling")

with st.expander("Model Setup", expanded=True):
    st.markdown("The cultured meat seed train process can be modeled as a sequence of black-box bioreactors. That is, each bioreactor can be considered a 'black box,' converting inputs to outputs without specifying or requiring information on system internals. This can help provide a generalizable framework for evaluating process designs by distilling various biological and engineering parameters and assumptions to their net effect on system inputs and outputs. Figure 1 provides a schematic of a seed train process made up of a sequence of black-box bioreactors. The seed train is comprised of $N$ discrete, sequential batch expansion steps, each with $d$ exponential growth doublings. Each step is defined in terms of the volume, $$V_N$$, and ending biomass, $X_N$, of the last or $N^{th}$-step bioreactor, BR-N. For example, the input to and volume required for the first expansion step are given in terms of $X_N$ and $V_N$ respectively and depend on $N$ and $d$. For each bioreactor, its volume is given by the expression beneath its label. The equation(s) below each bioreactors' labels are the occupancy time for each bioreactor. Each step is operated in batch mode, with only the last $N^{th}$-step operating in either batch, semi-continuous, or continuous mode. The occupancy time and the biomass leaving the $N^{th}$-step bioreactor thus depend on operation mode.")
    # Load the image only once using session state
    if 'nomenclature_image' not in st.session_state:
        st.session_state.nomenclature_image = Image.open('images/nomenclature.png')
    
    st.image(st.session_state.nomenclature_image, 
             caption="Figure 1: Schematic of a seed train process made up of a sequence of black-box bioreactors.", 
             use_container_width=True)

# =============================================================================
# Sidebar Configuration and Parameter Input
# =============================================================================

st.sidebar.header("Specify model parameters to see the effects on space-time yield and capital-investment dependent costs")
max_util = st.sidebar.checkbox("Maximize Reactor Utilization", value=True)

# Initialize parameter dictionaries
batch_params = {}
semi_params = {}
continuous_params = {}

# Model parameters section
col1, col2, col3, col4 = st.sidebar.columns([0.6, 1.5, 1.5, 1.5])

for col, label in zip([col1, col2, col3, col4], ["", "Batch", "Semi-continuous", "Continuous"]):
    col.markdown(f"<div style='display:flex; justify-content:center;'> <h3>{label}</h3> </div>", unsafe_allow_html=True)

# Parameter definitions
parameters = [
    ['N', 'Number of discrete sequential, batch expansion steps in seed train', [1, 10, (6, 6, 3), 1]],
    ['d', 'Number of exponential growth doublings per expansion step', [1.0, 9.0, (1.9, 1.9, 2.7), 0.1]],
    ['t_d', 'Cell doubling time [hours]', [10.0, 50.0, (24.0, 24.0, 24.0), 0.1]],
    ['t_{TAR}', 'Total reactor turnaround time [hours]', [0.0, 100.0, (10.0, 10.0, 10.0), 1.0]],
    ['i', 'Number of draw-and-fill harvests', [0, 50, (0, 3, 0), 1]],
    ['f', 'Draw-and-fill drawdown fraction', [0.0, 0.9, (0.0, 0.5, 0.0), 0.1]],
    ['t_{CH}', 'Length of operation during which there is continuous harvest [days]', [0.0, 100.0, (0.0, 0.0, 33.0), 1.0]],
    ['R', 'Equipment purchase cost and capacity power law relationship exponent', [0.1, 1.0, (0.69), 0.1]],
    ['S', 'Ratio of Nth-step required reactor volume to seed train largest individual reactor volume', [1.0, 100.0, (6.0, 6.0, 1.0), 0.5]],
    ['V_{max}', 'Seed train largest individual reactor volume [L]', [1, 10**6, (2*10**4, 2*10**4, 2*10**3), 100]],
    ['X_{max}', 'Maximum supported biomass concentration [g/L]', [1.0, 1000.0, (92.0, 92.0, 156.0), 1.0]],
    ['C_{tot}/C', 'Ratio of total reactor purchase cost associated with a seed train process to the purchase cost per volume of the largest individual reactor in the seed train', [1.0, 100.0, (3.3, 3.3, 11.7), 0.1]]
    ]

# Parameter input UI
for key, desc, values in parameters:
    if key != 'R':
        col1, col2, col3, col4 = st.sidebar.columns([0.6, 1.5, 1.5, 1.5], gap="small")
        min_val, max_val, value, step = values
        
        with col1:
            st.latex(key, help=desc)
            st.markdown("<div style='margin-bottom: 1px;'></div>", unsafe_allow_html=True)
        
        with col2:
            if f"batch_{key}" not in st.session_state:
                st.session_state[f"batch_{key}"] = value[0]
                
            batch_params[key] = st.number_input(
                f"batch_{key}", 
                min_value=min_val, 
                max_value=max_val, 
                step=step, 
                disabled=key in ['t_{CH}', 'i', 'f'], 
                key=f"batch_{key}", 
                label_visibility="collapsed"
            )
            
        with col3:
            disable = key in ['t_{CH}']
            if f"semi_{key}" not in st.session_state:
                st.session_state[f"semi_{key}"] = value[1]

            semi_params[key] = st.number_input(
                f"semi_{key}", 
                min_value=min_val, 
                max_value=max_val, 
                step=step, 
                disabled=disable, 
                key=f"semi_{key}", 
                label_visibility="collapsed"
            )
            
        with col4:
            disable = key in ['i', 'f']
            if f"cont_{key}" not in st.session_state:
                st.session_state[f"cont_{key}"] = value[2]
        
            continuous_params[key] = st.number_input(
                f"cont_{key}", 
                min_value=min_val, 
                max_value=max_val, 
                step=step, 
                disabled=disable, 
                key=f"cont_{key}", 
                label_visibility="collapsed"
            )
            
    else:
        col1, col2 = st.sidebar.columns([0.55, 4.5], gap="small")
        min_val, max_val, value, step = values
        
        with col1:
            st.latex(key, help=desc)
            st.markdown("<div style='margin-bottom: 1px;'></div>", unsafe_allow_html=True)
        
        with col2:
            if key not in st.session_state:
                st.session_state[key] = value
                
            batch_params[key] = st.number_input(
                key, 
                min_value=min_val, 
                max_value=max_val, 
                step=step, 
                disabled=False, 
                key=key, 
                label_visibility="collapsed"
            )
            semi_params[key] = batch_params[key]
            continuous_params[key] = batch_params[key]

# =============================================================================
# Calculate Results
# =============================================================================

batch_sty_factor, batch_cost_factor = get_factors(batch_params)
semi_sty_factor, semi_cost_factor = get_factors(semi_params)
continuous_sty_factor, continuous_cost_factor = get_factors(continuous_params)
cont_ratio = (batch_cost_factor/batch_sty_factor)/(continuous_cost_factor/continuous_sty_factor)
semi_ratio = (batch_cost_factor/batch_sty_factor)/(semi_cost_factor/semi_sty_factor)

# Waterfall chart values
semi_waterfall_data = waterfall_data(semi_params, semi_ratio)
continuous_waterfall_data = waterfall_data(continuous_params, cont_ratio)

# =============================================================================
# Display Results in Sidebar
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.header("Factors Results")

# Create metrics
col1, col2, col3 = st.sidebar.columns(3)
for col, labels, factors in zip([col1, col2, col3], ['Batch', 'Semi-continuous', 'Continuous'],
                                [[batch_sty_factor, batch_cost_factor], 
                                 [semi_sty_factor, semi_cost_factor], 
                                 [continuous_sty_factor, continuous_cost_factor]]):
    with col:
        st.subheader(labels)
        st.metric("$STY_f$", f"{factors[0]:.1f}")
        st.metric("$C_f$", f"{factors[1]:.3f}")

st.sidebar.markdown(f"Ratio of Batch to Semi-continuous $C_f/STY_f$ = {semi_ratio:.2f}")
st.sidebar.markdown(f"Ratio of Batch to Continuous $C_f/STY_f$ = {cont_ratio:.2f}")

# =============================================================================
# Summary of Processes
# =============================================================================

with st.expander("Your Processes", expanded=True):
    st.markdown("These tables provide overviews of your processes, summarizing the input and output biomass of each expansion step (relative to $X_N$), the required reactor volume of each step (relative to $V_N$), the reactor purchase cost per volume of each step (relative to $C_{{min}}$, the purchase cost per volume of the largest individual reactor in the seed train), and the individual reactors required for each step (whether parallel reactors required due to scalability constraints and/or staggered reactors required to maximize utilization.)")
    for labels, dicts in zip(['batch', 'semi-continuous', 'continuous'],
                             [batch_params, semi_params, continuous_params]):
        st.subheader(f"Your {labels.capitalize()} Process")
        st.table(calc_summary(dicts).style.format({
            "Biomass In/$X_N$": "{:.4f}",
            "Biomass Out/$X_N$": "{:.4f}",
            "Required Volume/$V_N$": "{:.4f}",
            "Volumetric Purchase Cost/$C_{min}$": "{:.3f}"
        }))

# =============================================================================
# Main Content Area - Visualization
# =============================================================================

st.header("Compare Operating Modes")
st.subheader("Space-Time Yield Factor")
st.markdown("Space-time yield ($STY$) describes the overall productivity of a process and is given by")
spacer, eq_column = st.columns([0.1, 0.9])
with eq_column:
    st.markdown(r"$$STY = \frac{X}{V \cdot t}$$")
st.markdown("where $X$ is biomass produced each process cycle, $V$ is total seed train bioreactor volume (including any staggered bioreactors), and $t$ is process cycle time. In a seed train process, a cycle begins with any necessary pre-cell culture steps for the first bioreactor and ends with the complete harvest and any post-cell culture steps from the final bioreactor. Space-time yield factor, $STY_f$, is the seed train process space-time yield relative to the maximum supported biomass concentration, $X_{max}$, and is defined here to facilitate the evaluation of process intensification approaches independent of the maximum supported biomass concentration. Increasing $X_{max}$ is clearly an important lever for increasing $STY$, but $STY_f$ is used here to focus on the other design and operating parameters that affect equipment utilization. $STY_f$ is given by")
spacer, eq_column = st.columns([0.1, 0.9])
with eq_column:
    st.markdown(r"$$STY_f = \frac{STY}{X_{max}}$$")
st.markdown("and $X_{max}$ is on a total reactor volume basis (as opposed to working volume) and is given by")
spacer, eq_column = st.columns([0.1, 0.9])
with eq_column:
    st.markdown(r"$$X_{max} = \frac{X_N}{V_N}$$")

st.markdown("The $STY_f$ for your batch, semi-continuous, and continuous processes are shown below, along with the $STY_f$ for a range of process designs.")

if "fig1" not in st.session_state:
    # Space-Time Yield Factor Plots 
    sub_plt = np.linspace(10, 30, 5)
    x = np.linspace(9, 1, 5)

    datasets = [['bar', ({'days_ss': 0},), {'fc':'gray', 'ec':None, 'label':'$STY_f$ for batch cell harvest'}],
                ['fill', ({'days_ss': 20, 'max_util': True}, {'days_ss': 60, 'max_util': True}),
                    {'alpha':0.2, 'fc':'b', 'ec':None, 'label':'$STY_f$ range for 20-60d continuous cell harvest and max utilization'}],
                    ['fill', ({'days_ss': 20, 'max_util': False}, {'days_ss': 60, 'max_util': False}),
                    {'alpha':0.2, 'fc':'none', 'ec':'red', 'label':'$STY_f$ range for 20-60d continuous cell harvest and min volume'}],
                    ['scatter', ({'days_ss': 40, 'max_util': True},), 
                    {'marker':'.', 'color':'blue', 's':3, 'label':'$STY_f$ for 40d continuous cell harvest and max utilization'}],
                    ['scatter', ({'days_ss': 40, 'max_util': False},), 
                    {'marker':'.', 'color':'red', 's':3, 'label':'$STY_f$ for 40d continuous cell harvest and min volume'}]]

    data = []
    for td in sub_plt:
        sub_set = []
        for dataset in datasets:

            style, params, kwargs = dataset
            z_vals = []
            
            for param_set in params:
                z_vals.append(np.array([STY_f(turnaround=tar, doubling_t=np.array([td]), doubs_per_reactor=x,
                                                N=5, **param_set).ravel() for tar in [0, 25, 50, 75]]).ravel())
                
            sub_set.append([style, z_vals, kwargs])
        

        for sty, l, style in zip([batch_sty_factor, semi_sty_factor, continuous_sty_factor], 
                                ['batch', 'semi-continuous', 'continuous'],
                                ['solid', 'dashed', 'dotted']):
            sub_set.append(['line', [[-0.86, 3.14], [sty, sty]], {'color':'black', 'linestyle':style,
                                                                'label':f'Your {l} $STY_f$'}])
            
        data.append(sub_set)

    fig1, ax1 = bar_plots(5, 1, sub_plt, x, data, (0, 700, 50), 'Space-time Yield Factor [$y^{-1}$]')
    st.session_state.fig1 = fig1
    st.session_state.ax1 = ax1

else:
    updates = {'Your batch $STY_f$': batch_sty_factor, 
               'Your semi-continuous $STY_f$': semi_sty_factor, 
               'Your continuous $STY_f$': continuous_sty_factor}
    for ax in st.session_state.ax1:
        for line in ax.lines:
            line.set_ydata(2*[updates[line.get_label()]])

st.pyplot(st.session_state.fig1)
st.caption(r"The space-time yield factors, $STY_f$ [$y^{-1}$], of your processes are shown, along with the $STY_f$ of continuous versus batch cell harvest from a range of process designs. With $N=5$, a range of doubling times (10-30h), of doublings per batch expansion step (1-9), and of turnaround times (0-75h) are shown. $STY_f$ when $t_{CH}=40d$ and the $STY_f$ range when $20d \leq t_{CH} \leq 60d$ are both shown for no staggered bioreactors (minimizes volume) and with staggered bioreactors to maximize utilization.")


st.subheader("Bioreactor Purchase Cost Factor")
st.markdown("Mirroring the definition of $STY_f$, a cost factor, $C_f$ is defined here to isolate the impact of equipment utilization from that of the other two key levers affecting purchase cost per volume: scalability and bioreactor-specific purchase cost. That is, $C_f$ is the seed train purchase cost per volume relative to the lowest individual bioreactor purchase cost per volume in the seed train (the purchase cost per volume of the largest individual bioreactor), $C_{min}$. $C_f$ is given by")
spacer, eq_column = st.columns([0.1, 0.9])
with eq_column:
    st.markdown(r"$$C_f = \frac{C/V}{C_{min}}$$")
st.markdown("where $C$ is the total reactor purchase cost associated with a seed train process and $C_{min}$ is the purchase cost per volume of the largest individual reactor in the seed train. The ratio of your batch to continuous and of your batch to semi-continuous $C_f/STY_f$ are shown below, along with the ratios for a range of process designs.")

if "fig2" not in st.session_state:
    # Batch to continuous ratio bar plot
    sub_plt = np.linspace(10, 30, 5)
    x = np.linspace(9, 1, 5)

    data = []
    cmin = 1
    cmax = np.tile((1-(1/2**x))/(1-(1/2**(x*0.6))), 4)

    datasets = [['fill', ([cmin, 20], [cmax, 60]), {'fc':'c', 'ec':None, 'alpha':0.8, 'label':'Range of Batch to Continuous $C_f/STY_f$ ratios'}],
                ['scatter', ([cmin, 20],), {'marker':'^', 's':5, 'ec':'black', 'fc':'none', 'lw':0.5, 'label':'Batch (min) : 20d continuous harvest'}],
                ['scatter', ([cmin, 60],), {'marker':'^', 's':5, 'ec':'black', 'fc':'black', 'lw':0.5, 'label':'Batch (min) : 60d continuous harvest'}],
                ['scatter', ([cmax, 20],), {'marker':'o', 's':5, 'ec':'black', 'fc':'none', 'lw':0.5, 'label':'Batch (max) : 20d continuous harvest'}],
                ['scatter', ([cmax, 60],), {'marker':'o', 's':5, 'ec':'black', 'fc':'black', 'lw':0.5, 'label':'Batch (max) : 60d continuous harvest'}]
                ]

    for td in sub_plt:
        sub_set = []

        sty = np.array([STY_f(doubs_per_reactor=x, doubling_t=np.array([td]),
                                turnaround=tar) for tar in [0, 25, 50, 75]]).ravel()

        for dataset in datasets:

            style, params, kwargs = dataset
            z_vals = []
            
            for c, days in params:
                c_f = np.array([cost_vol(R=0.6, doubs_per_reactor=x, doubling_t=np.array([td]), turnaround=tar, N=5,
                                        days_ss=days, max_util=True) for tar in [0, 25, 50, 75]]).ravel()
                sty_c = np.array([STY_f(doubs_per_reactor=x, doubling_t=np.array([td]), turnaround=tar, N=5,
                                        days_ss=days, max_util=True) for tar in [0, 25, 50, 75]]).ravel()
                z_vals.append((c/sty)/(c_f/sty_c))
                
            sub_set.append([style, z_vals, kwargs])

        for v, l, label in zip([0.1, 0.01, 0.001], ['dotted', 'dashed', 'solid'], 
                                ['Fold-increase in $C_{min}$ (↓ $V_{max}$ 1 order of magnitude)', 
                                'Fold-increase in $C_{min}$ (↓ $V_{max}$ 2 orders of magnitude)', 
                                'Fold-increase in $C_{min}$ (↓ $V_{max}$ 3 orders of magnitude)']):
            sub_set.append(['line', [[-0.86, 3.14], 2*[(v**0.6)/v]], {'lw': 0.75, 'color':'red', 'linestyle':l, 'label':label}])
        
        for ratio, linestyle, label in zip([semi_ratio, cont_ratio], ['dashed', 'dotted'], ['Semi-continuous', 'Continuous']):
            sub_set.append(['line', [[-0.86, 3.14], [ratio, ratio]], {'color':'black', 'linestyle':linestyle,
                                                                    'label':f'Your Batch to {label} $C_f/STY_f$ ratio'}])

        data.append(sub_set)

    fig2, ax2 = bar_plots(5, 1, sub_plt, x, data, (1, 17, 1), 
                          'Ratio of Batch (min and max) to Continuous $C_f/STY_f$')
    st.session_state.fig2 = fig2
    st.session_state.ax2 = ax2

else:
    updates = {'Your Batch to Semi-continuous $C_f/STY_f$ ratio': semi_ratio, 
               'Your Batch to Continuous $C_f/STY_f$ ratio': cont_ratio}
    for ax in st.session_state.ax2:
        for line in ax.lines:
            if line.get_label() in updates:
                line.set_ydata(2*[updates[line.get_label()]])

st.pyplot(st.session_state.fig2)
st.caption("The ratios of your batch to continuous and of your batch to semi-continuous $C_f/STY_f$ are shown, along with the ratios for a range of process designs. The ratio of batch $C_f$/$STY_f$ (min values are the ratios for $C_f=1$ and max values are the ratios for the asymptote $C_f$ approaches as $N$ goes to infinity) to continuous $C_f$/$STY_f$ (20d and 60d continuous cell harvest, $N=5$, and staggered reactors to maximize utilization) for a range of doubling times (10-30h), of doublings per batch expansion step (1-9), and of turnaround times (0-75h), all with $R=0.6$ are shown. For reference, the fold-increase in $C_{min}$ associated with $V_{max}$ decreasing by 1-3 orders of magnitude is also shown in all plots.")


st.subheader("Capital-investement Dependent Costs")
st.markdown("In technoeconomic modeling, several variables and assumptions affect capital investment contribution to cost of production, $COP_c$, but these variables can be grouped such that $COP_c$ is given by  ")
spacer, eq_column = st.columns([0.1, 0.9])
with eq_column:
    st.markdown(r"$$COP_c = \frac{C_f}{STY_f} \cdot \frac{C_{min}}{X_{max}} \cdot \frac{C_{tot}}{C} \cdot L_f \cdot A_f$$")

st.markdown(r"where $C_{tot}/C$ is the ratio of the total seed train equipment purchase cost (including any accounted for or necessary supporting equipment, i.e. tanks, pumps, and cleaning equipment) to seed train bioreactor purchase cost, $L_f$ is the Lang factor (cost escalation factor that estimates total capital investment, including costs such as installation, piping, instrumentation, buildings, and engineering, from bare equipment purchase cost), and $A_f$ is an annualization factor (converts total capital investment into an annual operating cost and represents modeled assumptions around depreciation and the cost of capital, as well as other annual costs such as maintenance, insurance, and taxes). This equation provides a useful framework through which to evaluate the impacts of different process designs and operating strategies on cost of production. $C_{min}$ and $X_{max}$ are bioreactor specific parameters, and $C_{tot}/C$ is a process specific parameter. The specification of $L_f$ and $A_f$ is unnecessary when comparing process designs on a relative basis rather than providing absolute values. If all other terms in the equation are equal, then comparing batch versus continuous $C_f$/$STY_f$ values directly indicates the impact of processing mode on $COP_c$. Equally, lower continuous processing $C_f$/$STY_f$ values relative to batch processing indicate the room available (or lack thereof) for potential trade-offs such as increases in $C_{min}$ (potentially due to, for example, a more expensive bioreactor type or scalability constraints), reductions in $X_{max}$, and increases in $C_{tot}$/$C$ (potentially due to additional supporting equipment). This is illustrated through the contour plot and waterfall charts shown below. Each contour line represents a batch to continuous $C_f/STY_f$ ratio (given by the inline label) and indicates the criteria that a continuous process must meet in order to reach $COP_c$ parity with the comparison batch process. For example, for a batch to continuous $C_f/STY_f$ ratio of 2 (and contour line labeled 2), a continuous process that results in a loss of one order of magnitude of bioreactor scalability requires a batch to continuous $1/X_{max} \cdot C_{tot}/C$ ratio of 1.25 to reach $COP_c$ parity with the comparison batch process. A ratio of 1.25 would indicate that compared to the batch process, the continuous process supports a greater $X_{max}$, requires less or cheaper supporting equipment (lower $C_{tot}/C$), or some combination of the two.")

# Contour plot (needs to be regenerated to add new ratio lines)
ratio = np.linspace(1,15, 15)
x = np.linspace(0, 2, 201)[1:]
y = 10**np.linspace(-1,3,21)
fig3, ax3 = contour_plots(x=x, y=y, levels=ratio)

ax3.annotate('Continuous preferred', 
            xy=(0.3,400),      
            xytext=(1.35, 0.2),  
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<|-|>', lw=0.5
    ))

# Add new ratio contour line
xx, yy = np.meshgrid(x, y)
zz = 1/(xx*(yy)**(batch_params['R']-1))
cs2 = ax3.contour(xx, yy, zz, levels=[semi_ratio, cont_ratio], colors='black', linestyles=['dashed', 'dotted'], linewidths=0.75)
ax3.clabel(cs2, inline=True, fontsize=8, fmt='%.2f', colors='black')

x = [(batch_params['C_{tot}/C']/batch_params['X_{max}'])/(continuous_params['C_{tot}/C']/continuous_params['X_{max}']),
     (batch_params['C_{tot}/C']/batch_params['X_{max}'])/(semi_params['C_{tot}/C']/semi_params['X_{max}'])]
y = [batch_params['V_{max}']/continuous_params['V_{max}'], batch_params['V_{max}']/semi_params['V_{max}']]

ax3.scatter(x[0], y[0], marker='o', color='black', s=15, label="Your continuous process")
ax3.scatter(x[1], y[1], marker='^', color='black', s=15, label="Your semi-continuous process")
ax3.legend(loc='upper right', fontsize=8)
      
ax3.text(0.1, 440, 'Batch preferred')
ax3.set_yscale('log')
st.pyplot(fig3)
st.caption(f"This contour plot illustrates the combinations of trade-offs for a given batch to continuous $C_f/STY_f$ ratio at which the continuous and batch process result in equivalent $COP_c$. Each contour line represents a batch to continuous $C_f/STY_f$ ratio (given by the inline label) and indicates the criteria that a continuous process must meet in order to reach $COP_c$ parity with the comparison batch process. The ratio of batch to continuous $C_{{min}}$ is calculated from the ratio of batch to continuous $V_{{max}}$ based on $R=0.6$ for the reference contour lines and based on $R={batch_params['R']}$ for your processes.")


# Waterfall chart
fig4, ax4 = plt.subplots(ncols=2)
share_max = max([max(semi_waterfall_data), max(continuous_waterfall_data)])  # Ensure both plots have the same y-axis limit
share_ticks = np.arange(-1, 0.5+share_max, 0.5) 
waterfall_plots(semi_waterfall_data, share_ticks, fig=fig4, ax=ax4[0], 
                ylab=r'Fold-change in $COP_c$ for Batch versus Semi-continuous')
waterfall_plots(continuous_waterfall_data, np.arange(-1, 0.5+max(continuous_waterfall_data), 0.5), 
                fig=fig4, ax=ax4[1])
fig4.tight_layout()
st.pyplot(fig4)
st.caption("The cumulative sequential impact of trade-offs in your batch versus semi-continuous and your batch versus continuous processes on $COP_c$ are shown. Green and red bars indicate relative advantages and disadvantages respectively of the continuous versus batch processes considered. Gray bars indicate the cumulative impact on $COP_c$.")

# =============================================================================
# Footer
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About this App**

This Streamlit app provides an interactive interface for comparing batch and continuous bioprocesses and visualizing space-time yield factors and cost factors. It supplements Westerhoff thesis (2025).
""")

end = time.time()
print(f"Total execution time: {end - start:.2f} seconds")