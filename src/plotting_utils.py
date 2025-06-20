import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from waterfall_ax import WaterfallChart


def bar_plots(ncols, nrows, sub_plt, x, data, ylims, ylab):

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharey=True,
                           sharex=True)
    
    bar_w = 0.18
    xx = np.array([[j - (k*bar_w) for k in range(len(x))] for j in range(4)]).ravel()

    for i, ax in enumerate(axes):

        ax.set_title('Doubling Time: {:.0f}h'.format(sub_plt[i]), fontsize=7)
        ax.set_yticks(np.arange(*ylims), np.arange(*ylims), fontsize=7)
        ax.set_xticks(np.arange(-1, 4,)+0.14, labels=[], minor=False)
        ax.set_xlim(-0.86, 3.14)
        ax.set_ylim(ylims[0], ylims[1]-ylims[2])

        ax.set_xticks(np.ravel([[j - (k*bar_w)
                                    for k in range(len(x)-1, -1, -1)]
                                   for j in range(4)]),
                         labels=['{:.0f}{}'.format(i,j)
                                 for i,j in zip(np.tile(x[::-1], 4),
                                                ['', '', '\n ( 0 )', '', '']+
                                                ['', '', '\n ( 25 )', '', '']+
                                                ['', '', '\n ( 50 )', '', '']+
                                                ['', '', '\n ( 75 )', '', ''])],
                         rotation=0, minor=True,
                         fontsize=4)

        for dataset in data[i]:
            plotting, z, kwargs = dataset
        
            if plotting == 'fill':
                for xxs, yy, yy2 in zip(xx, z[0], z[1]):
                    ax.fill_between([xxs-0.45*bar_w, xxs+0.45*bar_w], 2*[yy], 2*[yy2],
                                    **kwargs)
            elif plotting == 'bar':
                yy = z
                ax.bar(xx, *yy, width=0.9*bar_w, **kwargs)
                    
            elif plotting == 'scatter':
                yy = z
                ax.scatter(xx, *yy, **kwargs)
            
            elif plotting == 'line':
                ax.plot(*z, **kwargs)
    
    fig.supylabel(ylab, fontsize=9)
    fig.supxlabel('Doublings per Batch Expansion Step \n (Turnaround Time [h])',
                  fontsize=9)
    fig.tight_layout(pad=0.7)
    
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[-1].legend(unique.values(), unique.keys(), fontsize='xx-small', loc='upper right', framealpha=1)
    
    return fig, axes


def contour_plots(x=np.linspace(0, 2, 201)[1:], y=10**np.linspace(0,3,21), 
                  func=lambda xx, yy: 1/(xx*(yy)**(0.6-1)), 
                  xlab=r'Ratio of Batch to Continuous $1/X_{max} \cdot C_{tot}/C$', 
                  ylab=r'Ratio of Batch to Continuous $V_{max}$', levels=None,
                  fig=None, ax=None):
    
    if not fig or not ax:
        fig, ax = plt.subplots()

    xx, yy = np.meshgrid(x, y)
    zz = func(xx, yy)
    cs = ax.contour(xx, yy, zz, levels=levels)
    clabel = ax.clabel(cs, inline=1, fontsize='small', inline_spacing=1)
    ax.set_xlim(0, 2)

    ax.set_xlabel(r'Ratio of Batch to Continuous $1/X_{max} \cdot C_{tot}/C$')
    ax.set_ylabel(r'Ratio of Batch to Continuous $V_{max}$')

    return fig, ax


def waterfall_plots(data, ticks, ylab=r'Fold-change in $COP_c$ for Batch versus Continuous',
                    names=[r'$\frac{C_f}{STY_f}$', r'$C_{min}$', r'$X_{max}$', r'$\frac{C_{tot}}{C}$'],
                    fig=None, ax=None):
    
    if not fig or not ax:
        fig, ax = plt.subplots()

    waterfall = WaterfallChart(data, step_names=names, last_step_label=r'$COP_c$')
    waterfall.plot_waterfall(ax=ax, bar_labels=False,
                             color_kwargs={'c_bar_pos': 'g', 'c_bar_neg': 'r',
                                           'c_bar_start': 'g', 'c_bar_end': 'gray'})
    
    ax.plot([*ax.get_xlim()], [0,0], linestyle=':', color='black', lw=0.5)
    ax.set_yticks(ticks, ticks+1)
    ax.set_ylim(-1, ticks[-1])
    ax.set_ylabel(ylab)

    return fig, ax