import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
import math 
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

plt.rcParams.update(
          {
              "font.family": "serif",  # use serif/main font for text elements
              "text.usetex": True,  # use inline math for ticks
              "pgf.texsystem": "pdflatex",
              "pgf.preamble": "\n".join(
                  [
                      r"\usepackage[utf8x]{inputenc}",
                      r"\usepackage[T1]{fontenc}",
                      #r"\usepackage{cmbright}",
                  ]
              ),
          }
      )
plt.rcParams.update({'axes.labelsize': 14,    # Taille du texte des labels des axes
                     'axes.titlesize': 14,   # Taille du texte du titre des axes
                     'xtick.labelsize': 14,  # Taille du texte des labels des ticks X
                     'ytick.labelsize': 14,  # Taille du texte des labels des ticks Y
                     'font.size': 13,        # Taille générale du texte
                     'legend.fontsize': 15})

def compute_chi2 (nombres_rockfall_apres_cdt,nombres_no_rockfall_apres_cdt, nombres_rockfall_apres_pas_cdt ,nombres_no_rockfall_apres_pas_cdt, HR_values, HP_values , alpha= 0.05  ):

    # ajout d'une petite valeur
    epsilon = 1e-205
    seuil = 3.841
    p_values=np.zeros((len(HR_values), len(HP_values)))
    decisions = np.zeros((len(HR_values), len(HP_values)), dtype=str)
    for i, HR_ in enumerate(HR_values):
     for j, HP_ in enumerate(HP_values):
        data = [[nombres_rockfall_apres_cdt[i,j] + epsilon, nombres_no_rockfall_apres_cdt[i,j] + epsilon],
                [nombres_rockfall_apres_pas_cdt [i,j] + epsilon, nombres_no_rockfall_apres_pas_cdt[i,j] + epsilon]]
        stat, p, dof, expected = chi2_contingency(data)
        p_values[i,j]= p
        if p <= alpha:
            decisions[i, j] = 'D'  #dependante
        else:
            decisions[i, j] = 'I'  #independante

    return p_values, decisions



def calcul_IC(nombres_rockfall_apres_cdt, nombres_no_rockfall_apres_cdt, HR , HP, alpha=0.05 ):
  lower_bound = np.zeros((HR, HP))
  upper_bound = np.zeros((HR, HP))
  margin_of_error= np.zeros((HR, HP))
  for i in range(0,nombres_rockfall_apres_cdt.shape[0]):
    for j in range(nombres_rockfall_apres_cdt.shape[1]):
      rockfall = nombres_rockfall_apres_cdt[i, j]
      no_rockfall = nombres_no_rockfall_apres_cdt[i, j]
      n = rockfall + no_rockfall

      if n<30  or rockfall==0 :
        lower_bound[i, j] = np.nan
        upper_bound[i, j] = np.nan
        margin_of_error[i, j] = np.nan

      elif n>=30:
        #calcul de la proportion observée
        p = rockfall / n
        #niveau de confiance (95%)
        Z = 1.96
        SE = math.sqrt((p * (1 - p)) / n)
        #marge d'Erreur
        ME = Z * SE
        # Intervalle de Confiance
        lower_bound[i, j] = p - ME
        upper_bound[i, j] = p + ME
        margin_of_error[i, j] = ME

  return margin_of_error


cmap= "RdBu_r"

def plot_fig_proba(outputFolder, filepath, HR_values, HP_values):
    print(filepath)
    fname = os.path.basename(filepath).lower()
    print(fname)
    values = np.loadtxt(filepath)
    fig, ax = plt.subplots(figsize=(10, 6))

    if "lift" in fname:
        metric = "lift"

        fmt=".2f"
        sns.heatmap(values, xticklabels=HP_values, yticklabels=HR_values, cmap=cmap, annot=True, fmt=fmt, ax=ax, vmin=0, vmax=2, linewidths=0.1, linecolor='gray')
    elif "p(" in fname or "prob" in fname:
        metric = "probability"
        sns.heatmap(values, xticklabels=HP_values, yticklabels=HR_values, cmap=cmap, annot=True, ax=ax, vmin=0, vmax=1, linewidths=0.1, linecolor='gray')
    elif "nombres" in fname:
        fmt = ".0f"
        sns.heatmap(values, xticklabels=HP_values, yticklabels=HR_values, cmap=cmap, annot=True, fmt=fmt, ax=ax, linewidths=0.1, linecolor='gray')
    elif "margin" in fname :
        # cmap = "Greys_r"
        fmt=".2f"
        sns.heatmap(values, xticklabels=HP_values, yticklabels=HR_values, cmap=cmap[:-2], annot=True, fmt=fmt, ax=ax, linewidths=0.1, linecolor='gray')

    ax.invert_yaxis()
    ax.set_xlabel("Forecast Horizon (FH)")
    ax.set_ylabel("Retrospective Horizon (RH)")
    colorbar = ax.collections[0].colorbar
    colorbar.outline.set_linewidth(0.15)
    plt.tight_layout()
    plt.show()
    outname = os.path.basename(filepath).replace(".txt", ".png")
    print(outname)
    outpath = os.path.join(outputFolder, outname)

    fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.12)

def format_pvalue(x, threshold=1e-2):
    if x == 0:
        return "0"
    elif x < threshold:
        return f"{x:.1e}"   # notation scientifique
    else:
        return f"{x:.2f}"

def plot_fig_p_value(filepath, HR_values, HP_values):
    fname = os.path.basename(filepath).lower()
    values = np.loadtxt(filepath)
    plt.rcParams.update({'axes.labelsize': 14,    # Taille du texte des labels des axes
                     'axes.titlesize': 14,   # Taille du texte du titre des axes
                     'xtick.labelsize': 14,  # Taille du texte des labels des ticks X
                     'ytick.labelsize': 14,  # Taille du texte des labels des ticks Y
                     'font.size': 11,        # Taille générale du texte
                     'legend.fontsize': 15})
    fig, ax = plt.subplots(figsize=(10, 6))
    if "p_value" in fname :
        metric = "pvalue"
        # cmap = "Greys_r"
        annot_labels = np.vectorize(format_pvalue)(values)

        sns.heatmap(values, xticklabels=HP_values, yticklabels=HR_values, cmap=cmap[:-2], annot=annot_labels, fmt="", ax=ax, linewidths=0.1, linecolor='gray')

        ax.invert_yaxis()
        ax.set_xlabel("Forecast Horizon (FH)")
        ax.set_ylabel("Retrospective Horizon (RH)")
        colorbar = ax.collections[0].colorbar
        colorbar.outline.set_linewidth(0.15)
        plt.tight_layout()
        plt.show()
        os.makedirs("image", exist_ok=True)
        outname = os.path.basename(filepath).replace(".txt", ".png")
        outpath = os.path.join("image", outname)
        fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.12)

def plot_IC(margin_of_error, probabilities, filepath, HR_values, HP_values):
    fig, ax = plt.subplots(figsize=(17,10))
    #les annotations
    annotations = np.empty(margin_of_error.shape, dtype=object)
    for i in range(margin_of_error.shape[0]):
        for j in range(margin_of_error.shape[1]):
            if probabilities[i, j] == 0:
              annotations[i, j] = ""  #case vide pour probabilité nulle
            elif margin_of_error[i, j] == np.nan:
              annotations[i, j] = ""
            else:
              annotations[i, j] = f"{probabilities[i, j]:.2f}±{margin_of_error[i, j]:.2f}"

    #créer un masque pour les valeurs nulles
    mask = probabilities == 0
    #heatmap avec le masque
    # cmap='Greys'
    sns.heatmap(margin_of_error, xticklabels=HP_values, yticklabels=HR_values, annot=annotations, cmap=cmap[:-2], fmt="", mask=mask, cbar=True, cbar_kws={"pad": 0.01}, linewidths=0.1, linecolor='gray')
    ax.invert_yaxis()
    ax.set_xlabel('Forecast Horizon (FH)')
    ax.set_ylabel('Retrospective Horizon (RH)')
    colorbar = ax.collections[0].colorbar
    colorbar.outline.set_linewidth(0.15)
    plt.tight_layout()
    plt.show()
    os.makedirs("image", exist_ok=True)
    outname = os.path.basename(filepath).replace(".txt", ".png")
    outpath = os.path.join("image", outname)
    fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.12)



def plot_aggregation( LIFT, p_values, margin_of_error, HR, HP, outputFolder, filename, 
                            pval_threshold=0.05):
    # Colormaps
    cmap = plt.get_cmap("Greys")
    low_color  = cmap(0.0)
    med_color  = '#e7eac5'
    high_color = '#E8AABE'

    cmap_ind = LinearSegmentedColormap.from_list(
        'Custom_ind', [low_color, med_color], 2
    )
    norm_binary = BoundaryNorm([-0.5, 0.5, 1.5], cmap_ind.N)

    cmap_agg = LinearSegmentedColormap.from_list(
        'Custom_agg', [low_color, med_color, high_color], 3
    )
    norm_agg = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_agg.N)

    # Binary classification
    lift_classified = (LIFT > 1).astype(int)
    p_values_classified = (p_values < pval_threshold).astype(int)

    # Aggregation
    aggregated = lift_classified + p_values_classified

    # Figure
    fig, axs = plt.subplots(1, 4, figsize=(25, 5.5))

    # Lift
    sns.heatmap( lift_classified, xticklabels=HP, yticklabels=HR,
                  cmap=cmap_ind, norm=norm_binary, ax=axs[0],
                  cbar=True, cbar_kws={'ticks': [0, 1]},
                  linewidths=0.1, linecolor='gray')
    axs[0].invert_yaxis()
    axs[0].set_xlabel('Forecast Horizon (FH)')
    axs[0].set_ylabel('Retrospective Horizon (RH)')
    axs[0].set_title('$Lift > 1$')

    # P-values
    sns.heatmap(p_values_classified, xticklabels=HP, yticklabels=HR,
                cmap=cmap_ind, norm=norm_binary, ax=axs[1],
                cbar=True, cbar_kws={'ticks': [0, 1]},
                linewidths=0.1, linecolor='gray')
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Forecast Horizon (FH)')
    axs[1].set_title('$P-value < threshold$')

    # Aggregation
    sns.heatmap(aggregated, xticklabels=HP, yticklabels=HR,
                cmap=cmap_agg, norm=norm_agg, ax=axs[2],
                cbar=True, vmin=0, vmax=2,
                cbar_kws={'ticks': [0, 1, 2]},
        linewidths=0.1, linecolor='gray'
    )
    axs[2].invert_yaxis()
    axs[2].set_xlabel('Forecast Horizon (FH)')
    axs[2].set_title('Aggregated values')

    # Confidence interval levels
    thresholds = np.arange(0.01, 0.11, 0.01).tolist()

    colors_blue_yellow = [
        '#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
        '#c6dbef', '#ffffd4', '#fed98e', '#fe9929', '#cc4c02'
    ]

    cmap_ci = LinearSegmentedColormap.from_list(
        'BlueYellow_div', colors_blue_yellow, N=len(thresholds)
    ).reversed()

    bounds = np.arange(0.5, len(thresholds) + 1.5, 1)
    norm_ci = BoundaryNorm(bounds, cmap_ci.N)

    level_matrix = np.full_like(margin_of_error, len(thresholds) + 1, dtype=float)

    for i, thresh in enumerate(thresholds):
        mask = (margin_of_error < thresh) & (level_matrix == len(thresholds) + 1)
        level_matrix[mask] = i + 1

    level_matrix[np.isnan(margin_of_error)] = np.nan

    sns.heatmap(
        level_matrix, xticklabels=HP, yticklabels=HR,
        cmap=cmap_ci, norm=norm_ci, ax=axs[3],
        cbar=True,
        linewidths=0.5, linecolor='gray',
        vmax=len(thresholds) + 0.999
    )
    axs[3].invert_yaxis()
    axs[3].set_xlabel('Forecast Horizon (FH)')
    axs[3].set_title('CI by threshold level')

    # Colorbars formatting
    for i in [0, 1]:
        cb = axs[i].collections[0].colorbar
        cb.outline.set_linewidth(0.15)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['0', '1'])

    cb = axs[2].collections[0].colorbar
    cb.outline.set_linewidth(0.15)
    cb.set_ticks([0.25, 1, 1.75])
    cb.set_ticklabels(['0', '1', '2'])

    cb = axs[3].collections[0].colorbar
    cb.outline.set_linewidth(0.15)
    cb.set_ticks(range(1, 11))
    cb.set_ticklabels([rf"${i}\%$" for i in range(1, 11)])

    plt.tight_layout()
    plt.show()

    outpath = os.path.join(outputFolder, filename)
    fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.12)
    plt.close()





def compute_ci_level_matrix(margin_of_error, thresholds):
    level_matrix = np.full_like(margin_of_error, len(thresholds) + 1, dtype=float)

    for i, thresh in enumerate(thresholds):
        mask = (margin_of_error < thresh) & (level_matrix == len(thresholds) + 1)
        level_matrix[mask] = i + 1

    level_matrix[np.isnan(margin_of_error)] = np.nan
    return level_matrix




def plot_probability_heatmaps_levels(outputFolder, modality, conditions, HP_values, filename):

    plt.rcParams.update({
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'font.size': 15,
        'legend.fontsize': 15
    })

    files = os.listdir(outputFolder)
    ordered_files = []

    for c in conditions:
        for f in files:
            if c in f and f.startswith("p(R") and f.endswith(".txt"):
                ordered_files.append((c, os.path.join(outputFolder, f)))

    if len(ordered_files) == 0:
        raise ValueError("No valid probability files found in the folder.")

    fig, axes = plt.subplots(len(ordered_files), 1, figsize=(10, 4), sharex=True)

    if len(ordered_files) == 1:
        axes = [axes]

    for ax, (cond, filepath) in zip(axes, ordered_files):
        data = np.loadtxt(filepath).reshape(1, len(HP_values))

        sns.heatmap(data, ax=ax,cmap="RdBu_r", vmin=0, vmax=1, annot=True, fmt=".2f", xticklabels=HP_values,
            yticklabels=[], cbar=False ,  linewidths=0.1, linecolor='gray')

        ax.invert_yaxis()
        ax.set_ylabel(fr"${modality}_{{\mathrm{{{cond}}}}}$", rotation=0, labelpad=40)

    axes[-1].set_xlabel("Forecast Horizon (FH)")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.11, 0.015, 0.77])

    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])

    fig.colorbar(sm, cax=cbar_ax)

    plt.show()

    outpath = os.path.join(outputFolder, filename)
    fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.12)

    plt.close()




def plot_aggregation_heatmaps_levels(outputFolder, modality, conditions, HP_values, compute_ci_level_matrix, filename):
    files = os.listdir(outputFolder)
    print(files)
    ordered = []
    for c in conditions:
        lift_f = f"LIFT_{modality} = {c}.txt"
        pval_f = f"p_value {modality} = {c}.txt"
        ic_f   = f"margin_of_error_p(R|{modality} avec {modality} = {c}).txt"

        if all(f in files for f in [lift_f, pval_f, ic_f]):
            ordered.append((
                c,
                os.path.join(outputFolder, lift_f),
                os.path.join(outputFolder, pval_f),
                os.path.join(outputFolder, ic_f)
            ))


    if len(ordered) == 0:
        raise ValueError("No valid aggregation files found.")

    plt.rcParams.update({
                        'axes.labelsize': 16,
                        'axes.titlesize': 16,
                        'xtick.labelsize': 16,
                        'ytick.labelsize': 16,
                        'font.size': 15,
                        'legend.fontsize': 16 })

    cmap = plt.get_cmap("Greys")
    low_color  = cmap(0.0)
    med_color  = '#e7eac5'
    high_color = '#E8AABE'

    #  Binaires (0 / 1)
    colors_ind = [low_color, med_color]
    cmap_ind = LinearSegmentedColormap.from_list('Custom_ind', colors_ind, 2)
    norm_binary = BoundaryNorm([-0.5, 0.5, 1.5], cmap_ind.N)

    # Agrégation (0 / 1 / 2)
    colors_agg = [low_color, med_color, high_color]
    cmap_agg = LinearSegmentedColormap.from_list('Custom_agg', colors_agg, 3)
    norm_agg = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_agg.N)

    # CI
    thresholds = np.arange(0.01, 0.11, 0.01).tolist()
    colors_blue_yellow = [
        '#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
        '#c6dbef', '#ffffd4', '#fed98e', '#fe9929', '#cc4c02']
    cmap_ci = LinearSegmentedColormap.from_list('BlueYellow_div', colors_blue_yellow, N=len(thresholds)).reversed()
    bounds = np.arange(0.5, len(thresholds) + 1.5, 1)
    norm_ci = BoundaryNorm(bounds, cmap_ci.N)

    n_rows = len(ordered)

    fig = plt.figure(figsize=(20, 3))

    outer_gs = fig.add_gridspec(
        n_rows, 6,
        width_ratios=[1, 1, 1, 0.06, 1, 0.06],
        hspace=0.1, wspace=0.25
    )

    for row_idx, (cond, f_lift, f_pval, f_ic) in enumerate(ordered):

        LIFT = np.loadtxt(f_lift).reshape(1, len(HP_values))
        PVAL = np.loadtxt(f_pval).reshape(1, len(HP_values))
        IC   = np.loadtxt(f_ic).reshape(1, len(HP_values))

        lift_bin = (LIFT > 1).astype(int)
        pval_bin = (PVAL < 0.05).astype(int)
        agg = lift_bin + pval_bin

        level_matrix = compute_ci_level_matrix(IC, thresholds)

        ax0 = fig.add_subplot(outer_gs[row_idx, 0])
        ax1 = fig.add_subplot(outer_gs[row_idx, 1])
        ax2 = fig.add_subplot(outer_gs[row_idx, 2])
        ax3 = fig.add_subplot(outer_gs[row_idx, 4])

        im0 = sns.heatmap(
            lift_bin, ax=ax0,
            cmap=cmap_ind, norm=norm_binary,
            cbar=False, linewidths=0.1, linecolor='gray',
            xticklabels=HP_values, yticklabels=[]
        )

        im1 = sns.heatmap(
            pval_bin, ax=ax1,
            cmap=cmap_ind, norm=norm_binary,
            cbar=False, linewidths=0.1, linecolor='gray',
            xticklabels=HP_values, yticklabels=[]
        )

        im2 = sns.heatmap(
            agg, ax=ax2,
            cmap=cmap_agg, norm=norm_agg,
            cbar=False, linewidths=0.1, linecolor='gray',
            xticklabels=HP_values, yticklabels=[]
        )

        im3 = sns.heatmap(
            level_matrix, ax=ax3,
            cmap=cmap_ci, norm=norm_ci,
            cbar=False, linewidths=0.1, linecolor='gray',
            xticklabels=HP_values, yticklabels=[]
        )

        axes_row = [ax0, ax1, ax2, ax3]

        if row_idx == n_rows - 1:
            for ax in axes_row:
                ax.set_xlabel("Forecast Horizon (FH)")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        else:
            for ax in axes_row:
                ax.set_xticklabels([])

        for ax in axes_row:
            ax.invert_yaxis()

        ax0.set_ylabel(fr"${modality}_{{\mathrm{{{cond}}}}}$", rotation=0, labelpad=40)

        if row_idx == 0:
            ax0.set_title("$Lift > 1$")
            ax1.set_title("$P-value < 0.05$")
            ax2.set_title("Aggregated values")
            ax3.set_title("Confidence Interval")

            cax_agg = fig.add_subplot(outer_gs[:, 3])
            cbar_agg = fig.colorbar(im2.get_children()[0], cax=cax_agg)
            cbar_agg.set_ticks([0, 1, 2])
            cbar_agg.set_ticklabels(['0', '1', '2'])

            cax_ci = fig.add_subplot(outer_gs[:, 5])
            cbar_ci = fig.colorbar(im3.get_children()[0], cax=cax_ci)
            cbar_ci.set_ticks(bounds[1:])
            cbar_ci.set_ticklabels([rf"${i}\%$" for i in range(1, 11)])

    plt.show()
    outpath = os.path.join(outputFolder, filename)
    fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.12)
    plt.close()