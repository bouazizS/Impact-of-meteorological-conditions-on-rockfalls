import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
import multiprocessing
from scipy.stats import linregress
import scipy.stats as stats
import seaborn as sns
import os
import math
from matplotlib.colors import LinearSegmentedColormap
from utilis import plot_fig_proba , compute_chi2 , calcul_IC , plot_aggregation

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

#------------------------- GEL Daily computes ----------------------------------


def get_consecutive_negative_days(df, n, check):  
  df = df.copy()

  df["AAAAMMJJHH"] = pd.to_datetime(df["AAAAMMJJHH"])
  df["AAAAMMJJHH"] = df["AAAAMMJJHH"].dt.date
  if check== "any": 
    daily_negative = df.groupby("AAAAMMJJHH")[" T"].apply(lambda x: (x < 0).any())
    daily_negative_df = daily_negative.reset_index(name="negative_temp")

    # Consecutive periods where temperatures are negative
    periods = []
    current_period = []
    for i in range(len(daily_negative_df) - n + 1):
      if daily_negative_df['negative_temp'].iloc[i:i+n].all(): 
        debut = daily_negative_df['AAAAMMJJHH'].iloc[i]
        fin = daily_negative_df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Start_Date': debut, 'Date Fin': fin})

  elif check=='all':
    daily_positive = df.groupby("AAAAMMJJHH")[" T"].apply(lambda x: (x >= 0).all())
    daily_positive_df = daily_positive.reset_index(name="negative_temp")
    # Consecutive periods where temperatures are positive
    periods = []
    current_period = []
    for i in range(len(daily_positive_df) - n + 1):
      # There is at least one positive temperature (T>0) so discontinuous freeze
      if daily_positive_df['negative_temp'].iloc[i:i+n].any(): 
        debut = daily_positive_df['AAAAMMJJHH'].iloc[i]
        fin = daily_positive_df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Start_Date': debut, 'Date Fin': fin})

  return pd.DataFrame(periods)


def Probability (x, df_T, df_sismo,y, check):   
    # Slicing using dates available in the seismic catalog
    date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0]) 
    date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  
    date_ref = date_obj - pd.Timedelta(days=x)
    date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

    # Filter temperature data for the reference period
    df_T.set_index('AAAAMMJJHH', inplace=True)
    periods_df = df_T.loc[date_ref:date_fin_ref]
    periods_df.reset_index(inplace=True)
    df_T.reset_index(inplace=True)

    periods = get_consecutive_negative_days(periods_df, x, check)

    if periods.empty:
      return 0, 0, 0
    periods.set_index('Start_Date', inplace=True)

    df_sismo = df_sismo.copy()
    df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo.set_index('Date', inplace=True)

    # Initialisation
    rockfall_HR_days_after = []

    for date in periods.index:
      # Define the period (dates and hours) to analyze
      start_date = date + pd.Timedelta(days=x)
      end_date = date + pd.Timedelta(days=x + y - 1)

      df_sismo_filtered = df_sismo.loc[start_date:end_date]

      # Check for rockfall events
      rockfall_events = df_sismo_filtered[df_sismo_filtered['type'] == 'R']

      rockfall_detected = False
      if not rockfall_events.empty:  # If rockfalls exist
          rockfall_detected = True
      else:
          rockfall_detected = False

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

    resultat = periods.copy()
    resultat['rockfall sur période'] = rockfall_HR_days_after
    proba = resultat['rockfall sur période'].mean()  

    nbr= resultat['rockfall sur période'].sum()
    nbr_no=len(resultat)-nbr

    return proba, nbr, nbr_no

def probability_freeze(HR, df, df_sismo, HP):
    # Compute the probability of rockfall for freeze periods
    check_gel="any"
    # "any" means at least one temperature below 0 in the day
    proba_gel, nbr_gel_r ,nbr_gel_noR = Probability(HR, df, df_sismo, HP,check_gel)

    return proba_gel , nbr_gel_r ,nbr_gel_noR

def probability_non_freeze(HR, df, df_sismo, HP):
    # Compute the probability of rockfall for non-freeze periods
    condition_pas_gel = lambda x: x > 0
    #all pour dire tous les temp sont >0
    check_pas_gel="all"
    proba_pas_gel, nbr_pasG_r , nbr_pasG_noR= Probability(HR, df, df_sismo, HP,check_pas_gel)

    return proba_pas_gel, nbr_pasG_r, nbr_pasG_noR

def find_periodes_HR_freeze_non_freeze(df, n):
    periods = []
    for i in range(len(df) - n + 1):
        debut = df['AAAAMMJJHH'].iloc[i]
        fin = df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Start_Date': debut, 'Date Fin': fin})
    return pd.DataFrame(periods)

def Nbr_rockfall_onHP_LIFT(x, df, df_sismo, y):
  # Slicing using dates available in the seismic catalog
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # START of seismic
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # END of seismic
  date_ref = date_obj - pd.Timedelta(days=x)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)
  # Filter temperature data for the reference period
  df.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df.reset_index(inplace=True)

  # Find freeze and non-freeze periods
  periods = find_periodes_HR_freeze_non_freeze(periods_df, x)
  periods.set_index('Start_Date', inplace=True)

  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  rockfall_HR_days_after = []

  for date in periods.index:
    start_date = date + pd.Timedelta(days=x)
    end_date = date + pd.Timedelta(days=x + y - 1)

    # Filter events in the chosen period
    df_sismo_filtered = df_sismo.loc[start_date:end_date]

    # Verify rockfall events
    rockfall_events = df_sismo_filtered[df_sismo_filtered['type'] == 'R']

    rockfall_detected = False

    if not rockfall_events.empty:  # If rockfalls exist
      rockfall_detected = True
    else:
      rockfall_detected = False

    rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods.copy()
  resultat['rockfall sur période'] = rockfall_HR_days_after
  nbr_rockfall_HP = resultat['rockfall sur période'].sum()
  nbr_jours=len(resultat)

  return nbr_rockfall_HP, nbr_jours




def worker(task):
  """ Treat a couple (HR, HP)"""
  i, j, HR, HP = task
  proba_gel, nbr_rockfall, nbr_no_rockfall = probability_freeze(HR, df_T, df_sismo, HP)

  _, nbr_R_pasgel, nbr_pasR_pasgel  = probability_non_freeze(HR, df_T, df_sismo, HP)

  nbr_rockfall_HP, nbr_observables = Nbr_rockfall_onHP_LIFT(HR, df_Daily_T, df_sismo, HP)

  proba_R = nbr_rockfall_HP / nbr_observables if nbr_observables > 0 else 0
  lift_gel = proba_gel / proba_R if proba_R > 0 else 0   

  print(f"(HR={HR}, HP={HP}) finished")
  return i, j, proba_gel, nbr_rockfall, nbr_no_rockfall, nbr_R_pasgel, nbr_pasR_pasgel, lift_gel, proba_R


if __name__ == "__main__":

    df_T= pd.read_csv("/Data/1.0_ T_St_Hiliaire_filled.csv", delimiter='\t')
    df_T['AAAAMMJJHH'] = pd.to_datetime(df_T['AAAAMMJJHH'])
    df_T["AAAAMMJJHH"] = df_T["AAAAMMJJHH"].dt.date
    df_T['AAAAMMJJHH'] = pd.to_datetime(df_T['AAAAMMJJHH'])

    daily_T = df_T.groupby("AAAAMMJJHH")[" T"].apply(lambda x: (x < 0).any())
    df_Daily_T = daily_T.reset_index(name="negative_temp")


    df_sismo= pd.read_csv("/Data/ev_sismo2.csv")
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])

    # Conserve only common dates
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_T['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_T['AAAAMMJJHH'].max())

    # Filter DataFrames to keep only common dates
    df_sismo = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_T = df_T[(df_T['AAAAMMJJHH']>= start_date) & (df_T['AAAAMMJJHH'] <= end_date)]

    # File to save results
    outputFolder="/Probabilities_rockfall_conditions/continuous_freeze"
    os.makedirs(outputFolder, exist_ok=True)

    
    # Define size of retrospective horizon and forecasting horizon 
    HR_values = range(1, 15)
    HP_values = range(1, 15)
    # Compute probabilities for HR and HP horizons
    proba_values_gel = np.zeros((len(HR_values), len(HP_values)))

    numbers_rockfall_after_freeze=np.zeros((len(HR_values), len(HP_values)))
    numbers_rockfall_after_no_freeze= np.zeros((len(HR_values), len(HP_values)))

    numbers_no_rockfall_after_freeze = np.zeros((len(HR_values), len(HP_values)))
    numbers_no_rockfall_after_no_freeze = np.zeros((len(HR_values), len(HP_values)))

    lift=np.zeros((len(HR_values), len(HP_values)))
    proba_r=np.zeros((len(HR_values), len(HP_values)))

    tasks = [(i, j, HR, HP) for i, HR in enumerate(HR_values) for j, HP in enumerate(HP_values)]

    num_processes = int(multiprocessing.cpu_count()*0.8)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(worker, tasks)


    for result in results:
      i, j, proba_gel, nbr_rockfall, nbr_no_rockfall, nbr_R_pasgel, nbr_pasR_pasgel, lift_gel, proba_R = result
      proba_values_gel[i, j] = proba_gel
      numbers_rockfall_after_freeze[i, j] = nbr_rockfall
      numbers_no_rockfall_after_freeze[i, j] = nbr_no_rockfall
      numbers_rockfall_after_no_freeze[i, j] = nbr_R_pasgel
      numbers_no_rockfall_after_no_freeze[i, j] = nbr_pasR_pasgel
      lift[i,j]= lift_gel
      proba_r[i,j]= proba_R

 
    proba_Rockfall_given_freeze= "p(rHF|fHR).txt"
    np.savetxt(os.path.join(outputFolder, proba_Rockfall_given_freeze), proba_values_gel)
    np.savetxt(os.path.join(outputFolder,'numbers_rockfall_after_freeze.txt'), numbers_rockfall_after_freeze)
    np.savetxt(os.path.join(outputFolder,'numbers_no_rockfall_after_freeze.txt'), numbers_no_rockfall_after_freeze)
    np.savetxt(os.path.join(outputFolder,'numbers_rockfall_after_no_freeze.txt'), numbers_rockfall_after_no_freeze)
    np.savetxt(os.path.join(outputFolder,'numbers_no_rockfall_after_no_freeze.txt'), numbers_no_rockfall_after_no_freeze)
    np.savetxt(os.path.join(outputFolder, 'lift.txt'), lift)
    np.savetxt(os.path.join(outputFolder, 'proba_rockfall_global.txt'), proba_r)
    
    # Compute and save the P-values and margin of error 
    HR=14
    HP=14
    margin_of_error= calcul_IC(numbers_rockfall_after_freeze, numbers_no_rockfall_after_freeze, HR , HP)
    p_values, decisions = compute_chi2(numbers_rockfall_after_freeze,numbers_no_rockfall_after_freeze,
                                       numbers_rockfall_after_no_freeze,numbers_no_rockfall_after_no_freeze, HR_values, HP_values )
    np.savetxt(os.path.join(outputFolder, f'Khi-deux_gel-Rockfall.txt'), p_values)
    np.savetxt(os.path.join(outputFolder, f'margin_of_error_freeze_Rockfall.txt'), margin_of_error)


    # Plot figure of probability and aggregation metrics
    files=os.listdir(outputFolder)
    for f in files:
      if f.endswith(".txt") : 
        print(f)
        # Plot the figure if it is a probability file
        if "p(r" in f:
          plt.rcParams.update({'axes.labelsize': 14,    # Size of text of labels axes
                        'axes.titlesize': 14,   # Size of axes title text
                        'xtick.labelsize': 14,  # Size of text of X ticks labels
                        'ytick.labelsize': 14,  # Size of text of Y ticks labels
                        'font.size': 13,        # General text size
                        'legend.fontsize': 15})
          plot_fig_proba(outputFolder , os.path.join(outputFolder, f), HR_values, HP_values )


    # Plot figure of aggregation
    # Figure size config :
    plt.rcParams.update({'axes.labelsize': 18,    # Size of text of labels axes
                                'axes.titlesize': 18,   # Size of axes title text
                                'xtick.labelsize': 18,  # Size of text of X ticks labels
                                'ytick.labelsize': 18,  # Size of text of Y ticks labels
                                'font.size': 16,        # General text size
                                'legend.fontsize': 18})
    plot_aggregation( LIFT=lift, p_values=p_values, margin_of_error=margin_of_error, outputFolder=outputFolder,
                     filename="Aggregation_freeze.png", HR=np.arange(1, 15), HP=np.arange(1, 15),
                       )








