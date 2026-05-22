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

    #les périodes consécutives où les températures sont négatives
    periods = []
    current_period = []
    for i in range(len(daily_negative_df) - n + 1):
      if daily_negative_df['negative_temp'].iloc[i:i+n].all(): 
        debut = daily_negative_df['AAAAMMJJHH'].iloc[i]
        fin = daily_negative_df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Date Début': debut, 'Date Fin': fin})

  elif check=='all':
    daily_positive = df.groupby("AAAAMMJJHH")[" T"].apply(lambda x: (x >= 0).all())
    daily_positive_df = daily_positive.reset_index(name="negative_temp")
    #les périodes consécutives où les températures sont positives
    periods = []
    current_period = []
    for i in range(len(daily_positive_df) - n + 1):
      if daily_positive_df['negative_temp'].iloc[i:i+n].any(): #il y a au moins une temp>0 donc gel discontinue
        debut = daily_positive_df['AAAAMMJJHH'].iloc[i]
        fin = daily_positive_df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Date Début': debut, 'Date Fin': fin})

  return pd.DataFrame(periods)


def get_consecutive_pgel_days(df, n, check):  
  df = df.copy()

  df["AAAAMMJJHH"] = pd.to_datetime(df["AAAAMMJJHH"])
  df["AAAAMMJJHH"] = df["AAAAMMJJHH"].dt.date
  if check== "any": 
    daily_negative = df.groupby("AAAAMMJJHH")["pgel"].apply(lambda x: (x < 0).any())
    daily_negative_df = daily_negative.reset_index(name="negative_temp")

    #les périodes consécutives où les températures sont négatives
    periods = []
    current_period = []
    for i in range(len(daily_negative_df) - n + 1):
      if daily_negative_df['negative_temp'].iloc[i:i+n].all(): 
        debut = daily_negative_df['AAAAMMJJHH'].iloc[i]
        fin = daily_negative_df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Date Début': debut, 'Date Fin': fin})

  elif check=='all':
    daily_positive = df.groupby("AAAAMMJJHH")["pgel"].apply(lambda x: (x >= 0).all())
    daily_positive_df = daily_positive.reset_index(name="negative_temp")
    #les périodes consécutives où les températures sont positives
    periods = []
    current_period = []
    for i in range(len(daily_positive_df) - n + 1):
      if daily_positive_df['negative_temp'].iloc[i:i+n].any(): #il y a au moins une temp>0 donc gel discontinue
        debut = daily_positive_df['AAAAMMJJHH'].iloc[i]
        fin = daily_positive_df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Date Début': debut, 'Date Fin': fin})

  return pd.DataFrame(periods)

def Probability (x, df_T, df_sismo,y, check):   
    #slicing à partir des dates disponibles pour le catalogue sismo
    date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
    date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique
    date_ref = date_obj - pd.Timedelta(days=x)
    date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

    #filtrer les données de temp pour la période de référence
    df_T.set_index('AAAAMMJJHH', inplace=True)
    periods_df = df_T.loc[date_ref:date_fin_ref]
    periods_df.reset_index(inplace=True)
    df_T.reset_index(inplace=True)

    periods_2013 = get_consecutive_negative_days(periods_df, x, check)

    if periods_2013.empty:
      return 0, 0, 0
    periods_2013.set_index('Date Début', inplace=True)

    df_sismo = df_sismo.copy()
    df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo.set_index('Date', inplace=True)

    #initialisation
    rockfall_HR_days_after = []

    # Parcourir les dates de précipitations/ periodes
    for date in periods_2013.index:
      # Définir la période (dates et heures) à analyser
      start_date = date + pd.Timedelta(days=x)
      end_date = date + pd.Timedelta(days=x + y - 1)

      df_sismo_filtered = df_sismo.loc[start_date:end_date]

      # Vérifier si on a des événements rockfall
      rockfall_events = df_sismo_filtered[df_sismo_filtered['type'] == 'R']

      rockfall_detected = False
      if not rockfall_events.empty:  # Si des rockfalls existent
          rockfall_detected = True
      else:
          rockfall_detected = False

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

    resultat = periods_2013.copy()
    resultat['rockfall sur période'] = rockfall_HR_days_after
    proba = resultat['rockfall sur période'].mean()   #(nbr periodes suivie de rockfall / nbr de periodes pluie HP))

    nbr= resultat['rockfall sur période'].sum()
    nbr_no=len(resultat)-nbr

    return proba, nbr, nbr_no

def probability_gel(HR, df, df_sismo, HP):
    #calcl de la probabilité d'éboulement pour les périodes de gel
    check_gel="any"
    #any pour dire que au moins une température au dessous de 0 dans la journée
    proba_gel, nbr_gel_r ,nbr_gel_noR = Probability(HR, df, df_sismo, HP,check_gel)

    return proba_gel , nbr_gel_r ,nbr_gel_noR

def proba_pas_gel(HR, df, df_sismo, HP):
    #calcul de la probabilité d'éboulement pour les périodes de non-gel
    condition_pas_gel = lambda x: x > 0
    #all pour dire tous les temp sont >0
    check_pas_gel="all"
    proba_pas_gel, nbr_pasG_r , nbr_pasG_noR= Probability(HR, df, df_sismo, HP,check_pas_gel)

    return proba_pas_gel, nbr_pasG_r, nbr_pasG_noR

def trouver_periodes_HR_gel_pas_gel(df, n):
    periods = []
    for i in range(len(df) - n + 1):
        debut = df['AAAAMMJJHH'].iloc[i]
        fin = df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Date Début': debut, 'Date Fin': fin})
    return pd.DataFrame(periods)

def Nbr_rockfall_surHP_LIFT(x, df, df_sismo, y):
  #slicing à partir des dates disponibles pour le catalogue sismo
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique
  date_ref = date_obj - pd.Timedelta(days=x)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)
  #filtrer les données de temp pour la période de référence
  df.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df.reset_index(inplace=True)

  #trouver les périodes de gel et pas de gel
  periods_2013 = trouver_periodes_HR_gel_pas_gel(periods_df, x)
  periods_2013.set_index('Date Début', inplace=True)

  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  rockfall_HR_days_after = []

  for date in periods_2013.index:
    start_date = date + pd.Timedelta(days=x)
    end_date = date + pd.Timedelta(days=x + y - 1)

    #filtrer les événements dans la période choisie
    df_sismo_filtered = df_sismo.loc[start_date:end_date]

    #verifier si on a des événements rockfall (type 'R')
    rockfall_events = df_sismo_filtered[df_sismo_filtered['type'] == 'R']

    rockfall_detected = False

    if not rockfall_events.empty:  # Si des rockfalls existent
      rockfall_detected = True
    else:
      rockfall_detected = False

    rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods_2013.copy()
  resultat['rockfall sur période'] = rockfall_HR_days_after
  nbr_rockfall_HP = resultat['rockfall sur période'].sum()
  nbr_jours=len(resultat)

  return nbr_rockfall_HP, nbr_jours




def worker(task):
  """traite un couple (HR, HP)"""
  i, j, HR, HP = task
  proba_gel, nbr_rockfall, nbr_no_rockfall = probability_gel(HR, df_T, df_sismo, HP)

  _, nbr_R_pasgel, nbr_pasR_pasgel  = proba_pas_gel(HR, df_T, df_sismo, HP)

  nbr_rockfall_HP, nbr_observables = Nbr_rockfall_surHP_LIFT(HR, df_Daily_T, df_sismo, HP)

  proba_R = nbr_rockfall_HP / nbr_observables if nbr_observables > 0 else 0
  lift_gel = proba_gel / proba_R if proba_R > 0 else 0   

  print(f"(HR={HR}, HP={HP}) terminé")
  return i, j, proba_gel, nbr_rockfall, nbr_no_rockfall, nbr_R_pasgel, nbr_pasR_pasgel, lift_gel, proba_R


if __name__ == "__main__":

    df_T= pd.read_csv("/mnt/SSD1/bouazizs/GradCam/Probabilities/1.0_ T_St_Hiliaire_filled_complete.csv", delimiter='\t')
    df_T['AAAAMMJJHH'] = pd.to_datetime(df_T['AAAAMMJJHH'])
    df_T["AAAAMMJJHH"] = df_T["AAAAMMJJHH"].dt.date
    df_T['AAAAMMJJHH'] = pd.to_datetime(df_T['AAAAMMJJHH'])
    #df_T.set_index('AAAAMMJJHH', inplace=True)

    daily_T = df_T.groupby("AAAAMMJJHH")[" T"].apply(lambda x: (x < 0).any())
    df_Daily_T = daily_T.reset_index(name="negative_temp")


    df_sismo= pd.read_csv("/mnt/SSD1/bouazizs/GradCam/Probabilities/ev_sismo2.csv")
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    #on regarde que les types de "R"
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])

    #conserver que les dates communes
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_T['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_T['AAAAMMJJHH'].max())

    #filtrer les DataFrames pour ne conserver que les dates communes
    df_sismo = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_T = df_T[(df_T['AAAAMMJJHH']>= start_date) & (df_T['AAAAMMJJHH'] <= end_date)]

    # File to save results
    outputFolder="/mnt/SSD1/bouazizs/GradCam/Probabilities_rockfall_conditions/continuous_freeze_filled"
    os.makedirs(outputFolder, exist_ok=True)

    #calcul des probabilités pour les valeurs de HR et HP
    HR_values = range(1, 15)
    HP_values = range(1, 15)
    proba_values_gel = np.zeros((len(HR_values), len(HP_values)))

    nombres_rockfall_apres_gel=np.zeros((len(HR_values), len(HP_values)))
    nombres_rockfall_apres_pas_gel= np.zeros((len(HR_values), len(HP_values)))

    nombres_no_rockfall_apres_gel = np.zeros((len(HR_values), len(HP_values)))
    nombres_no_rockfall_apres_pas_gel = np.zeros((len(HR_values), len(HP_values)))

    lift=np.zeros((len(HR_values), len(HP_values)))
    proba_r=np.zeros((len(HR_values), len(HP_values)))

    tasks = [(i, j, HR, HP) for i, HR in enumerate(HR_values) for j, HP in enumerate(HP_values)]

    num_processes = int(multiprocessing.cpu_count()*0.8)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(worker, tasks)


    for result in results:
      i, j, proba_gel, nbr_rockfall, nbr_no_rockfall, nbr_R_pasgel, nbr_pasR_pasgel, lift_gel, proba_R = result
      proba_values_gel[i, j] = proba_gel
      nombres_rockfall_apres_gel[i, j] = nbr_rockfall
      nombres_no_rockfall_apres_gel[i, j] = nbr_no_rockfall
      nombres_rockfall_apres_pas_gel[i, j] = nbr_R_pasgel
      nombres_no_rockfall_apres_pas_gel[i, j] = nbr_pasR_pasgel
      lift[i,j]= lift_gel
      proba_r[i,j]= proba_R

 
    proba_R_G= "p(rHP|gHR).txt"
    np.savetxt(os.path.join(outputFolder, proba_R_G), proba_values_gel)
    np.savetxt(os.path.join(outputFolder,'nombres_rockfall_apres_gel.txt'), nombres_rockfall_apres_gel)
    np.savetxt(os.path.join(outputFolder,'nombres_no_rockfall_apres_gel.txt'), nombres_no_rockfall_apres_gel)
    np.savetxt(os.path.join(outputFolder,'nombres_rockfall_apres_pas_gel.txt'), nombres_rockfall_apres_pas_gel)
    np.savetxt(os.path.join(outputFolder,'nombres_no_rockfall_apres_pas_gel.txt'), nombres_no_rockfall_apres_pas_gel)
    np.savetxt(os.path.join(outputFolder, 'lift.txt'), lift)
    np.savetxt(os.path.join(outputFolder, 'proba_rockfall_global.txt'), proba_r)
    
    #Compute and save the P-values and margin of error 
    HR=14
    HP=14
    margin_of_error= calcul_IC(nombres_rockfall_apres_gel, nombres_no_rockfall_apres_gel, HR , HP)
    p_values, decisions = compute_chi2(nombres_rockfall_apres_gel,nombres_no_rockfall_apres_gel,
                                       nombres_rockfall_apres_pas_gel,nombres_no_rockfall_apres_pas_gel, HR_values, HP_values )
    np.savetxt(os.path.join(outputFolder, f'Khi-deux_gel-Rockfall.txt'), p_values)
    np.savetxt(os.path.join(outputFolder, f'margin_of_error_gel_Rockfall.txt'), margin_of_error)


    # --------Plot figure of probability and aggregation metrics
    files=os.listdir(outputFolder)
    for f in files:
      if f.endswith(".txt") :   #and f.startswith('LIFT_pluie')
        print(f)
        # Plot the figure if it is a probability file
        if "p(r" in f:
          plt.rcParams.update({'axes.labelsize': 14,    # Taille du texte des labels des axes
                        'axes.titlesize': 14,   # Taille du texte du titre des axes
                        'xtick.labelsize': 14,  # Taille du texte des labels des ticks X
                        'ytick.labelsize': 14,  # Taille du texte des labels des ticks Y
                        'font.size': 13,        # Taille générale du texte
                        'legend.fontsize': 15})
          plot_fig_proba(outputFolder , os.path.join(outputFolder, f), HR_values, HP_values )


    # --------Plot figure of aggregation 
    #figure size config :
    plt.rcParams.update({'axes.labelsize': 18,    # Taille du texte des labels des axes
                'axes.titlesize': 18,   # Taille du texte du titre des axes
                'xtick.labelsize': 18,  # Taille du texte des labels des ticks X
                'ytick.labelsize': 18,  # Taille du texte des labels des ticks Y
                'font.size': 16,        # Taille générale du texte
                'legend.fontsize': 18})

    plot_aggregation( LIFT=lift, p_values=p_values, margin_of_error=margin_of_error, outputFolder=outputFolder,
                     filename="Aggregation_freeze.png", HR=np.arange(1, 15), HP=np.arange(1, 15),
                       )








