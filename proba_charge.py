import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import os 
import time 
import multiprocessing
import multiprocessing as mp
import math
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from utilis import plot_fig_proba , compute_chi2 , calcul_IC, plot_probability_heatmaps_levels, compute_ci_level_matrix , plot_aggregation_heatmaps_levels


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


def charge(dfpluie, lamb):
  pluv = dfpluie['pluvio'].values  
  H = np.zeros_like(pluv) 
  Dt = 1 / 24  
  for p in range(1, len(pluv)):
      H[p] = H[p-1] * np.exp(-Dt / lamb) + pluv[p]
  return H


def trouver_periodes_H_threshold(df, low_th, high_th, check):
  periods = []
  if check=='low':
    df_H=df[df['H']< low_th]
  elif check=='high':
    df_H=df[df['H']>= high_th]
  elif check=='tot':
    df_H=df[df['H']> low_th]
  else:
    df_H=df[(df['H']>=low_th) &  (df['H']< high_th )]

  df_H=df_H[['AAAAMMJJHH', 'H']]

  return df_H

def Probability ( df_charge, df_sismo, y, low_th, high_th, check='high'):     #periodes HR (pluie ou non selon condition)
  #slicing à partir des dates disponibles pour le catalogue sismo
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique

  date_ref = date_obj 
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  #filtrer les données de précipitations pour la période de référence
  df_charge.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_charge.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_charge.reset_index(inplace=True)
  #touver les périodes de charge selon condition (low, or meduim or high)
  periods_2013 = trouver_periodes_H_threshold(periods_df, low_th, high_th, check)
  periods_2013.set_index('AAAAMMJJHH', inplace=True)

  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  #initialisation
  rockfall_after_H = []

  #parcourir les dates de  charge/décharge
  for date in periods_2013.index:
    #verifier s'il y a un rockfall dans les y heures suivant la date de charge/decharge de pluie
    end_date = date + pd.Timedelta(days= y )
    rockfall_condition = df_sismo.loc[date + pd.Timedelta(days= 1) :end_date]['type'] == 'R'
    rockfall_detected = rockfall_condition.any()
    rockfall_after_H.append(1 if rockfall_detected else 0)


  resultat = periods_2013.copy()
  resultat['rockfall sur période'] = rockfall_after_H
  proba = resultat['rockfall sur période'].mean()   #(nbr periodes de H suivie de rockfall / nbr de periodes de charge selon condtion(low/high/meduim))

  nbr= resultat['rockfall sur période'].sum()
  nbr_no=len(resultat)-nbr

  return proba, nbr, nbr_no


def trouver_periodes_charge_pluie_pas_charge(df, n):
  periods = []
  for i in range(len(df) - n + 1):
    debut = df['AAAAMMJJHH'].iloc[i]
    fin = df['AAAAMMJJHH'].iloc[i+n-1]
    periods.append({'Date Début': debut, 'Date Fin': fin})
  return pd.DataFrame(periods)
    
def Nbr_rockfall_surHP_LIFT(df_charge, df_sismo, y):
  ''' df_charge ici c'est df_precip '''
  #slicing à partir des dates disponibles pour le catalogue sismo
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique

  date_ref = date_obj - pd.Timedelta(days=1)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  #filtrer les données de précipitations pour la période de référence
  df_charge.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_charge.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_charge.reset_index(inplace=True)

  #trouver les périodes de charge ou non possibles --> ici on prend tt simplement df_H ! 
  #periods_2013 = trouver_periodes_charge_pluie_pas_charge(periods_df)
  periods_2013=periods_df
  periods_2013.set_index('AAAAMMJJHH', inplace=True)

  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  #initialisation
  rockfall_HR_days_after = []

  #parcourir les dates de précipitations
  for date in periods_2013.index:
      #vérifier s'il y a un rockfall dans les y jours suivant la date de précipitation
      end_date = date + pd.Timedelta(days=y)
      rockfall_condition = df_sismo.loc[date + pd.Timedelta(days=1):end_date]['type'] == 'R'
      rockfall_detected = rockfall_condition.any()

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods_2013.copy()
  resultat['rockfall sur période'] = rockfall_HR_days_after
  nbr_rockfall_HP_apres_H_poss = resultat['rockfall sur période'].sum()
  nbr_jours=len(resultat)
  nbr_no_rockfall_HP_apres_H_poss= nbr_jours- nbr_rockfall_HP_apres_H_poss

  return nbr_rockfall_HP_apres_H_poss, nbr_no_rockfall_HP_apres_H_poss, nbr_jours



def worker(task):
  i, HP = task
  proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall = Probability( df_precip, df_sismo, HP, low_threshold, high_threshold, intensity) 

  nbr_periodes_rockfall_HP_apres_Hposs, nbr_periodes_no_rockfall_HP_apres_Hposs , nbr_observables_H_HP = Nbr_rockfall_surHP_LIFT( df_precip, df_sismo, HP)

  #proba_R = nbr_periodes_rockfall_HP / nbr_observables_H_HP if nbr_observables_H_HP > 0 else 0
  
  #lift = proba_H_R / proba_R if proba_R > 0 else 0 

  print(f"(HP={HP}) terminé")

  return i, proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall , nbr_periodes_rockfall_HP_apres_Hposs , nbr_periodes_no_rockfall_HP_apres_Hposs, nbr_observables_H_HP  #, nbr_pasP_rockfall, nbr_pasP_no_rockfall



if __name__ == "__main__":
    df_sismo= pd.read_csv("/mnt/SSD1/bouazizs/GradCam/Probabilities/ev_sismo2.csv")
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    #pour avoir les dates sismo en jour
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    df = pd.read_csv('/mnt/SSD1/bouazizs/GradCam/Probabilities/data_meteo_Sabrine2.csv', delimiter=';')
    df_precip0 = pd.read_csv('/mnt/SSD1/bouazizs/GradCam/Probabilities/1.0_RR1.csv', delimiter='\t')
    df['AAAAMMJJHH'] = df_precip0['AAAAMMJJHH'].values
    df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH'])

    lamb = 0.2
    print(f'traitement for lambda = {lamb}')
    df['H'] = charge(df,lamb) 

    #en prenant la valeur max dans la journée : 
    df['date'] = df['AAAAMMJJHH'].dt.date
    #l'index ou H est maximal pour chaque jour
    idx = df.groupby('date')['H'].idxmax()
    #garder uniquement ces lignes
    df_precip = df.loc[idx].copy().reset_index(drop=True)
    df_precip['AAAAMMJJHH']=df_precip['AAAAMMJJHH'].dt.date
    df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])


    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_precip['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_precip['AAAAMMJJHH'].max())
    df_sismo = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]

    outputFolder="/mnt/SSD1/bouazizs/GradCam/Probabilities_rockfall_conditions/Charge-discharge"
    os.makedirs(outputFolder, exist_ok=True)

    #pour seuils :
    lth=5
    low_threshold = np.percentile(df_precip['H'] , lth)
    high_threshold= 5  

    HP_values = range(1, 15)
    HR_values = range(1, 2)

    conditional=['high', 'medium','low', 'global']

    for intensity in conditional :
        print(f'---------- intensity = {intensity}---------------')
        proba1_values_H= np.zeros((1, len(HP_values)))
        proba1_values_pas_H = np.zeros((1, len(HP_values)))

        nombres_rockfall_apres_H=np.zeros((1, len(HP_values)))
        nombres_rockfall_apres_pas_H=np.zeros((1, len(HP_values)))

        nombres_no_rockfall_apres_H=np.zeros((1, len(HP_values)))
        nombres_no_rockfall_apres_pas_H=np.zeros((1, len(HP_values)))

        nbr_R_HP_H_possible= np.zeros((1, len(HP_values)))
        nbr_No_R_HP_H_possible= np.zeros((1, len(HP_values)))

        LIFT_H=np.zeros((1, len(HP_values)))

        
        tasks = [(i, HP) for i, HP in enumerate(HP_values)]

        start_time= time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(worker, tasks)


        for result in results:
            i, proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall , nbr_periodes_rockfall_HP_apres_Hposs , nbr_periodes_no_rockfall_HP_apres_Hposs, nbr_observables_H_HP  = result 
            proba1_values_H[:,i] = proba_H_R
            nombres_rockfall_apres_H[:,i] = nbr_H_rockfall
            nombres_no_rockfall_apres_H[:,i] = nbr_H_no_rockfall
                
            proba_R = nbr_periodes_rockfall_HP_apres_Hposs / nbr_observables_H_HP
            lift= proba_H_R / proba_R if proba_R > 0 else 0 

            LIFT_H[:, i] = lift 

            nbr_R_HP_H_possible[:,i] = nbr_periodes_rockfall_HP_apres_Hposs
            nbr_No_R_HP_H_possible[:,i] = nbr_periodes_no_rockfall_HP_apres_Hposs


            
        file_proba = f'p(R|H avec H = {intensity})_daily.txt'
        np.savetxt(os.path.join(outputFolder, file_proba), proba1_values_H)
        np.savetxt(os.path.join(outputFolder, f'nombres_rockfall_apres_H = {intensity}.txt'), nombres_rockfall_apres_H)
        np.savetxt(os.path.join(outputFolder, f'nombres_no_rockfall_apres_H = {intensity}.txt'), nombres_no_rockfall_apres_H)
        np.savetxt(os.path.join(outputFolder, f'LIFT_H = {intensity}.txt'), LIFT_H)
        
        margin_of_error = calcul_IC(nombres_rockfall_apres_H, nombres_no_rockfall_apres_H, 1 , len(HP_values))
        np.savetxt(os.path.join(outputFolder, f'margin_of_error_p(R|H avec H = {intensity}).txt'), margin_of_error)


        #Compute de KHI-DEUX 
        n11=nombres_rockfall_apres_H
        n10=nombres_no_rockfall_apres_H
        n01= nbr_R_HP_H_possible - n11
        n00= nbr_No_R_HP_H_possible -n10
        p_value_test_khi, d = compute_chi2 ( n11, n10, n01 , n00, HR_values, HP_values)
        np.savetxt(os.path.join(outputFolder, f'p_value H = {intensity}.txt'), p_value_test_khi)


    plot_probability_heatmaps_levels( outputFolder, modality="H", conditions=conditional,
                                     HP_values=range(1, 15), filename="proba_charge.png" )
    


    plot_aggregation_heatmaps_levels( outputFolder=outputFolder, modality= "H", conditions=conditional, HP_values=range(1, 15),
                            compute_ci_level_matrix=compute_ci_level_matrix , filename="Aggregation_charge.png")
