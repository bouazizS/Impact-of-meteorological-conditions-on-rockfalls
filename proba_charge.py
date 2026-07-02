import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import os 
import time 
import multiprocessing as mp
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from utilis import  compute_chi2 , calcul_IC, plot_probability_heatmaps_levels, compute_ci_level_matrix , plot_aggregation_heatmaps_levels


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


def find_periods_H_threshold(df, low_th, high_th, check):
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

def Probability ( df_charge, df_sismo, y, low_th, high_th, check='high'):   
  # Slicing using dates available in the seismic catalog
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  

  date_ref = date_obj 
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  # Filter precipitation data for the reference period
  df_charge.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_charge.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_charge.reset_index(inplace=True)
  # Find periods of charge according to condition (low, medium or high)
  periods = find_periods_H_threshold(periods_df, low_th, high_th, check)
  periods.set_index('AAAAMMJJHH', inplace=True)

  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  # Initialisation
  rockfall_after_H = []

  # Browse the dates of charge/discharge
  for date in periods.index:
    # Verify if there is a rockfall in the y hours following the charge/discharge date
    end_date = date + pd.Timedelta(days= y )
    rockfall_condition = df_sismo.loc[date + pd.Timedelta(days= 1) :end_date]['type'] == 'R'
    rockfall_detected = rockfall_condition.any()
    rockfall_after_H.append(1 if rockfall_detected else 0)


  resultat = periods.copy()
  resultat['rockfall on period'] = rockfall_after_H
  proba = resultat['rockfall on period'].mean()   #(nbr of periods of H follwed de rockfall / nbr de periodes de charge selon condtion(low/high/meduim))

  nbr= resultat['rockfall on period'].sum()
  nbr_no=len(resultat)-nbr

  return proba, nbr, nbr_no


def find_period_charge_non_charge(df, n):
  periods = []
  for i in range(len(df) - n + 1):
    debut = df['AAAAMMJJHH'].iloc[i]
    fin = df['AAAAMMJJHH'].iloc[i+n-1]
    periods.append({'Date Début': debut, 'Date Fin': fin})
  return pd.DataFrame(periods)
    
def Nbr_rockfall_onHP_LIFT(df_charge, df_sismo, y):
  # Slicing from available seismic catalog dates
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0]) 
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1]) 

  date_ref = date_obj - pd.Timedelta(days=1)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  # Filter precipitation data for the reference period
  df_charge.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_charge.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_charge.reset_index(inplace=True)

  # Find possible periods of charge or non-charge
  periods = periods_df
  periods.set_index('AAAAMMJJHH', inplace=True)

  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  # Initialisation
  rockfall_HR_days_after = []

  # Browse the dates of precipitation
  for date in periods.index:
      # Check if there is a rockfall in the y days following the precipitation date
      end_date = date + pd.Timedelta(days=y)
      rockfall_condition = df_sismo.loc[date + pd.Timedelta(days=1):end_date]['type'] == 'R'
      rockfall_detected = rockfall_condition.any()

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods.copy()
  resultat['rockfall on period'] = rockfall_HR_days_after
  nbr_rockfall_HP_apres_H_poss = resultat['rockfall on period'].sum()
  nbr_jours=len(resultat)
  nbr_no_rockfall_HP_apres_H_poss= nbr_jours- nbr_rockfall_HP_apres_H_poss

  return nbr_rockfall_HP_apres_H_poss, nbr_no_rockfall_HP_apres_H_poss, nbr_jours



def worker(task):
  i, HP = task
  proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall = Probability( df_precip, df_sismo, HP, low_threshold, high_threshold, intensity) 

  nbr_periodes_rockfall_HP_apres_Hposs, nbr_periodes_no_rockfall_HP_apres_Hposs , nbr_observables_H_HP = Nbr_rockfall_onHP_LIFT( df_precip, df_sismo, HP)

  print(f"(HP={HP}) terminé")

  return i, proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall , nbr_periodes_rockfall_HP_apres_Hposs , nbr_periodes_no_rockfall_HP_apres_Hposs, nbr_observables_H_HP  #, nbr_pasP_rockfall, nbr_pasP_no_rockfall



if __name__ == "__main__":
    df_sismo= pd.read_csv("/Data/ev_sismo2.csv")
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    # For having the seismic dates in day format
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])

    df = pd.read_csv('/Data/data_meteo.csv', delimiter=';')
    df_precip0 = pd.read_csv('/Data/1.0_RR1.csv', delimiter='\t')
    df['AAAAMMJJHH'] = df_precip0['AAAAMMJJHH'].values
    df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH'])

    lamb = 0.2
    print(f'traitement for lambda = {lamb}')
    df['H'] = charge(df,lamb) 

    # We take the maximum value for each day
    df['date'] = df['AAAAMMJJHH'].dt.date
    # The index where H is maximal for each day
    idx = df.groupby('date')['H'].idxmax()
    #  just these rows 
    df_precip = df.loc[idx].copy().reset_index(drop=True)
    df_precip['AAAAMMJJHH']=df_precip['AAAAMMJJHH'].dt.date
    df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])

    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_precip['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_precip['AAAAMMJJHH'].max())
    df_sismo = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]

    outputFolder="/Probabilities_rockfall_conditions/Charge-discharge"
    os.makedirs(outputFolder, exist_ok=True)

    # Low and high thresholds
    lth=5
    low_threshold = np.percentile(df_precip['H'] , lth)
    high_threshold= 5  
 
    # Define size of retrospective horizon and forecasting horizon 
    HR_values = range(1, 2)
    HP_values = range(1, 15)

    conditional=['high', 'medium','low', 'global']

    for intensity in conditional :
        print(f'---------- intensity = {intensity}---------------')
        proba1_values_H= np.zeros((1, len(HP_values)))
        proba1_values_pas_H = np.zeros((1, len(HP_values)))

        numbers_rockfall_after_H=np.zeros((1, len(HP_values)))
        nombres_rockfall_apres_pas_H=np.zeros((1, len(HP_values)))

        numbers_no_rockfall_after_H=np.zeros((1, len(HP_values)))
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
            numbers_rockfall_after_H[:,i] = nbr_H_rockfall
            numbers_no_rockfall_after_H[:,i] = nbr_H_no_rockfall
                
            proba_R = nbr_periodes_rockfall_HP_apres_Hposs / nbr_observables_H_HP
            lift= proba_H_R / proba_R if proba_R > 0 else 0 

            LIFT_H[:, i] = lift 

            nbr_R_HP_H_possible[:,i] = nbr_periodes_rockfall_HP_apres_Hposs
            nbr_No_R_HP_H_possible[:,i] = nbr_periodes_no_rockfall_HP_apres_Hposs


            
        file_proba = f'p(R|H with H = {intensity})_daily.txt'
        np.savetxt(os.path.join(outputFolder, file_proba), proba1_values_H)
        np.savetxt(os.path.join(outputFolder, f'numbers_rockfall_after_H = {intensity}.txt'), numbers_rockfall_after_H)
        np.savetxt(os.path.join(outputFolder, f'numbers_no_rockfall_after_H = {intensity}.txt'), numbers_no_rockfall_after_H)
        np.savetxt(os.path.join(outputFolder, f'LIFT_H = {intensity}.txt'), LIFT_H)
        
        margin_of_error = calcul_IC(numbers_rockfall_after_H, numbers_no_rockfall_after_H, 1 , len(HP_values))
        np.savetxt(os.path.join(outputFolder, f'margin_of_error_p(R|H with H = {intensity}).txt'), margin_of_error)


        #Compute de KHI-DEUX 
        n11=numbers_rockfall_after_H
        n10=numbers_no_rockfall_after_H
        n01= nbr_R_HP_H_possible - n11
        n00= nbr_No_R_HP_H_possible -n10
        p_value_test_khi, d = compute_chi2 ( n11, n10, n01 , n00, HR_values, HP_values)
        np.savetxt(os.path.join(outputFolder, f'p_value H = {intensity}.txt'), p_value_test_khi)


    plot_probability_heatmaps_levels( outputFolder, modality="H", conditions=conditional,
                                     HP_values=range(1, 15), filename="proba_charge.png" )
    


    plot_aggregation_heatmaps_levels( outputFolder=outputFolder, modality= "H", conditions=conditional, HP_values=range(1, 15),
                            compute_ci_level_matrix=compute_ci_level_matrix , filename="Aggregation_charge.png")
