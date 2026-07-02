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
from utilis import compute_chi2 , calcul_IC, plot_probability_heatmaps_levels, compute_ci_level_matrix, plot_aggregation_heatmaps_levels

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def simulate_temperature(file, Xmax=8, dx=0.08, dt=3600, lamb=1.5, C=800, rho=2500,  factor=2): 

  with open(file) as file_name:
      dataTemp = np.loadtxt(file_name, delimiter=",",skiprows=1)

  tTemp=dataTemp[:,0]+dataTemp[:,1] #unité = jour, pas =1 h
  dt=tTemp[1]-tTemp[0]
  Temp=dataTemp[:,2]
  tTemp=tTemp-tTemp[0]
  print(len(tTemp),len(Temp))
  #surechantillonnage pour permettre plus de detail en profondeur en conservant r<0
  tTempNew=np.arange(min(tTemp),max(tTemp)+dt/factor,dt/factor)
  print(len(tTemp),len(tTempNew))
  #TempNew=

  iall=len(tTemp)
  print(iall)
  imax=iall
  tSim=tTemp[0:iall]*24*60*60 #convert to seconds
  TSim=Temp[0:iall]
  plt.plot(tSim/365/24/3600,TSim,'-')
  plt.xlabel('Time (years)')
  plt.ylabel('T (°C)')
  plt.title('Air Temperature measured close to St Eynard')
  plt.show()

  #paramètre numeriques
  tpsMax=max(tSim)
  Nx = int(Xmax/dx) # Nombre de nœuds de la grille spatiale
  dt =  60*60 #pas de calcul temporel en s

  #Nt = int(tpsMax/dt) # Nombre d'itérations temporelles
  Nt = len(TSim) - 1  #--> donne 124374

  # Paramètre de stabilité
    #Propriétés thermiques
  lamb=1.5 #conductivité thermique W/m.K, calcaire compact
  C=800      #chaleur specifique J/kg
  rho=2500     #masse volumique kg/m3
  alpha = lamb/(rho*C) # diffusivité thermique
  r = alpha*dt/dx**2
  print('Paramètre de stabilité=',r,' doit être inférieur à 0.5 pour converger')

  # Discrétisation spatiale
  x = np.linspace(0, Xmax, Nx+1) # Grille de nœuds
  # Espacement entre les nœuds

  # Discrétisation temporelle
  t = np.linspace(0, tpsMax, Nt+1) # Grille de temps

  # stockage des profils de temperature à chaque pas de temps
  TempXTime=np.zeros((len(x),len(t)))

  #conditions aux limites
  # Température imposée à l'extrémité gauche du model
  # Ttemperature paroi variable selon mesures
  T_left=TSim
  T_right=np.mean(TSim) #condition en profondeur
  print('T en profondeur',T_right)

  # Conditions initiales
  u0 = np.zeros((Nx+1))+T_right # Profil initial de température
  u = u0

  # Condition de Dirichlet à l'extrémité gauche = surface libre
  u[0] = T_left[0];
  u[Nx]=T_right
  TempXTime[:,0]=u[:]
  # Boucle temporelle
  for n in range(Nt):
      # Calcul de la solution au temps suivant
      u_new = np.zeros((Nx+1));
      for i in range(1,Nx):
          u_new[i] = u[i] + r*(u[i-1] - 2*u[i] + u[i+1])

      # Conditions aux limites de Dirichlet
      u_new[0] = T_left[n]; #T=fn(t) en surface
      u_new[-1] = T_right #T=cte en profondeur

      # Mise à jour de la solution courante
      u = u_new
      #stockage
      TempXTime[:,n]=u


  #profondeur de gel
  pgel=np.zeros(len(TempXTime[0,:]))
  print(len(pgel))
  for p in range(len(pgel)):
      profilT=TempXTime[:,p]  #pour chaque temps pas temporelle
      igel=np.where(profilT<0)[0]   #on cherche ou T<0 ,
      if len(igel)>0:                #s'i l y a des temp neg, on prends la profendeur d'indice maximale (ekher pronf fih temp<0 yaani aghrek wehed) , sino 0
          pgel[p]=x[max(igel)]
      else:
          pgel[p]=0
  plt.figure(2)
  fig=plt.plot(t/3600/24/365,pgel)
  plt.xlabel('Time')
  plt.ylabel('Depth(T<0) m')
  plt.show()
  return pgel


def find_periodes_depeth_freeze(df, thresh1min, check):
    df = df.copy()
    df["AAAAMMJJHH"] = df["AAAAMMJJHH"].dt.date

    # The maximum depth (max value of 'pgel') per day
    df_daily_max = df.groupby("AAAAMMJJHH")["pgel"].max().reset_index(name="max_pgel")

    if check == 'low':
        # Days are considered low when the maximum depth is < threshold and > 0
        df_daily_max["pgel"] = (df_daily_max["max_pgel"] < thresh1min) & (df_daily_max["max_pgel"] > 0)

    elif check == 'high':
        # high when the maximum depth is >= threshold
        df_daily_max["pgel"] = df_daily_max["max_pgel"] >= thresh1min
    elif check == 'global':
        df_daily_max["pgel"] = df_daily_max["max_pgel"] > 0
    else:
        raise ValueError("Check the level of pgel!!")

    # Keep only valid days
    df_pgel = df_daily_max[df_daily_max["pgel"] == True][["AAAAMMJJHH", "pgel"]]

    return df_pgel


def Probability ( df_gel, df_sismo, y, thr1, check='high'):
  # Slicing from available dates for the seismic catalog
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])   # start of seismic
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # end of seismic

  date_ref = date_obj
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  # Filtering freeze depth data for the reference period
  df_gel.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_gel.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_gel.reset_index(inplace=True)


  # Filtering freeze depth data for the reference period depending on the condition "check" (low, high or global)
  periods = find_periodes_depeth_freeze(periods_df, thr1, check)
  periods.set_index('AAAAMMJJHH', inplace=True)

  df_sismo = df_sismo.copy()
  df_sismo.loc[:, 'Date']= pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  # Initialisation
  rockfall_after_H = []

  for date in periods.index:
    # Verify if there is a rockfall in the y hours following the freeze depth date
    end_date = date + pd.Timedelta(days= y )
    rockfall_condition = df_sismo.loc[date + pd.Timedelta(days= 1) :end_date]['type'] == 'R'
    rockfall_detected = rockfall_condition.any()
    rockfall_after_H.append(1 if rockfall_detected else 0)


  resultat = periods.copy()
  resultat['rockfall sur période'] = rockfall_after_H
  proba = resultat['rockfall sur période'].mean()   #(nbr of periodes of depth freeze followed by rockfall / nbr of periodes of depth freeze depending on condtion(low/high/global))
  nbr= resultat['rockfall sur période'].sum()
  nbr_no=len(resultat)-nbr

  return proba, nbr, nbr_no


def Nbr_rockfall_onHP_LIFT(df_T_daily, df_sismo, y):
  # Slicing from available dates for the seismic catalog
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  

  date_ref = date_obj - pd.Timedelta(days=1)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  # Filtering freeze depth data for the reference period
  df_T_daily.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_T_daily.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_T_daily.reset_index(inplace=True)


  periods=periods_df
  periods.set_index('AAAAMMJJHH', inplace=True)

  df_sismo = df_sismo.copy()
  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  # Initialisation
  rockfall_HR_days_after = []

  # Browse the dates of freeze depth 
  for date in periods.index:
      # Verify if there is a rockfall in the y days following the freeze depth date
      end_date = date + pd.Timedelta(days=y)
      rockfall_condition = df_sismo.loc[date + pd.Timedelta(days=1):end_date]['type'] == 'R'
      rockfall_detected = rockfall_condition.any()

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods.copy()
  resultat['rockfall sur période'] = rockfall_HR_days_after
  nbr_rockfall_HP_apres_H_poss = resultat['rockfall sur période'].sum()
  nbr_jours=len(resultat)
  nbr_no_rockfall_HP_apres_H_poss= nbr_jours- nbr_rockfall_HP_apres_H_poss

  return nbr_rockfall_HP_apres_H_poss, nbr_no_rockfall_HP_apres_H_poss, nbr_jours



def worker(task):
  i, HP = task
  proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall = Probability( df, df_sismo, HP, threshold1, intensity)

  nbr_periods_rockfall_HP_after_possible_depth_freeze, nbr_periods_no_rockfall_HP_after_possible_depth_freeze , nbr_observables_depth_freeze_HP = Nbr_rockfall_onHP_LIFT( df_Daily_T, df_sismo, HP)

  print(f"(HP={HP}) finished")

  return i, proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall , nbr_periods_rockfall_HP_after_possible_depth_freeze , nbr_periods_no_rockfall_HP_after_possible_depth_freeze, nbr_observables_depth_freeze_HP  



if __name__ == "__main__":

    df_sismo= pd.read_csv("/Data/ev_sismo2.csv")
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    # For having daily sismo dates
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])

    df = pd.read_csv('/Data/data_Temp_st-H.csv', delimiter=',')
    df_T = pd.read_csv('/Data/1.0_ T_St_Hiliaire_filled_complete.csv', delimiter='\t')

    df['AAAAMMJJHH'] = df_T['AAAAMMJJHH'].values
    df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH']) 

    pgel = simulate_temperature('/Data/data_Temp_st-H.csv')
    df["pgel"] = pgel  
   
    # For having daily temperature dates
    df_T['AAAAMMJJHH'] = pd.to_datetime(df_T['AAAAMMJJHH'])
    df_T = df_T.copy()
    df_T["AAAAMMJJHH"] = df_T["AAAAMMJJHH"].dt.date
    df_T['AAAAMMJJHH'] = pd.to_datetime(df_T['AAAAMMJJHH'])

    daily_T = df_T.groupby("AAAAMMJJHH")[" T"].apply(lambda x: (x < 0).any())
    df_Daily_T = daily_T.reset_index(name="negative_temp")

    start_date = max(df_sismo['AAAAMMJJHH'].min(), df['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df['AAAAMMJJHH'].max())
    df_sismo = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]

    outputFolder="/Probabilities_rockfall_conditions/Freeze_depth"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Define size of retrospective horizon and forecasting horizon     
    HR_values = range(1, 2)
    HP_values = range(1, 15)


    threshold1= 0.16

    conditional=[ 'global', 'low', 'high']
    for intensity in conditional:
        print(f'---------- intensity = {intensity}---------------')
        proba1_values_depthfreeze= np.zeros((1, len(HP_values)))
        proba1_values_pas_depthfreeze = np.zeros((1, len(HP_values)))
        numbers_rockfall_after_depthfreeze=np.zeros((1, len(HP_values)))
        nombres_rockfall_apres_pas_depthfreeze=np.zeros((1, len(HP_values)))

        numbers_no_rockfall_after_depthfreeze=np.zeros((1, len(HP_values)))
        nombres_no_rockfall_apres_pas_depthfreeze=np.zeros((1, len(HP_values)))

        nbr_R_HP_depth_freeze_possible= np.zeros((1, len(HP_values)))
        nbr_No_R_HP_depth_freeze_possible= np.zeros((1, len(HP_values)))

        LIFT_H=np.zeros((1, len(HP_values)))

        tasks = [(i, HP) for i, HP in enumerate(HP_values)]

        start_time= time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(worker, tasks)

        for result in results:
            i, proba_depthfreeze_R, nbr_depthfreeze_rockfall, nbr_depthfreeze_no_rockfall , nbr_periods_rockfall_HP_after_possible_depthfreeze , nbr_periods_no_rockfall_HP_after_possible_depthfreeze, nbr_observables_depthfreeze_HP  = result
            proba1_values_depthfreeze[:,i] = proba_depthfreeze_R
            numbers_rockfall_after_depthfreeze[:,i] = nbr_depthfreeze_rockfall
            numbers_no_rockfall_after_depthfreeze[:,i] = nbr_depthfreeze_no_rockfall

            proba_R = nbr_periods_rockfall_HP_after_possible_depthfreeze / nbr_observables_depthfreeze_HP
            lift= proba_depthfreeze_R / proba_R if proba_R > 0 else 0

            LIFT_H[:, i] = lift

            nbr_R_HP_depth_freeze_possible[:,i] = nbr_periods_rockfall_HP_after_possible_depthfreeze
            nbr_No_R_HP_depth_freeze_possible[:,i] = nbr_periods_no_rockfall_HP_after_possible_depthfreeze

        file_proba = f'p(R|pgel with pgel = {intensity})_daily.txt'
        np.savetxt(os.path.join(outputFolder, file_proba), proba1_values_depthfreeze)
        np.savetxt(os.path.join(outputFolder, f'numbers_rockfall_after_depthfreeze = {intensity}.txt'), numbers_rockfall_after_depthfreeze)
        np.savetxt(os.path.join(outputFolder, f'numbers_no_rockfall_after_depthfreeze = {intensity}.txt'), numbers_no_rockfall_after_depthfreeze)
        np.savetxt(os.path.join(outputFolder, f'LIFT_pgel = {intensity}.txt'), LIFT_H)

        margin_of_error = calcul_IC(numbers_rockfall_after_depthfreeze, numbers_no_rockfall_after_depthfreeze, 1 , len(HP_values))
        np.savetxt(os.path.join(outputFolder, f'margin_of_error_p(R|pgel with pgel = {intensity}).txt'), margin_of_error)

        # Compute de KHI-DEUX
        n11=numbers_rockfall_after_depthfreeze
        n10=numbers_no_rockfall_after_depthfreeze
        n01= nbr_R_HP_depth_freeze_possible - n11
        n00= nbr_No_R_HP_depth_freeze_possible -n10
        p_value_test_khi,d = compute_chi2 (n11, n10, n01 , n00, HR_values, HP_values)
        np.savetxt(os.path.join(outputFolder, f'p_value freeze depth = {intensity}.txt'), p_value_test_khi)


    plot_probability_heatmaps_levels( outputFolder, modality="pgel", conditions=conditional,
                                     HP_values=range(1, 15), filename="proba_DEPTH_FREEZE.png" )
    
    plot_aggregation_heatmaps_levels( outputFolder=outputFolder, modality="pgel", conditions=conditional, HP_values=range(1, 15),
                            compute_ci_level_matrix=compute_ci_level_matrix , filename="Aggregation_freeze_depth.png")
