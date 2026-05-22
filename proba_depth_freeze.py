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

def simulate_temperature_( file_path, Xmax=8, dx=0.08, dt_sim=3600, lamb=1.5,
                          C=800, rho=2500,  factor=2, plot=False ):
    # Load temperature data 
    with open(file_path) as file_name:
        dataTemp = np.loadtxt(file_name, delimiter=",",skiprows=1)
    # Time reconstruction (in days) and temperature extraction
    tTemp = dataTemp[:, 0] + dataTemp[:, 1]
    Temp = dataTemp[:, 2]

    # Time step in the original data
    dt_data = tTemp[1] - tTemp[0]

    # Normalize time to start at zero
    tTemp = tTemp - tTemp[0]

    # resampling for higher temporal resolution 
    tTempNew = np.arange(min(tTemp), max(tTemp) + dt_data/factor, dt_data/factor)
    Temp = np.interp(tTempNew, tTemp, Temp)
    tTemp = tTempNew

    # Convert time to seconds
    iall=len(tTemp)
    tSim=tTemp[0:iall]*24*60*60 
    TSim=Temp[0:iall]
    tpsMax = max(tSim)

    # Spatial and temporal discretization 
    Nx = int(Xmax / dx)     # Number of spatial nodes
    Nt = len(TSim) - 1      # Number of time steps
    x = np.linspace(0, Xmax, Nx + 1)      # Spatial grid
    t = np.linspace(0, tpsMax, Nt + 1)    # Time grid

    # Physical parameters 
    alpha = lamb / (rho * C)      # Thermal diffusivity
    r = alpha * dt_sim / dx**2    # Stability parameter

    print("Stability parameter =", r)

    if r >= 0.5:
        print("scheme may be unstable (r >= 0.5)")

    # Initialization
    TempXTime = np.zeros((len(x), len(t)))  # Temperature field

    # Boundary conditions
    T_left = TSim                  # Surface temperature (time-dependent)
    T_right = np.mean(TSim)        # Deep ground temperature (constant)

    # Initial condition: uniform temperature
    u = np.zeros((Nx+1))+T_right 
    u[0] = T_left[0]
    u[Nx] = T_right
    TempXTime[:, 0] = u[:]

    # Time integration loop 
    for n in range(Nt):
        u_new = np.zeros(Nx + 1)

        for i in range(1, Nx):
            u_new[i] = u[i] + r * (u[i-1] - 2*u[i] + u[i+1])

        # Apply boundary conditions
        u_new[0] = T_left[n]   # Surface (Dirichlet)
        u_new[-1] = T_right    # Depth (Dirichlet)

        # Update solution
        u = u_new
        TempXTime[:, n] = u

    # Freezing depth computation (T < 0°C) 
    pgel=np.zeros(len(TempXTime[0,:]))
    for p in range(len(pgel)):
        profilT = TempXTime[:, p]
        igel = np.where(profilT < 0)[0]

        # If freezing occurs, take the deepest point
        if len(igel) > 0:
            pgel[p] = x[max(igel)]
        else:
            pgel[p] = 0

    if plot:
        vmin = min(TSim)
        vmax = max(TSim)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # Temperature field
        plt.figure()
        plt.contourf(t/3600/24/365, x, TempXTime, levels=100, cmap='bwr', norm=norm)
        plt.colorbar(label='Temperature (°C)')
        plt.xlabel('Time (years)')
        plt.ylabel('Depth (m)')
        plt.title('Temperature evolution in the ground')
        plt.show()

        # Freezing depth
        plt.figure()
        plt.plot(t/3600/24/365, pgel)
        plt.xlabel('Time (years)')
        plt.ylabel('Freezing depth (m)')
        plt.title('Evolution of freezing depth')
        plt.show()

    return pgel

def trouver_periodes_H_threshold_max(df, thresh1min, check):
    df = df.copy()
    df["AAAAMMJJHH"] = df["AAAAMMJJHH"].dt.date

    #la profondeur maximale (valeur max de 'pgel') par jour
    df_daily_max = df.groupby("AAAAMMJJHH")["pgel"].max().reset_index(name="max_pgel")

    if check == 'low':
        #la journée est considérée low si la profondeur maximale est < 0.1 et >0
        df_daily_max["pgel"] = (df_daily_max["max_pgel"] < thresh1min) & (df_daily_max["max_pgel"] > 0)

    elif check == 'high':
        #high si la profondeur maximale dépasse le 0.1
        df_daily_max["pgel"] = df_daily_max["max_pgel"] >= thresh1min
    elif check == 'global':
        df_daily_max["pgel"] = df_daily_max["max_pgel"] > 0
    else:
        raise ValueError("Check the level of pgel!!")

    # On ne garde que les journées valides
    df_pgel = df_daily_max[df_daily_max["pgel"] == True][["AAAAMMJJHH", "pgel"]]

    return df_pgel


def Probability ( df_gel, df_sismo, y, thr1, check='high'):
  #slicing à partir des dates disponibles pour le catalogue sismo
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique

  date_ref = date_obj
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  #filtrer les données de précipitations pour la période de référence
  df_gel.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_gel.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_gel.reset_index(inplace=True)


  #touver les périodes de depth freeze selon condition (low, or high or global)
  periods_2013 = trouver_periodes_H_threshold_max(periods_df, thr1, check)
  periods_2013.set_index('AAAAMMJJHH', inplace=True)

  df_sismo = df_sismo.copy()
  df_sismo.loc[:, 'Date']= pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  #initialisation
  rockfall_after_H = []

  for date in periods_2013.index:
    #verifier s'il y a un rockfall dans les y heures suivant la date de profondeur de gel 
    end_date = date + pd.Timedelta(days= y )
    rockfall_condition = df_sismo.loc[date + pd.Timedelta(days= 1) :end_date]['type'] == 'R'
    rockfall_detected = rockfall_condition.any()
    rockfall_after_H.append(1 if rockfall_detected else 0)


  resultat = periods_2013.copy()
  resultat['rockfall sur période'] = rockfall_after_H
  proba = resultat['rockfall sur période'].mean()   #(nbr periodes de pgel suivie de rockfall / nbr de periodes de pgel selon condtion(low/high/global))
  nbr= resultat['rockfall sur période'].sum()
  nbr_no=len(resultat)-nbr

  return proba, nbr, nbr_no


def Nbr_rockfall_surHP_LIFT(df_T_daily, df_sismo, y):
  ''' df_charge ici c'est df_precip '''
  #slicing à partir des dates disponibles pour le catalogue sismo
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique

  date_ref = date_obj - pd.Timedelta(days=1)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  #filtrer les données de précipitations pour la période de référence
  df_T_daily.set_index('AAAAMMJJHH', inplace=True)
  periods_df = df_T_daily.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  df_T_daily.reset_index(inplace=True)


  periods_2013=periods_df
  periods_2013.set_index('AAAAMMJJHH', inplace=True)

  df_sismo = df_sismo.copy()
  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  #initialisation
  rockfall_HR_days_after = []

  #parcourir les dates de pgel
  for date in periods_2013.index:
      #vérifier s'il y a un rockfall dans les y jours suivant la date de pgel
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
  proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall = Probability( df, df_sismo, HP, threshold1, intensity)

  nbr_periodes_rockfall_HP_apres_Hposs, nbr_periodes_no_rockfall_HP_apres_Hposs , nbr_observables_H_HP = Nbr_rockfall_surHP_LIFT( df_Daily_T, df_sismo, HP)

  print(f"(HP={HP}) terminé")

  return i, proba_H_R, nbr_H_rockfall, nbr_H_no_rockfall , nbr_periodes_rockfall_HP_apres_Hposs , nbr_periodes_no_rockfall_HP_apres_Hposs, nbr_observables_H_HP  #, nbr_pasP_rockfall, nbr_pasP_no_rockfall



if __name__ == "__main__":

    df_sismo= pd.read_csv("/Data/ev_sismo2.csv")
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    #pour avoir les dates sismo en jour
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    df = pd.read_csv('/Data/data_Temp_st-H.csv', delimiter=',')
    df_T = pd.read_csv('/Data/1.0_ T_St_Hiliaire_filled_complete.csv', delimiter='\t')

    df['AAAAMMJJHH'] = df_T['AAAAMMJJHH'].values
    df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH']) 


    pgel = simulate_temperature('/Data/data_Temp_sabrine_st-H.csv')
    df["pgel"] = pgel  # dans le cas de pgel non !! on prend toutes le heures de la journée et on voit si au moins pgel> 16cm !!
   
    #avoir les dates en journaliers des temp
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

    HP_values = range(1, 15)
    HR_values = range(1, 2)

    threshold1= 0.16

    conditional=[ 'global', 'low', 'high']
    for intensity in conditional:
        print(f'---------- intensity = {intensity}---------------')
        proba1_values_depthfreeze= np.zeros((1, len(HP_values)))
        proba1_values_pas_depthfreeze = np.zeros((1, len(HP_values)))
        nombres_rockfall_apres_depthfreeze=np.zeros((1, len(HP_values)))
        nombres_rockfall_apres_pas_depthfreeze=np.zeros((1, len(HP_values)))

        nombres_no_rockfall_apres_depthfreeze=np.zeros((1, len(HP_values)))
        nombres_no_rockfall_apres_pas_depthfreeze=np.zeros((1, len(HP_values)))

        nbr_R_HP_depth_freeze_possible= np.zeros((1, len(HP_values)))
        nbr_No_R_HP_depth_freeze_possible= np.zeros((1, len(HP_values)))

        LIFT_H=np.zeros((1, len(HP_values)))

        tasks = [(i, HP) for i, HP in enumerate(HP_values)]

        start_time= time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(worker, tasks)

        for result in results:
            i, proba_depthfreeze_R, nbr_depthfreeze_rockfall, nbr_depthfreeze_no_rockfall , nbr_periodes_rockfall_HP_apres_depthfreeze_poss , nbr_periodes_no_rockfall_HP_apres_depthfreeze_poss, nbr_observables_depthfreeze_HP  = result
            proba1_values_depthfreeze[:,i] = proba_depthfreeze_R
            nombres_rockfall_apres_depthfreeze[:,i] = nbr_depthfreeze_rockfall
            nombres_no_rockfall_apres_depthfreeze[:,i] = nbr_depthfreeze_no_rockfall

            proba_R = nbr_periodes_rockfall_HP_apres_depthfreeze_poss / nbr_observables_depthfreeze_HP
            lift= proba_depthfreeze_R / proba_R if proba_R > 0 else 0

            LIFT_H[:, i] = lift

            nbr_R_HP_depth_freeze_possible[:,i] = nbr_periodes_rockfall_HP_apres_depthfreeze_poss
            nbr_No_R_HP_depth_freeze_possible[:,i] = nbr_periodes_no_rockfall_HP_apres_depthfreeze_poss

        file_proba = f'p(R|pgel avec pgel = {intensity})_daily.txt'
        np.savetxt(os.path.join(outputFolder, file_proba), proba1_values_depthfreeze)
        np.savetxt(os.path.join(outputFolder, f'nombres_rockfall_apres_pgel = {intensity}.txt'), nombres_rockfall_apres_depthfreeze)
        np.savetxt(os.path.join(outputFolder, f'nombres_no_rockfall_apres_pgel = {intensity}.txt'), nombres_no_rockfall_apres_depthfreeze)
        np.savetxt(os.path.join(outputFolder, f'LIFT_pgel = {intensity}.txt'), LIFT_H)

        margin_of_error = calcul_IC(nombres_rockfall_apres_depthfreeze, nombres_no_rockfall_apres_depthfreeze, 1 , len(HP_values))
        np.savetxt(os.path.join(outputFolder, f'margin_of_error_p(R|pgel avec pgel = {intensity}).txt'), margin_of_error)

        #Compute de KHI-DEUX
        n11=nombres_rockfall_apres_depthfreeze
        n10=nombres_no_rockfall_apres_depthfreeze
        n01= nbr_R_HP_depth_freeze_possible - n11
        n00= nbr_No_R_HP_depth_freeze_possible -n10
        p_value_test_khi,d = compute_chi2 (n11, n10, n01 , n00, HR_values, HP_values)
        np.savetxt(os.path.join(outputFolder, f'p_value pgel = {intensity}.txt'), p_value_test_khi)


    plot_probability_heatmaps_levels( outputFolder, modality="pgel", conditions=conditional,
                                     HP_values=range(1, 15), filename="proba_DEPTH_FREEZE.png" )
    
    plot_aggregation_heatmaps_levels( outputFolder=outputFolder, modality="pgel", conditions=conditional, HP_values=range(1, 15),
                            compute_ci_level_matrix=compute_ci_level_matrix , filename="Aggregation_freeze_depth.png")
