import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import math
import os 
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utilis import plot_fig_proba , compute_chi2 , calcul_IC, plot_aggregation


def trouver_periodes_HR_pluie_pas_pluie(df, n):
    periods = []
    for i in range(len(df) - n + 1):
        debut = df['AAAAMMJJHH'].iloc[i]
        fin = df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Date Début': debut, 'Date Fin': fin})
    return pd.DataFrame(periods)


def Nbr_rockfall_surHP_apres_HR_possibles(x, precipitation_per_day, df_sismo, y, seuil_R, cond_amp):
  #slicing à partir des dates disponibles pour le catalogue sismo
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique
  date_ref = date_obj - pd.Timedelta(days=x)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  #filtrer les données de précipitations pour la période de référence
  precipitation_per_day.set_index('AAAAMMJJHH', inplace=True)
  periods_df = precipitation_per_day.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  precipitation_per_day.reset_index(inplace=True)
  #trouver les périodes de pluie et pas de pluie
  periods_2013 = trouver_periodes_HR_pluie_pas_pluie(periods_df, x)
  periods_2013.set_index('Date Début', inplace=True)


  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  #initialisation
  rockfall_HR_days_after = []

  # Parcourir les dates de précipitations/ periodes
  for date in periods_2013.index:
      # Définir la période (dates et heures) à analyser
      start_date = date + pd.Timedelta(days=x)
      end_date = date + pd.Timedelta(days=x + y - 1)

      # Filtrer les événements dans la période choisie
      df_sismo_filtered = df_sismo.loc[start_date:end_date]

      # Vérifier si on a des événements rockfall (type 'R')
      rockfall_events = df_sismo_filtered[df_sismo_filtered['type'] == 'R']

      rockfall_detected = False

      if not rockfall_events.empty:  # Si des rockfalls existent
          amplitudes_rockfall = rockfall_events['A(nm/s)']

          # Condition 2 : vérifier l'amplitude selon 'cond_amp'
          if (seuil_R is not None):
              if cond_amp == 'sup':
                  # Vérifier s'il existe au moins un rockfall avec amplitude supérieure ou égale à seuil_R
                  rockfall_detected = (amplitudes_rockfall >= seuil_R).any()

              elif cond_amp == 'inf':
                  # Vérifier que tous les rockfalls détectés ont une amplitude inférieure à seuil_R
                  rockfall_detected = (amplitudes_rockfall < seuil_R).all()
          else:
              # Si seuil_R est None, on peut définir rockfall_detected comme True
              rockfall_detected = True
      else:
          rockfall_detected = False

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods_2013.copy()
  resultat['rockfall sur période'] = rockfall_HR_days_after
  nbr_rockfall_HP = resultat['rockfall sur période'].sum()
  nbr_jours=len(resultat)
  nbr_no_rockfall_HP= nbr_jours- nbr_rockfall_HP

  return resultat, nbr_rockfall_HP, nbr_no_rockfall_HP, nbr_jours


def Nbr_rockfall_surHP_LIFT(x, precipitation_per_day, df_sismo, y, seuil_R, cond_amp):
  #slicing à partir des dates disponibles pour le catalogue sismo
  date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
  date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique
  date_ref = date_obj - pd.Timedelta(days=x)
  date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

  #filtrer les données de précipitations pour la période de référence
  precipitation_per_day.set_index('AAAAMMJJHH', inplace=True)
  periods_df = precipitation_per_day.loc[date_ref:date_fin_ref]
  periods_df.reset_index(inplace=True)
  precipitation_per_day.reset_index(inplace=True)
  #trouver les périodes de pluie et pas de pluie
  periods_2013 = trouver_periodes_HR_pluie_pas_pluie(periods_df, x)
  periods_2013.set_index('Date Début', inplace=True)


  df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo.set_index('Date', inplace=True)

  #initialisation
  rockfall_HR_days_after = []

  # Parcourir les dates de précipitations/ periodes
  for date in periods_2013.index:
      # Définir la période (dates et heures) à analyser
      start_date = date + pd.Timedelta(days=x)
      end_date = date + pd.Timedelta(days=x + y - 1)

      # Filtrer les événements dans la période choisie
      df_sismo_filtered = df_sismo.loc[start_date:end_date]

      # Vérifier si on a des événements rockfall (type 'R')
      rockfall_events = df_sismo_filtered[df_sismo_filtered['type'] == 'R']

      rockfall_detected = False

      if not rockfall_events.empty:  # Si des rockfalls existent
          amplitudes_rockfall = rockfall_events['A(nm/s)']

          # Condition 2 : vérifier l'amplitude selon 'cond_amp'
          if (seuil_R is not None):
              if cond_amp == 'sup':
                  # Vérifier s'il existe au moins un rockfall avec amplitude supérieure ou égale à seuil_R
                  rockfall_detected = (amplitudes_rockfall >= seuil_R).any()

              elif cond_amp == 'inf':
                  # Vérifier que tous les rockfalls détectés ont une amplitude inférieure à seuil_R
                  rockfall_detected = (amplitudes_rockfall < seuil_R).all()
          else:
              # Si seuil_R est None, on peut définir rockfall_detected comme True
              rockfall_detected = True
      else:
          rockfall_detected = False

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods_2013.copy()
  resultat['rockfall sur période'] = rockfall_HR_days_after
  nbr_rockfall_HP = resultat['rockfall sur période'].sum()
  nbr_jours=len(resultat)

  return nbr_rockfall_HP, nbr_jours

def trouver_periodes_HR(df, n, seuil, check_type, condition_continue):
    periods = []
    cdt = lambda x: x > 0
    x = 1
    for i in range(len(df) - n + 1):
        # check whether we are in a "continuous rain" condition or not
        if condition_continue(x) == cdt(x):
            # Rain is considered continuous: all values in the window must satisfy the condition
            pluie_continue = df['Pluie'].iloc[i:i+n].all()
        else:
            # Otherwise: at least one value in the window doesn not satisfy the condition
            pluie_continue = df['Pluie'].iloc[i:i+n].any()

        # Extract precipitation values over the HR window
        somme_precipitations = df['RR1'].iloc[i:i+n]

        # Case 1: threshold on precipitation (sup case)
        if (seuil != None) and (check_type == 'sup') and (pluie_continue):
            # At least one value exceeds or equals the threshold
            if (somme_precipitations >= seuil).any():
                debut = df['AAAAMMJJHH'].iloc[i]
                fin = df['AAAAMMJJHH'].iloc[i+n-1]
                periods.append({'Date Début': debut, 'Date Fin': fin})

        # Case 2: threshold on precipitation (inf case)
        elif (seuil != None) and (check_type == 'inf') and (pluie_continue):
            # All values must be below the threshold
            if (somme_precipitations < seuil).all():
                debut = df['AAAAMMJJHH'].iloc[i]
                fin = df['AAAAMMJJHH'].iloc[i+n-1]
                periods.append({'Date Début': debut, 'Date Fin': fin})

        # Case 3: no precipitation threshold (only rain condition)
        elif (seuil == None) and (pluie_continue):
            debut = df['AAAAMMJJHH'].iloc[i]
            fin = df['AAAAMMJJHH'].iloc[i+n-1]
            periods.append({'Date Début': debut, 'Date Fin': fin})

    return pd.DataFrame(periods)


def Probability (x, precipitation_per_day, df_sismo, y,  cond_precip, cond_amp, condition_pluie_continue, seuil_P= None, seuil_R =None ):     #periodes HR (pluie ou non selon condition)
    #ajouter colonne Pluie selon condition pour condition : pluie continue ou non
    precipitation_per_day['Pluie'] = precipitation_per_day['RR1'].apply(condition_pluie_continue)

    #slicing à partir des dates disponibles pour le catalogue sismo
    date_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[0])  # début de sismique
    date_fin_obj = pd.to_datetime(df_sismo['AAAAMMJJHH'].iloc[-1])  # fin de sismique
    date_ref = date_obj - pd.Timedelta(days=x)
    date_fin_ref = date_fin_obj - pd.Timedelta(days=y)

    #filtrer les données de précipitations pour la période de référence
    precipitation_per_day.set_index('AAAAMMJJHH', inplace=True)
    periods_df = precipitation_per_day.loc[date_ref:date_fin_ref]
    periods_df.reset_index(inplace=True)
    precipitation_per_day.reset_index(inplace=True)

    #touver les périodes de pluie faibles ou de pluie fortes selon condition
    periods_2013 = trouver_periodes_HR(periods_df, x,seuil_P, cond_precip, condition_pluie_continue)
    if periods_2013.empty:
      return 0, 0, 0
    periods_2013.set_index('Date Début', inplace=True)

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
            amplitudes_rockfall = rockfall_events['A(nm/s)']

            # Condition 2 : l'amplitude
            if (seuil_R is not None):
                if cond_amp == 'sup':
                    # Vérifier s'il existe au moins un rockfall avec amplitude supérieure ou égale à seuil_R
                    rockfall_detected = (amplitudes_rockfall >= seuil_R).any()

                elif cond_amp == 'inf':
                    # Vérifier que tous les rockfalls détectés ont une amplitude inférieure à seuil_R
                    rockfall_detected = (amplitudes_rockfall < seuil_R).all()
            else:
                # Si seuil_R est None
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



if __name__ == "__main__":

    df = pd.read_csv('/mnt/SSD1/bouazizs/GradCam/Probabilities/1.0_RR1.csv', delimiter='\t')
    df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH'])
    df.set_index('AAAAMMJJHH', inplace=True)
    precipitation_per_day = df.resample('1D').sum()
    precipitation_per_day.reset_index(inplace=True)
    precipitation_per_day['AAAAMMJJHH'] = pd.to_datetime(precipitation_per_day['AAAAMMJJHH'])

    df_sismo= pd.read_csv("/mnt/SSD1/bouazizs/GradCam/Probabilities/ev_sismo2.csv")
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')

    #----------------------------------------------
    #on garde que les events par jour
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_unique = df_sismo.drop_duplicates(subset='AAAAMMJJHH')
    #filtrer les lignes avec type 'R'
    df_R = df_sismo[df_sismo['type'] == 'R']
    #supprimer les doublons basés sur la date
    df_R_unique = df_R.drop_duplicates(subset='AAAAMMJJHH')
    #le nombre de lignes uniques =nbr de rockfall
    nombre_R_unique = df_R_unique.shape[0]
    start_date = max(df_sismo['AAAAMMJJHH'].min(), precipitation_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), precipitation_per_day['AAAAMMJJHH'].max())

    # Filtrer les DataFrames pour ne conserver que les dates communes
    df_sismo = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    # precipitation_per_day = precipitation_per_day[(precipitation_per_day['AAAAMMJJHH']>= start_date) & (precipitation_per_day['AAAAMMJJHH'] <= end_date)]


    HR_values = range(1, 15)
    HP_values = range(1, 15)

    outputFolder="/mnt/SSD1/bouazizs/GradCam/Probabilities_rockfall_conditions/cdt_thresholding"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)


    #listes des conditions qu'on a :
    cdts_pluie=[lambda x: x > 0]#pour aspect continue de pluie
    cdts_Seuil_P=['sup','inf'] #      #pour forte et faible pluie
    cdts_Seuil_R=['sup', 'inf'] #     #pour forte et faible amplitude

    # Uncomment this to treat just the condition of continous rain
    # cdts_Seuil_P=[None] 
    # cdts_Seuil_R=[None] 

    seuil_5_P= round(precipitation_per_day['RR1'].quantile(0.95),2)
    seuil_10_R= round(df_R['A(nm/s)'].quantile(0.90),2 )

    # For computing the Khi-2 
    nbr_observables_HR_HP= np.zeros((len(HR_values), len(HP_values)))
    nbr_periodes_rockfall_HP= np.zeros((len(HR_values), len(HP_values)))
    nbr_no_periodes_rockfall_HP= np.zeros((len(HR_values), len(HP_values)))

    continued=None
    for cdt_Seuil_R in cdts_Seuil_R:
        print('cdt_Seuil_R=',cdt_Seuil_R)
        seuil_amp= seuil_10_R
        if cdt_Seuil_R==None:
            seuil_amp = None
        for i, HR in enumerate(HR_values):
            for j, HP in enumerate(HP_values):
                _, nbr_periodes_rockfall_HP[i,j], nbr_no_periodes_rockfall_HP[i,j] , nbr_observables_HR_HP[i,j] = Nbr_rockfall_surHP_apres_HR_possibles(HR, precipitation_per_day,df_sismo, HP, seuil_R= seuil_amp, cond_amp=cdt_Seuil_R)


        np.savetxt(os.path.join(outputFolder,f'nombres_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie possible avec continue = {continued}.txt'), nbr_periodes_rockfall_HP)
        np.savetxt(os.path.join(outputFolder,f'nombres_no_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie possible avec continue = {continued}.txt'), nbr_no_periodes_rockfall_HP)


    proba_values = np.zeros((len(HR_values), len(HP_values)))
    nombres_rockfall_apres_pluie=np.zeros((len(HR_values), len(HP_values)))
    nombres_no_rockfall_apres_pluie=np.zeros((len(HR_values), len(HP_values)))
    nbr_observables_HR_HP= np.zeros((len(HR_values), len(HP_values)))
    nbr_periodes_rockfall_HP= np.zeros((len(HR_values), len(HP_values)))
    proba_R_HP = np.zeros((len(HR_values), len(HP_values)))
    LIFT = np.zeros((len(HR_values), len(HP_values)))

    for cdt_pluie in cdts_pluie:
        print('cdt_pluie=',cdt_pluie)
        for cdt_Seuil_P in cdts_Seuil_P:
            print('cdt_Seuil_P=',cdt_Seuil_P)
            seuil_inten = seuil_5_P
            if cdt_Seuil_P==None:
                seuil_inten = None
            for cdt_Seuil_R in cdts_Seuil_R:
                print('cdt_Seuil_R=',cdt_Seuil_R)
                seuil_amp= seuil_10_R
                if cdt_Seuil_R==None:
                    seuil_amp = None

                for i, HR in enumerate(HR_values):
                    for j, HP in enumerate(HP_values):
                        proba_values[i, j], nombres_rockfall_apres_pluie[i,j], nombres_no_rockfall_apres_pluie[i,j] = Probability (HR, precipitation_per_day, df_sismo, HP, cond_precip=cdt_Seuil_P, cond_amp=cdt_Seuil_R, condition_pluie_continue= cdt_pluie , seuil_P=seuil_inten, seuil_R =seuil_amp )

                        nbr_periodes_rockfall_HP[i,j] , nbr_observables_HR_HP[i,j] = Nbr_rockfall_surHP_LIFT(HR, precipitation_per_day,df_sismo, HP, seuil_R= seuil_amp, cond_amp=cdt_Seuil_R)

                        proba_R_HP[i,j]= nbr_periodes_rockfall_HP[i,j]  / nbr_observables_HR_HP[i,j]
                        LIFT[i,j] = proba_values[i,j]  / proba_R_HP[i,j]

                if cdt_pluie == cdts_pluie[0]:
                    continued= "yes"
                else:
                    continued= "no"
                
                # Save Proba values
                file_proba= f'p(R {cdt_Seuil_R} {seuil_amp}|P{cdt_Seuil_P} {seuil_inten} avec continue = {continued}).txt'
                np.savetxt(os.path.join(outputFolder, file_proba), proba_values)
                # Save Nbr of rockfalls values
                np.savetxt(os.path.join(outputFolder,f'nombres_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie {cdt_Seuil_P} {seuil_inten} avec continue = {continued}.txt'), nombres_rockfall_apres_pluie)
                np.savetxt(os.path.join(outputFolder, f'nombres_no_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie {cdt_Seuil_P} {seuil_inten} avec continue = {continued}.txt'), nombres_no_rockfall_apres_pluie)
                # Save lift values and plot
                file_lift= f'Lift(R {cdt_Seuil_R} {seuil_amp}|P{cdt_Seuil_P} {seuil_inten} avec continue = {continued}).txt'
                np.savetxt(os.path.join(outputFolder, file_lift), LIFT)
                #calcul IC et sauvgarde:
                margin_of_error= calcul_IC(nombres_rockfall_apres_pluie, nombres_no_rockfall_apres_pluie, HR=14, HP=14 ,alpha=0.05)
                output_file_IC_path = os.path.join(outputFolder, f'IC_{file_proba}')
                np.savetxt(output_file_IC_path, margin_of_error)

                plt.rcParams.update({'axes.labelsize': 14,    # Taille du texte des labels des axes
                                'axes.titlesize': 14,   # Taille du texte du titre des axes
                                'xtick.labelsize': 14,  # Taille du texte des labels des ticks X
                                'ytick.labelsize': 14,  # Taille du texte des labels des ticks Y
                                'font.size': 13,        # Taille générale du texte
                                'legend.fontsize': 15})
                plot_fig_proba(outputFolder, os.path.join(outputFolder, file_proba), HR_values, HP_values )
                if seuil_amp == None and seuil_inten == None:
                    plot_fig_proba(outputFolder, os.path.join(outputFolder,f'nombres_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie {cdt_Seuil_P} {seuil_inten} avec continue = {continued}.txt'), HR_values, HP_values )

                #Compute Khi-deux 

                output_file_khi = os.path.join(outputFolder, f"Khi-deux-P{cdt_Seuil_P}_RF{cdt_Seuil_R}")
                rockfall_cond = np.loadtxt(
                    f'{outputFolder}/nombres_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie {cdt_Seuil_P} {seuil_inten} avec continue = yes.txt')

                no_rockfall_cond = np.loadtxt(
                    f'{outputFolder}/nombres_no_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie {cdt_Seuil_P} {seuil_inten} avec continue = yes.txt')

                rockfall_total = np.loadtxt(
                    f'{outputFolder}/nombres_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie possible avec continue = None.txt')

                no_rockfall_total = np.loadtxt(
                    f'{outputFolder}/nombres_no_rockfall {cdt_Seuil_R} {seuil_amp}_apres_pluie possible avec continue = None.txt')

                rockfall_not_cond = rockfall_total - rockfall_cond
                no_rockfall_not_cond = no_rockfall_total - no_rockfall_cond    

                p_values, d = compute_chi2(rockfall_cond, no_rockfall_cond, rockfall_not_cond, no_rockfall_not_cond, HR_values, HP_values)
                np.savetxt(output_file_khi, p_values)


                # --------Plot figure of aggregation 
                #figure size config :
                plt.rcParams.update({'axes.labelsize': 18,    # Taille du texte des labels des axes
                            'axes.titlesize': 18,   # Taille du texte du titre des axes
                            'xtick.labelsize': 18,  # Taille du texte des labels des ticks X
                            'ytick.labelsize': 18,  # Taille du texte des labels des ticks Y
                            'font.size': 16,        # Taille générale du texte
                            'legend.fontsize': 18})

                plot_aggregation( LIFT=LIFT, p_values=p_values, margin_of_error=margin_of_error, outputFolder=outputFolder,
                                filename=f"Aggregation_P{cdt_Seuil_P}_RF{cdt_Seuil_R}.png", HR=np.arange(1, 15), HP=np.arange(1, 15),
                                )

