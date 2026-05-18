import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('1.0_RR1.csv', delimiter='\t')
df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH'])
df.set_index('AAAAMMJJHH', inplace=True)
precipitation_per_day = df.resample('1D').sum()
precipitation_per_day.reset_index(inplace=True)
precipitation_per_day['AAAAMMJJHH'] = pd.to_datetime(precipitation_per_day['AAAAMMJJHH'])

df_sismo= pd.read_csv("ev_sismo2.csv")
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


def trouver_periodes_HR(df, n, check_type='all'):
    periods = []
    for i in range(len(df) - n + 1):
        if check_type == 'all' and df['Pluie'].iloc[i:i+n].all():
            debut = df['AAAAMMJJHH'].iloc[i]
            fin = df['AAAAMMJJHH'].iloc[i+n-1]
            periods.append({'Date Début': debut, 'Date Fin': fin})
        elif check_type == 'any' and df['Pluie'].iloc[i:i+n].any():
            debut = df['AAAAMMJJHH'].iloc[i]
            fin = df['AAAAMMJJHH'].iloc[i+n-1]
            periods.append({'Date Début': debut, 'Date Fin': fin})
    return pd.DataFrame(periods)

def Probability (x, precipitation_per_day, df_sismo, y, condition, check):     #periodes HR (pluie ou non selon condition)
    #ajouter colonne Pluie selon condition
    precipitation_per_day['Pluie'] = precipitation_per_day['RR1'].apply(condition)

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

    #touver les périodes de pluie ou pas de pluie
    periods_2013 = trouver_periodes_HR(periods_df, x, check)
    periods_2013.set_index('Date Début', inplace=True)

    df_sismo['Date'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo.set_index('Date', inplace=True)

    #initialisation
    rockfall_HR_days_after = []

    # Parcourir les dates de précipitations/ periodes
    for date in periods_2013.index:
        #verifier s'il y a un rockfall dans les y jours suivant la date de précipitation/ periodes
        end_date = date + pd.Timedelta(days=x + y - 1)
        rockfall_condition = df_sismo.loc[date + pd.Timedelta(days=x):end_date]['type'] == 'R'
        rockfall_detected = rockfall_condition.any()

        rockfall_HR_days_after.append(1 if rockfall_detected else 0)

    resultat = periods_2013.copy()
    resultat['rockfall sur période'] = rockfall_HR_days_after
    proba = resultat['rockfall sur période'].mean()   #(nbr periodes suivie de rockfall / nbr de periodes pluie HP))

    nbr= resultat['rockfall sur période'].sum()
    nbr_no=len(resultat)-nbr

    return proba, nbr, nbr_no


def trouver_periodes_HR_pluie_pas_pluie(df, n):
    periods = []
    for i in range(len(df) - n + 1):
        debut = df['AAAAMMJJHH'].iloc[i]
        fin = df['AAAAMMJJHH'].iloc[i+n-1]
        periods.append({'Date Début': debut, 'Date Fin': fin})
    return pd.DataFrame(periods)


def Nbr_rockfall_surHP(x, precipitation_per_day, df_sismo, y):
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

  #parcourir les dates de précipitations
  for date in periods_2013.index:
      #vérifier s'il y a un rockfall dans les y jours suivant la date de précipitation
      end_date = date + pd.Timedelta(days=x + y-1)
      rockfall_condition = df_sismo.loc[date + pd.Timedelta(days=x):end_date]['type'] == 'R'
      rockfall_detected = rockfall_condition.any()

      rockfall_HR_days_after.append(1 if rockfall_detected else 0)

  resultat = periods_2013.copy()
  resultat['rockfall sur période'] = rockfall_HR_days_after
  nbr_rockfall_HP = resultat['rockfall sur période'].sum()

  return nbr_rockfall_HP

def proba_1(HR, precipitation_per_day, df_sismo, HP):
    #calcul de la probabilité d'éboulement pour les périodes de pluie continue
    condition_pluie = lambda x: x > 0
    check_pluie="all"
    proba_pluie, nbr_p_r ,nbr_p_noR = Probability(HR, precipitation_per_day, df_sismo, HP, condition_pluie,check_pluie)

    return proba_pluie , nbr_p_r ,nbr_p_noR

def proba_1_pas_pluie(HR, precipitation_per_day, df_sismo, HP):
    #calcul de la probabilité d'éboulement pour les périodes de non-pluie continue
    condition_pas_pluie = lambda x: x == 0
    check_pas_pluie="any"
    proba_pas_pluie, nbr_pasP_r , nbr_pasP_noR= Probability(HR, precipitation_per_day, df_sismo, HP, condition_pas_pluie,check_pas_pluie)

    return proba_pas_pluie, nbr_pasP_r, nbr_pasP_noR



# Calcul des probabilités pour les valeurs de HR et HF
HR_values = range(1, 15)
HP_values = range(1, 15)
proba1_values_pluie = np.zeros((len(HR_values), len(HP_values)))
proba1_values_pas_pluie = np.zeros((len(HR_values), len(HP_values)))

nombres_rockfall_apres_pluie=np.zeros((len(HR_values), len(HP_values)))
nombres_rockfall_apres_pas_pluie=np.zeros((len(HR_values), len(HP_values)))

nombres_no_rockfall_apres_pluie=np.zeros((len(HR_values), len(HP_values)))
nombres_no_rockfall_apres_pas_pluie=np.zeros((len(HR_values), len(HP_values)))

for i, HR in enumerate(HR_values):
    for j, HP in enumerate(HP_values):
        proba1_values_pluie[i, j], nombres_rockfall_apres_pluie[i,j], nombres_no_rockfall_apres_pluie[i,j] = proba_1(HR, precipitation_per_day, df_sismo, HP)

        proba1_values_pas_pluie[i, j], nombres_rockfall_apres_pas_pluie[i,j], nombres_no_rockfall_apres_pas_pluie[i,j]  = proba_1_pas_pluie(HR, precipitation_per_day, df_sismo, HP)

import os
dossier = "/Probabilities_rockfall_conditions/cdt_pluie_continue"
os.makedirs(dossier, exist_ok=True)
np.savetxt(os.path.join(dossier, 'p(rHP|pHR).txt'), proba1_values_pluie)
np.savetxt(os.path.join(dossier, 'nombres_rockfall_apres_pluie.txt'), nombres_rockfall_apres_pluie)
np.savetxt(os.path.join(dossier, 'nombres_rockfall_apres_pas_pluie.txt'), nombres_rockfall_apres_pas_pluie)
np.savetxt(os.path.join(dossier, 'nombres_no_rockfall_apres_pluie.txt'), nombres_no_rockfall_apres_pluie)
np.savetxt(os.path.join(dossier, 'nombres_no_rockfall_apres_pas_pluie.txt'), nombres_no_rockfall_apres_pas_pluie)