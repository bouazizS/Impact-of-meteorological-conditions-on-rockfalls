# Impact-of-meteorological-conditions-on-rockfalls

This repository computes probabilities of rockfall conditioned on meteorological variables (precipitation, water charge, freezing and freezing depth). It contains scripts that read local data, compute conditional probabilities, lift, chi-squared tests, confidence interval and aggregation plots, and save results.

Requirements
- Python 3.8+
- Core packages: numpy, pandas, matplotlib, seaborn, scipy
- Install with pip :
  pip install numpy pandas matplotlib seaborn scipy 


Main scripts
- [`proba_rain.py`](Probabilities_rockfall_conditions/proba_pluie.py):  Computes conditional rockfall probabilities based on rainfall conditions.  
  The script supports two analysis modes:
  1. **Continuous rainfall analysis**: evaluates rockfall occurrence following periods of continuous rainfall without threshold conditioning.
    ###### Configuration example

    To run the continuous rainfall analysis only, disable thresholding by setting:

    ```python
    cdts_Seuil_P = [None]
    cdts_Seuil_R = [None]
    ```

  2. **Threshold-based analysis**: evaluates probabilities of rockfall events with amplitudes above or below a seismic threshold, conditioned on low or high intensity rainfall.


- [`proba_freeze.py`](Probabilities_rockfall_conditions/proba_gel.py): Computes conditional rockfall probabilities with respect to continuous freezing.


- [`proba_charge.py`](Probabilities_rockfall_conditions/proba_charge.py): Computes conditional rockfall probabilities using water charge/discharge conditions derived from precipitation accumulation and exponential decay dynamics.

- [`proba_depth_freeze.py`](Probabilities_rockfall_conditions/proba_depth_freeze.py): Computes conditional rockfall probabilities conditioned on freezing depth conditions.

- [`utilis.py`](Probabilities_rockfall_conditions/utilis.py): Utilities shared across scripts for statistical analysis and plotting.


Data

- Sources:
    - Meteorological datasets are available on Zenodo: 
    - Rockfall catalog: 

- Place meteorological and seismic catalog CSV files in the same paths expected by the scripts. Example paths used in scripts:
  - /Data/1.0_RR1.csv
  - /Data/ev_sismo2.csv
  - See each script for exact expected filenames.


Usage
- Run the script you need. Examples:
```python
  python3 Probabilities_rockfall_conditions/proba_rain.py
  python3 Probabilities_rockfall_conditions/proba_freeze.py
  python3 Probabilities_rockfall_conditions/proba_charge.py
  python3 Probabilities_rockfall_conditions/proba_depth_freeze.py
```

Outputs
- Scripts create output folders under `Probabilities_rockfall_conditions/` such as:
  - [cdt_thresholding/](Probabilities_rockfall_conditions/cdt_thresholding) — Results and plots of probabilities using rainfall conditions.
  - [continuous_freeze/](Probabilities_rockfall_conditions/continuous_freeze) — Results and plots of probabilities using continuous-freeze.
  - [Charge-discharge/](Probabilities_rockfall_conditions/Charge-discharge) — H / Results and plots of probabilities using charge.
  - [Freeze_depth/](Probabilities_rockfall_conditions/Freeze_depth) — Results and plots of probabilities using depth-freeze.