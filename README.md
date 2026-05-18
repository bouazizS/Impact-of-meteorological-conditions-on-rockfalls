# Impact-of-meteorological-conditions-on-rockfalls

This repository computes probabilities of rockfall conditioned on meteorological variables (precipitation, water charge, freezing and freezing depth). It contains scripts that read local data, compute conditional probabilities, lift, chi-squared tests, confidence interval and aggregation plots, and save results.

Requirements
- Python 3.8+
- Core packages: numpy, pandas, matplotlib, seaborn, scipy, pillow
- Install with pip (example):
  pip install numpy pandas matplotlib seaborn scipy pillow

Data
Sources:
    - Meteorological datasets are available on Zenodo: 
    - Rockfall catalog: 

- Place meteorological and seismic catalog CSV files in the same paths expected by the scripts. Example paths used in scripts:
  - /Data/1.0_RR1.csv
  - /Data/ev_sismo2.csv
  - See each script for exact expected filenames.



Usage
- Run the scripts. Examples:
  python3 Probabilities_rockfall_conditions/proba_rain.py
  python3 Probabilities_rockfall_conditions/proba_freeze.py
  python3 Probabilities_rockfall_conditions/proba_charge.py
  python3 Probabilities_rockfall_conditions/proba_depth_freeze.py


  Outputs
- Scripts create output folders under `Probabilities_rockfall_conditions/` such as:
  - [cdt_thresholding/](Probabilities_rockfall_conditions/cdt_thresholding) — Results and plots of probabilities using rainfall conditions.
  - [continuous_freeze/](Probabilities_rockfall_conditions/continuous_freeze) — Results and plots of probabilities using continuous-freeze.
  - [Charge-discharge/](Probabilities_rockfall_conditions/Charge-discharge) — H / Results and plots of probabilities using charge.
  - [Freeze_depth/](Probabilities_rockfall_conditions/Freeze_depth) — Results and plots of probabilities using depth-freeze.