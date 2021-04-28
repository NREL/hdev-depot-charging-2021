# Code to Generate Heavy-Duty Electric Truck Depot Load Profiles

## Overview:  
Code developed to generate heavy-duty electric truck depot load profiles for
the study, "Heavy-Duty Truck Electrification and the Impacts of Depot Charging
on Electricity Distribution Systems", by Borlaug et al., [published]() in 2021. **This software is provided as-is without dedicated support**. The programming environment for this study may be reproduced with conda (installed via the Anaconda [website](https://docs.anaconda.com/anaconda/install/)):  
  
`conda env create -f environment.yml`  
  
To activate the environment:  
  
`conda activate hdev-depot-charging-2021`  

## Project Organization:

    hdev-depot-charging-2021/
    ├── data/
    │   ├── fleet-schedules/          <- op. summaries for fleets studied
    │   │   ├── fleet1-beverage-delivery/
    │   │   │   ├──veh_op_days.csv    <- vehicle operating day summaries
    │   │   │   ├──veh_schedules.csv  <- on/off-shift vehicle schedules
    │   │   │
    │   │   ├── fleet2-warehouse-delivery/
    │   │   │   ├──veh_op_days.csv    <- ...
    │   │   │   ├──veh_schedules.csv  <- ...
    │   │   │
    │   │   ├── fleet3-food-delivery/
    │   │       ├──veh_op_days.csv    <- ...
    │   │       ├──veh_schedules.csv  <- ...
    │   │
    │   ├── outputs/
    │       ├── 1s-load-profiles/     <- sim. 1s daily load profiles
    │       ├── 15min-load-profiles/  <- agg. 15-min daily load profiles
    │   
    ├── figures/                      <- plotting output dir
    │
    ├── notebooks/                    <- Jupyter notebooks
    │   ├── demo.ipynb                <- demonstrates code use
    │
    ├── src/                          <- source code
        ├── depot_charging.py         <- gen/plots/agg load profiles
  
## Citation:  
`Borlaug, B., Muratori, M., Gilleran, M., Woody, D., Muston, W., Canada, T., Ingram, A., Gresham, H., and McQueen, C., (2021). "Heavy-Duty Truck Electrification and the Impacts of Depot Charging on Electricity Distribution Systems". Forthcoming.`  
  
## License:  
This code is licensed for use under the terms of the Berkeley Software Distribution 3-clause (BSD-3) license; see **LICENSE**.
