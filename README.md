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
    │   ├── outputs/                  <- sim. 1s & agg. 15-min daily load profiles
    │   
    ├── figures/                      <- plotting output dir
    │
    ├── notebooks/                    <- Jupyter notebooks
    │   ├── demo.ipynb                <- demonstrates code use
    │
    ├── src/                          <- source code
        ├── depot_charging.py         <- generate/plot/aggregate load profiles
  
## Citation:  
`Borlaug, B., Muratori, M., Gilleran, M. et al. Heavy-duty truck electrification and the impacts of depot charging on electricity distribution systems. Nat Energy 6, 673–682 (2021). https://doi.org/10.1038/s41560-021-00855-0`  
  
## License:  
This code is licensed for use under the terms of the Berkeley Software Distribution 3-clause (BSD-3) license; see **LICENSE**.
