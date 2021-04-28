import os
import math
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator


def generate_load_profiles(fleet,
                           fleet_size, 
                           charge_strat,
                           n_samples=50, 
                           kwh_per_mile=1.8,
                           kw=100,
                           seed=0, 
                           to_file=False,
                           ylim=None,
                           print_pk_loads=True,
                           agg_15min=False):
        """
        Generates n_samples fleet load profiles for fleet ('fleet1-beverage-
        delivery', 'fleet2-warehouse-delivery', or 'fleet3-food-delivery')
        and fleet_size for a particular charge_strat ('immediate', 'delayed', 
        'min_power') assuming kwh_per_mile average energy consumption rate. If 
        charging_strategy is 'immediate' or 'delayed', EV is charged at kw constant 
        charging power. If to_file==True, second-by-second average day, max peak 
        load day, and min peak load day fleet charging load profiles are written to 
        .csv at '../data/outputs/' and plots are written to .png at 
        '../figures/'.

        Args:
            fleet (str): {'fleet1-beverage-delivery', 'fleet2-warehouse-delivery', 
                'fleet3-food-delivery'}
            fleet_size (int): number of electric trucks operating per day
            charge_strat (str): {'immediate', 'delayed', 'min_power'}
            n_samples (int, optional): number of sample fleet-days to approx. the
                sampling distribution
            kwh_per_mile (float, optional): average fuel consumption rate
            kw (int, optional): if method in {'immediate', 'delayed'}, constant charging 
                power level
            seed (int, optional): random seed for reproduceability
            to_file (bool, optional): if True: write load profiles to file at
                '../data/load_profiles/' and plot to file at '../figures/load_profiles/'
            y_lim (int, optional): fixes y-axis limit for plotting. If None, y-axis limit
                is set automatically
            prink_pk_loads (bool, optional): if True, prints peak load (kW) for min peak
                load day, average day, and max peak load day to STDOUT.
            agg_15min (bool, optional): if True, load profile is aggregated over 
                15-min interval (avg.), else load profile second-by-second

        Returns:
            pd.DataFrame of concatenated vehicle charging schedules
        """
        
        assert fleet in ['fleet1-beverage-delivery', 
                         'fleet2-warehouse-delivery', 
                         'fleet3-food-delivery'], "fleet not recognized!"
        
        assert charge_strat in ['immediate', 
                                'delayed', 
                                'min_power'], "charge_strat not recognized!"

        if agg_15min:
            res='15min'
        else:
            res='1s'
        
        # load fleet veh_op_day summaries, veh_scheds
        v_days_df = pd.read_csv(os.path.join('..', 'data', 'fleet-schedules', f'{fleet}', 'veh_op_days.csv'))
        v_scheds_df = pd.read_csv(os.path.join('..', 'data', 'fleet-schedules', f'{fleet}', 'veh_schedules.csv'))
        
        # produce random seeds
        random.seed(seed)
        rand_ints = [random.randint(0,999) for i in range(n_samples)]
        
        # Color Pallette:
        if fleet == 'fleet1-beverage-delivery': #red
            main_color = '#E77377'
            accent_color = '#f5d0d1'
        elif fleet == 'fleet2-warehouse-delivery': #green
            main_color = '#8dccbe'
            accent_color = '#bfe3db'
        elif fleet == 'fleet3-food-delivery': #blue
            main_color = '#355070'
            accent_color = '#a2bee0' 

        # init:
        avg_veh_loads, max_veh_loads, min_veh_loads = [], [], []
        avg_fleet_loads, max_fleet_loads, min_fleet_loads = [], [], []
        total_load_all_samples = np.zeros(86399)
        max_peak_load_all_fleet_profs = 0
        min_peak_load_all_fleet_profs = np.inf
        fig, ax = plt.subplots(figsize=(2, 1.67))
        
        for rand_int in rand_ints:
            v_days_sample_df = v_days_df.sample(fleet_size, 
                                                replace=True,
                                                random_state=rand_int)

            # Combine charging profiles
            charge_profs_df = pd.DataFrame()
            for i, vday in v_days_sample_df.iterrows():
                vday_sched_df = v_scheds_df[v_scheds_df.veh_op_day_id==vday.veh_op_day_id]

                if charge_strat == 'min_power':
                    # Calculate daily energy consumption w/ kwh/mi assumption
                    total_energy_kwh = kwh_per_mile * vday.vmt

                    # Find min constant power to offset daily energy consumption
                    day_off_shift_hrs = vday.time_off_shift_s / 3600
                    min_power_kw = total_energy_kwh / day_off_shift_hrs

                    # Extend on/off-shift reporting to sec-by-sec
                    on_shift_s = [] #init             
                    for i, pattern in vday_sched_df.iterrows():
                        on_shift = pattern.on_shift
                        total_s = pattern.total_time_s
                        on_shift_s.extend([on_shift] * total_s)

                    # Construct sec-by-sec charging profile
                    charge_prof_df = pd.DataFrame({'veh_num': i,
                                                   'rel_s': range(len(on_shift_s)),
                                                   'on_shift': on_shift_s})

                    inst_power_func = lambda x: min_power_kw if x==0 else 0
                    inst_pwr = charge_prof_df['on_shift'].apply(inst_power_func)
                    charge_prof_df['kw'] = inst_pwr
                    
                else: #immediate or delayed charging strategies
                    power_kw = kw
                    three_vday_scheds_df = pd.concat([vday_sched_df]*3).reset_index(drop=True)
                    start_times, end_times, total_time_secs = [],[],[] #init
                    on_shifts, vmts = [],[] #init

                    i = 0
                    while i < len(three_vday_scheds_df):
                        row = three_vday_scheds_df.iloc[i]
                        if i == len(three_vday_scheds_df) - 1:
                            start_times.append(row.start_time)
                            end_times.append(row.end_time)
                            total_time_secs.append(row.total_time_s)
                            on_shifts.append(row.on_shift)
                            vmts.append(row.vmt)
                            i+=1
                        else:
                            next_row = three_vday_scheds_df.iloc[i+1]
                            if row.on_shift == next_row.on_shift:
                                start_times.append(row.start_time)
                                end_times.append(next_row.end_time)
                                time_s = row.total_time_s + next_row.total_time_s
                                total_time_secs.append(time_s)
                                on_shifts.append(row.on_shift)
                                vmts.append(row.vmt+next_row.vmt)
                                i+=2
                            else:
                                start_times.append(row.start_time)
                                end_times.append(row.end_time)
                                total_time_secs.append(row.total_time_s)
                                on_shifts.append(row.on_shift)
                                vmts.append(row.vmt)
                                i+=1

                    three_vday_scheds_df = pd.DataFrame({'start_time': start_times,
                                                        'end_time': end_times,
                                                        'total_time_s': total_time_secs,
                                                        'on_shift': on_shifts,
                                                        'vmt': vmts})

                    net_energy_consumed_kwh = 0 #init
                    charging_power_kw, on_shift_s = [], [] #init
                    for i, row in three_vday_scheds_df.iterrows():
                        if row.on_shift == 1: #if: on-shift...
                            net_energy_consumed_kwh += (row.vmt * kwh_per_mile) #add energy
                            charging_power_kw.extend([0] * row.total_time_s)
                            on_shift_s.extend([1] * row.total_time_s)
                        else: #if: not on-shift...
                            req_charging_s = math.ceil(net_energy_consumed_kwh / power_kw * 3600)
                            dwell_s = row.total_time_s
                            on_shift_s.extend([0] * row.total_time_s)
                            if (req_charging_s > dwell_s)&(dwell_s > 0): #if: required charging time > dwell time... 
                                energy_charged_kwh = power_kw * dwell_s / 3600
                                net_energy_consumed_kwh -= energy_charged_kwh #subtract energy
                                charging_power_kw.extend([power_kw] * dwell_s)
                            else: # if: required charging time <= dwell_time...
                                net_energy_consumed_kwh = 0 #charge to full
                                if charge_strat == 'immediate':
                                    charging_power_kw.extend([power_kw] * req_charging_s)
                                    charging_power_kw.extend([0] * (dwell_s - req_charging_s))
                                elif charge_strat == 'delayed':
                                    charging_power_kw.extend([0] * (dwell_s - req_charging_s))
                                    charging_power_kw.extend([power_kw] * req_charging_s)    

                    charging_power_kw = charging_power_kw[86399: 172798] #middle day
                    on_shift_s = on_shift_s[86399: 172798] #middle day
                    
                    # Construct charging profile
                    charge_prof_df = pd.DataFrame({'veh_num': i,
                                                   'rel_s': range(len(charging_power_kw)),
                                                   'on_shift': on_shift_s,
                                                   'kw': charging_power_kw})
                     
                # Combine w/ other charging profiles
                charge_profs_df = pd.concat([charge_profs_df, charge_prof_df]).reset_index(drop=True)

            avg_veh_load_kw = charge_profs_df[charge_profs_df.on_shift==0]['kw'].mean()
            avg_veh_loads.append(avg_veh_load_kw)
            max_veh_load_kw = max(charge_profs_df['kw'])
            max_veh_loads.append(max_veh_load_kw)

            fleet_prof_df = charge_profs_df.groupby('rel_s')['kw'].sum()
            fleet_prof_df = fleet_prof_df.reset_index()
            avg_fleet_load_kw = fleet_prof_df[fleet_prof_df.kw!=0]['kw'].mean()
            avg_fleet_loads.append(avg_fleet_load_kw)
            max_fleet_load_kw = max(fleet_prof_df['kw'])
            max_fleet_loads.append(max_fleet_load_kw)
            
            if max_fleet_load_kw > max_peak_load_all_fleet_profs:
                max_peak_load_all_fleet_profs = max_fleet_load_kw
                pk_load_fleet_prof_df = fleet_prof_df
            
            if max_fleet_load_kw < min_peak_load_all_fleet_profs:
                min_peak_load_all_fleet_profs = max_fleet_load_kw
                min_load_fleet_prof_df = fleet_prof_df

            total_load_all_samples += np.array(fleet_prof_df['kw'])

            # Plot fleet daily load profile
            ax.plot(fleet_prof_df['rel_s'], 
                    fleet_prof_df['kw'], 
                    color=accent_color,
                    linewidth=0.5,
                    alpha=0.4)

        # Plot average fleet daily load profile
        avg_load_all_samples = total_load_all_samples / n_samples
        ax.plot(range(len(avg_load_all_samples)),
                avg_load_all_samples,
                color=main_color)

        plt.xlim(0, len(fleet_prof_df['rel_s']))
        
        if ylim != None:
            plt.ylim(-0.1, ylim)
        else:
            plt.ylim(-0.1)
        
        ax.set_xticks(np.linspace(0,len(fleet_prof_df['rel_s']), 25)[::4])
        ax.set_xticklabels(range(0,26)[::4], fontsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        plt.yticks(fontsize=8)
        plt.grid(axis='both', linestyle='--')
        
        if to_file==True:
            # Save plot as .png
            plot_fp = os.path.join('..', 'figures', f'{fleet}_{fleet_size}vehs_{charge_strat}.png')
            plt.savefig(plot_fp, bbox_inches='tight', dpi=300)
            
            # Write average load profile to .csv
            times = []
            for hour in range(24):
                for minute in range(60):
                    for second in range(60):
                        times.append(str(datetime.time(hour, minute, second)))

            avg_load_all_samples = list(avg_load_all_samples)            
            avg_load_profile_df = pd.DataFrame({'time': times,
                                                'power_kW': [avg_load_all_samples[0]] + avg_load_all_samples})
            
            if agg_15min: #aggregate load to avg. over 15-min. increments
                avg_load_profile_df = agg_15_min_load_profile(avg_load_profile_df)
            
            avg_lp_fp = os.path.join('..', 'data', 'outputs', f'{fleet}_{fleet_size}vehs_avg-prof_{charge_strat}_{res}.csv')
            avg_load_profile_df.to_csv(avg_lp_fp, index=False)
            
            # Write max peak load profile to .csv
            max_pk_loads = list(pk_load_fleet_prof_df['kw'])
            pk_load_profile_df = pd.DataFrame({'time': times,
                                               'power_kW': [max_pk_loads[0]] + max_pk_loads})

            if agg_15min: #aggregate load to avg. over 15-min. increments
                pk_load_profile_df = agg_15_min_load_profile(pk_load_profile_df)
            
            pk_lp_fp = os.path.join('..', 'data', 'outputs', f'{fleet}_{fleet_size}vehs_peak-prof_{charge_strat}_{res}.csv')              
            pk_load_profile_df.to_csv(pk_lp_fp, index=False)
            
            # Write min peak load profile to .csv
            min_pk_loads = list(min_load_fleet_prof_df['kw'])
            min_load_profile_df = pd.DataFrame({'time': times,
                                                'power_kW': [min_pk_loads[0]] + min_pk_loads})

            if agg_15min: #aggregate load to avg. over 15-min. increments
                min_load_profile_df = agg_15_min_load_profile(min_load_profile_df)
            
            min_lp_fp = os.path.join('..', 'data', 'outputs', f'{fleet}_{fleet_size}vehs_min-prof_{charge_strat}_{res}.csv')
            min_load_profile_df.to_csv(min_lp_fp, index=False)
            
        if print_pk_loads:
            print('Low Bound Peak Demand (kW): {}'.format(round(min_peak_load_all_fleet_profs,2)))
            print('Average Peak Demand (kW): {}'.format(round(np.array(avg_load_all_samples).max(),2)))
            print('Upper Bound Peak Demand (kW): {}'.format(round(max_peak_load_all_fleet_profs,2)))
            
        plt.show()
        
        return charge_profs_df


def agg_15_min_load_profile (load_profile_df):
    """
    Aggregates 1-Hz load profile by taking average demand over 15-min 
    increments.
    """

    s_in_15min = 15 * 60
    
    # prepare idx slices
    start_idxs = np.arange(0, len(load_profile_df), s_in_15min)
    end_idxs = np.arange(s_in_15min, len(load_profile_df) + s_in_15min, s_in_15min)

    # generate list of avg kw over 15-min increments
    avg_15min_kw = [] #init
    for s_idx, e_idx in zip(start_idxs, end_idxs):
        avg_15min_kw.append(load_profile_df['power_kW'][s_idx:e_idx].mean())

    times = [] #init
    for hour in range(24):
        for minute in range(0, 60, 15):
            times.append(str(datetime.time(hour, minute, 0)))

    # create pd.DataFrame
    agg_15min_load_profile_df = pd.DataFrame({'time': times,
                                              'avg_power_kw': avg_15min_kw})
    
    return agg_15min_load_profile_df