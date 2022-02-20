from memilio.simulation import UncertainContactMatrix, ContactMatrix, Damping
from memilio.simulation.secir import SecirModel, simulate, AgeGroup, Index_InfectionState, SecirSimulation, interpolate_simulation_result
from memilio.simulation.secir import InfectionState as State
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date

import warnings

def test_SA(hello):
    print(str(hello)+"!")


def simulate_model(params):
    '''
    TODO: Explain parameters
    # num_groups
    # num_compartments
    # start_date = (date(start_year, start_month, start_day) - date(start_year, 1, 1)).days
    # baseline_contact_matrix = baseline_contact_matrix, 
    # minimum_contact_matrix = minimum_contact_matrix
    # damping_coeff
    # damping_time
    # damping_level
    # daming_type
    # days
    # dt
    # output_index : index of compartment list where maximal value over time is given as output
    '''
    # Initialize Parameters
    model = SecirModel(params["num_groups"])

    # Set parameters
    for i in range(params["num_groups"]):
        # Compartment transition duration
        model.parameters.IncubationTime[AgeGroup(i)] = 5.2 #params["incubation_time"]
        model.parameters.InfectiousTimeMild[AgeGroup(i)] = params["infectious_mild_time"]
        model.parameters.SerialInterval[AgeGroup(i)] = 4.2 #params["serial_interval"]
        model.parameters.HospitalizedToHomeTime[AgeGroup(i)] = params["hospitalized_to_home_time"]
        model.parameters.HomeToHospitalizedTime[AgeGroup(i)] = params["home_to_hospitalized_time"]
        model.parameters.HospitalizedToICUTime[AgeGroup(i)] = params["hospitalized_to_icu_time"]
        model.parameters.ICUToHomeTime[AgeGroup(i)] = params["icu_to_home_time"]
        model.parameters.ICUToDeathTime[AgeGroup(i)] = params["icu_to_death_time"]

        t_inf_asymp = 1.0 / (0.5 / (5.2 - 4.2)) + 0.5 * params["infectious_mild_time"]
        model.parameters.InfectiousTimeAsymptomatic[AgeGroup(i)] = t_inf_asymp

        # Initial number of peaople in each compartment
        model.populations[AgeGroup(i), Index_InfectionState(State.Exposed)] = params["init_exposed"]
        model.populations[AgeGroup(i), Index_InfectionState(State.Carrier)] = params["init_carrier"]
        model.populations[AgeGroup(i), Index_InfectionState(State.Infected)] = params["init_infected"]
        model.populations[AgeGroup(i), Index_InfectionState(State.Hospitalized)] = params["init_hospitalized"]
        model.populations[AgeGroup(i), Index_InfectionState(State.ICU)] = params["init_ICU"]
        model.populations[AgeGroup(i), Index_InfectionState(State.Recovered)] = params["init_recovered"]
        model.populations[AgeGroup(i), Index_InfectionState(State.Dead)] = params["init_dead"]
        model.populations.set_difference_from_group_total_AgeGroup((AgeGroup(i), Index_InfectionState(State.Susceptible)), params["populations"][i])
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.Exposed)].value)
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.Carrier)].value)
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.Infected)].value)
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.Hospitalized)].value)
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.ICU)].value)
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.Recovered)].value)
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.Dead)].value)
        # print(model.populations[AgeGroup(i), Index_InfectionState(State.Susceptible)].value)
        
        # print("Population: ", params["populations"][i])
        
        

         # Compartment transition propabilities
        model.parameters.RelativeCarrierInfectability[AgeGroup(i)] = params["relative_carrier_infectability"]  
        model.parameters.InfectionProbabilityFromContact[AgeGroup(i)] = params["infection_probability_from_contact"]
        model.parameters.AsymptoticCasesPerInfectious[AgeGroup(i)] = params["asymptotic_cases_per_infectious"]  # 0.01-0.16
        model.parameters.RiskOfInfectionFromSympomatic[AgeGroup(i)] = params["risk_of_infection_from_symptomatic"]  # 0.05-0.5
        model.parameters.HospitalizedCasesPerInfectious[AgeGroup(i)] = params["hospitalized_cases_per_infectious"]  # 0.1-0.35
        model.parameters.ICUCasesPerHospitalized[AgeGroup(i)] = params["ICU_cases_per_hospitalized"]  # 0.15-0.4
        model.parameters.DeathsPerICU[AgeGroup(i)] = params["deaths_per_hospitalized"]  # 0.15-0.77
        model.parameters.MaxRiskOfInfectionFromSympomatic[AgeGroup(i)] = params["max_risk_of_infection_from_symptomatic"] # twice the value of RiskOfInfectionFromSymptomatic

    model.parameters.StartDay = params["start_day"]

    # set contact rates and emulate some mitigations
    # set contact frequency matrix
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = params["baseline_contact_matrix"]
    model.parameters.ContactPatterns.cont_freq_mat[0].minimum = params["minimum_contact_matrix"]
    # Define Damping on Contacts
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(Damping(coeffs = np.ones((params["num_groups"], params["num_groups"])) * params["damping_coeff"], t = 30.0, level = 0, type = 0))
   
    model.apply_constraints()

    # Run Simulation
    result = simulate(0, params["days"], params["dt"], model) 
    result = interpolate_simulation_result(result)
    
    # return maximal number of infected persons during the given time interval
    num_time_points = result.get_num_time_points()
    result_array = result.as_ndarray()
    t = result_array[0, :]
    group_data = np.transpose(result_array[1:, :])

    #sum over all groups
    data = np.zeros((num_time_points, params["num_compartments"]))
    for i in range(params["num_groups"]):
        data += group_data[:, i * params["num_compartments"] : (i + 1) * params["num_compartments"]]
    
    output = data[:, params["output_index"]]

    if params["output_operation"] == "max":
        output = np.max(output, axis = 0)
    elif params["output_operation"] == "sum":
        output = np.sum(output, axis = 0)
    elif params["output_operation"] == "mean":
        output = np.mean(output, axis = 0)

    output = np.squeeze(output)
    
    return output
    


"""
points = ot.Sample([[1.0], [2.0], [3.0]])

weights = ot.Point([0.4, 0.5, 1.0])

my_distribution = ot.UserDefined(points, weights)

print(my_distribution)

test = my_distribution.getSample(20)

print(test)
"""


'''   
def simulate_model(params):
    """
    sampledFactors, compartments = compartments, populations = populations, days = days, 
                   start_day = start_day, start_month = start_month, start_year = start_year, 
                   dt = dt, num_groups = num_groups, num_compartments = num_compartments, 
                   baseline_contact_matrix = baseline_contact_matrix, 
                   minimum_contact_matrix = minimum_contact_matrix
    """

    # Initialize Parameters
    model = secir.SecirModel(num_groups)

    # Set parameters
    for i in range(num_groups):
        # Compartment transition duration
        model.parameters.times[i].set_incubation(sampledFactors[0])
        model.parameters.times[i].set_infectious_mild(sampledFactors[1])
        model.parameters.times[i].set_serialinterval(sampledFactors[2])
        model.parameters.times[i].set_hospitalized_to_home(sampledFactors[3])
        model.parameters.times[i].set_home_to_hospitalized(sampledFactors[4])
        model.parameters.times[i].set_hospitalized_to_icu(sampledFactors[5])
        model.parameters.times[i].set_icu_to_home(sampledFactors[6])
        model.parameters.times[i].set_icu_to_death(sampledFactors[7])

        # Initial number of peaople in each compartment
        model.populations[secir.AgeGroup(i), secir.Index_InfectionState(secir.InfectionState.Exposed)] = 100
        model.populations[secir.AgeGroup(i), secir.Index_InfectionState(secir.InfectionState.Carrier)] = 40
        model.populations[secir.AgeGroup(i), secir.Index_InfectionState(secir.InfectionState.Infected)] = 80
        model.populations[secir.AgeGroup(i), secir.Index_InfectionState(secir.InfectionState.Hospitalized)] = 40
        model.populations[secir.AgeGroup(i), secir.Index_InfectionState(secir.InfectionState.ICU)] = 20
        model.populations[secir.AgeGroup(i), secir.Index_InfectionState(secir.InfectionState.Recovered)] = 7
        model.populations[secir.AgeGroup(i), secir.Index_InfectionState(secir.InfectionState.Dead)] = 3
        model.populations.set_difference_from_group_total_AgeGroup((secir.AgeGroup(i), 
            secir.Index_InfectionState(secir.InfectionState.Susceptible)), populations[i])

         # Compartment transition propabilities
        model.parameters.probabilities[i].set_infection_from_contact(1.0)
        model.parameters.probabilities[i].set_carrier_infectability(0.67)
        model.parameters.probabilities[i].set_asymp_per_infectious(0.09)
        model.parameters.probabilities[i].set_risk_from_symptomatic(0.25)
        model.parameters.probabilities[i].set_hospitalized_per_infectious(0.2)
        model.parameters.probabilities[i].set_icu_per_hospitalized(0.25)
        model.parameters.probabilities[i].set_dead_per_icu(0.3)

    model.parameters.set_start_day(start_day + start_month * 30) # TODO: start day has to adapted more precisely!
    
    # set contact rates and emulate some mitigations
    # set contact frequency matrix
    model.parameters.get_contact_patterns().cont_freq_mat[0].baseline = baseline_contact_matrix
    model.parameters.get_contact_patterns().cont_freq_mat[0].minimum = minimum_contact_matrix

    # Define Damping on Contacts
    model.parameters.get_contact_patterns().cont_freq_mat.add_damping(secir.Damping(np.ones((num_groups, num_groups)) * 0.9, 30, 0, 0))

     # Apply mathematical constraints to parameters
    model.apply_constraints()

    # Run Simulation
    result = secir.simulate(0, days, dt, model)
    
    # return maximal number of infected persons during the given time interval
    num_time_points = result.get_num_time_points()
    result_array = result.as_ndarray()
    t = result_array[0, :]
    group_data = np.transpose(result_array[1:, :])

    #sum over all groups
    data = np.zeros((num_time_points, num_compartments))
    for i in range(num_groups):
        data += group_data[:, i * num_compartments : (i + 1) * num_compartments]
    
    infections = data[:, compartments.index("Infected")]
    return np.max(infections)
'''
# use dictionary for input factors 
'''compartments = compartments, populations = populations, days = days, 
                   start_day = start_day, start_month = start_month, start_year = start_year, 
                   dt = dt, num_groups = num_groups, num_compartments = num_compartments, 
                   baseline_contact_matrix = baseline_contact_matrix, 
                   minimum_contact_matrix = minimum_contact_matrix

'''

def generate_output_daywise(inputDesign, input_factor_names, static_params):
    # how many timepoints does the integration return?
    output = np.zeros((len(inputDesign), static_params["days"]+1))
    
    for i in range(len(inputDesign)):
        result = simulate_model({**dict(zip(input_factor_names, inputDesign[i])), **static_params})
        output[i] = result
        
    return output