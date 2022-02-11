#############################################################################
# Copyright (C) 2020-2021 German Aerospace Center (DLR-SC)
#
# Authors: Martin J. Kuehn, Wadim Koslow, Annalena Lange
#
# Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#############################################################################
from memilio.simulation import UncertainContactMatrix, ContactMatrix, Damping
from memilio.simulation.secir import SecirModel, simulate, AgeGroup, Index_InfectionState, SecirSimulation
from memilio.simulation.secir import InfectionState as State
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
import os


def run_secir_groups_simulation():
    """
    Runs the c++ secir model using mulitple age groups 
    and plots the results
    """

    # Define Comartment names
    compartments = ['Susceptible', 'Exposed', 'Carrier',
                    'Infected', 'Hospitalized', 'ICU', 'Recovered', 'Dead']
    # Define age Groups
    groups = ['0-4', '5-14', '15-34', '35-59', '60-79', '80+']
    # Define population of age groups
    populations = [40000, 70000, 190000, 290000, 180000, 60000]

    days = 100  # number of days to simulate
    start_day = 1
    start_month = 1
    start_year = 2019
    dt = 0.1
    num_groups = len(groups)
    num_compartments = len(compartments)

    # set contact frequency matrix
    data_dir = os.path.join(os.path.dirname(
        __file__), "..", "..", "..", "data")
    baseline_contact_matrix0 = os.path.join(
        data_dir, "contacts/baseline_home.txt")
    baseline_contact_matrix1 = os.path.join(
        data_dir, "contacts/baseline_school_pf_eig.txt")
    baseline_contact_matrix2 = os.path.join(
        data_dir, "contacts/baseline_work.txt")
    baseline_contact_matrix3 = os.path.join(
        data_dir, "contacts/baseline_other.txt")

    # Initialize Parameters
    model = SecirModel(len(populations))

    # set parameters
    for i in range(num_groups):
        # Compartment transition duration

        model.parameters.IncubationTime[AgeGroup(i)] = 5.2  # R_2^(-1)+R_3^(-1)
        model.parameters.InfectiousTimeMild[AgeGroup(
            i)] = 6.  # 4-14  (=R4^(-1))
        # 4-4.4 // R_2^(-1)+0.5*R_3^(-1)
        model.parameters.SerialInterval[AgeGroup(i)] = 4.2
        model.parameters.HospitalizedToHomeTime[AgeGroup(
            i)] = 12.  # 7-16 (=R5^(-1))
        model.parameters.HomeToHospitalizedTime[AgeGroup(
            i)] = 5.  # 2.5-7 (=R6^(-1))
        model.parameters.HospitalizedToICUTime[AgeGroup(
            i)] = 2.  # 1-3.5 (=R7^(-1))
        model.parameters.ICUToHomeTime[AgeGroup(i)] = 8.  # 5-16 (=R8^(-1))
        model.parameters.ICUToDeathTime[AgeGroup(i)] = 5.  # 3.5-7 (=R5^(-1))

        # Initial number of peaople in each compartment
        model.populations[AgeGroup(
            i), Index_InfectionState(State.Exposed)] = 100
        model.populations[AgeGroup(
            i), Index_InfectionState(State.Carrier)] = 50
        model.populations[AgeGroup(
            i), Index_InfectionState(State.Infected)] = 50
        model.populations[AgeGroup(i), Index_InfectionState(
            State.Hospitalized)] = 20
        model.populations[AgeGroup(i), Index_InfectionState(State.ICU)] = 10
        model.populations[AgeGroup(
            i), Index_InfectionState(State.Recovered)] = 10
        model.populations[AgeGroup(i), Index_InfectionState(State.Dead)] = 0
        model.populations.set_difference_from_total(
            (AgeGroup(i), Index_InfectionState(State.Susceptible)), populations[i])

        # Compartment transition propabilities

        model.parameters.RelativeCarrierInfectability[AgeGroup(i)] = 0.67
        model.parameters.InfectionProbabilityFromContact[AgeGroup(i)] = 1.0
        model.parameters.AsymptoticCasesPerInfectious[AgeGroup(
            i)] = 0.09  # 0.01-0.16
        model.parameters.RiskOfInfectionFromSympomatic[AgeGroup(
            i)] = 0.25  # 0.05-0.5
        model.parameters.HospitalizedCasesPerInfectious[AgeGroup(
            i)] = 0.2  # 0.1-0.35
        model.parameters.ICUCasesPerHospitalized[AgeGroup(
            i)] = 0.25  # 0.15-0.4
        model.parameters.DeathsPerICU[AgeGroup(i)] = 0.3  # 0.15-0.77
        # twice the value of RiskOfInfectionFromSymptomatic
        model.parameters.MaxRiskOfInfectionFromSympomatic[AgeGroup(i)] = 0.5

    model.parameters.StartDay = (
        date(start_year, start_month, start_day) - date(start_year, 1, 1)).days

    # set contact rates and emulate some mitigations
    # set contact frequency matrix
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.loadtxt(baseline_contact_matrix0) \
        + np.loadtxt(baseline_contact_matrix1) + \
        np.loadtxt(baseline_contact_matrix2) + \
        np.loadtxt(baseline_contact_matrix3)
    model.parameters.ContactPatterns.cont_freq_mat[0].minimum = np.ones(
        (num_groups, num_groups)) * 0
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(Damping(
        coeffs=np.ones((num_groups, num_groups)) * 0.9, t=30.0, level=0, type=0))

    # Apply mathematical constraints to parameters
    model.apply_constraints()

    # Run Simulation
    result = simulate(0, days, dt, model)
    # print(result.get_last_value())

    num_time_points = result.get_num_time_points()
    result_array = result.as_ndarray()
    t = result_array[0, :]
    group_data = np.transpose(result_array[1:, :])

    # sum over all groups
    data = np.zeros((num_time_points, num_compartments))
    for i in range(num_groups):
        data += group_data[:, i * num_compartments: (i + 1) * num_compartments]

    # Plot Results
    datelist = np.array(pd.date_range(datetime(start_year, start_month,
                        start_day), periods=days, freq='D').strftime('%m-%d').tolist())

    tick_range = (np.arange(int(days / 10) + 1) * 10)
    tick_range[-1] -= 1
    fig, ax = plt.subplots()
    ax.plot(t, data[:, 0], label='#Susceptible')
    ax.plot(t, data[:, 1], label='#Exposed')
    ax.plot(t, data[:, 2], label='#Carrier')
    ax.plot(t, data[:, 3], label='#Infected')
    ax.plot(t, data[:, 4], label='#Hospitalzed')
    ax.plot(t, data[:, 5], label='#ICU')
    ax.plot(t, data[:, 6], label='#Recovered')
    ax.plot(t, data[:, 7], label='#Dead')
    ax.set_title("SECIR simulation results (entire population)")
    ax.set_xticks(tick_range)
    ax.set_xticklabels(datelist[tick_range], rotation=45)
    ax.legend()
    fig.tight_layout
    fig.savefig('Secir_by_compartments.pdf')

    # plot dynamics in each comparment by age group
    fig, ax = plt.subplots(4, 2, figsize=(12, 15))

    for i, title in zip(range(num_compartments), compartments):

        for j, group in enumerate(groups):
            ax[int(np.floor(i / 2)), int(i % 2)].plot(t,
                                                      group_data[:, j*num_compartments+i], label=group)

        ax[int(np.floor(i / 2)), int(i % 2)].set_title(title, fontsize=10)
        ax[int(np.floor(i / 2)), int(i % 2)].legend()

        ax[int(np.floor(i / 2)), int(i % 2)].set_xticks(tick_range)
        ax[int(np.floor(i / 2)), int(i % 2)
           ].set_xticklabels(datelist[tick_range], rotation=45)
    plt.subplots_adjust(hspace=0.5, bottom=0.1, top=0.9)
    fig.suptitle('SECIR simulation results by age group in each compartment')
    fig.savefig('Secir_age_groups_in_compartments.pdf')

    fig, ax = plt.subplots(4, 2, figsize=(12, 15))
    for i, title in zip(range(num_compartments), compartments):
        ax[int(np.floor(i / 2)), int(i % 2)].plot(t, data[:, i])
        ax[int(np.floor(i / 2)), int(i % 2)].set_title(title, fontsize=10)

        ax[int(np.floor(i / 2)), int(i % 2)].set_xticks(tick_range)
        ax[int(np.floor(i / 2)), int(i % 2)
           ].set_xticklabels(datelist[tick_range], rotation=45)
    plt.subplots_adjust(hspace=0.5, bottom=0.1, top=0.9)
    fig.suptitle('SECIR simulation results by compartment (entire population)')
    fig.savefig('Secir_all_parts.pdf')

    plt.show()
    plt.close()

    # return data


if __name__ == "__main__":
    run_secir_groups_simulation()
