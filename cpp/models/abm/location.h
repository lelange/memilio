/* 
* Copyright (C) 2020-2021 German Aerospace Center (DLR-SC)
*
* Authors: Daniel Abele
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef EPI_ABM_LOCATION_H
#define EPI_ABM_LOCATION_H

#include "abm/parameters.h"
#include "abm/testing_scheme.h"
#include "abm/state.h"
#include "abm/location_type.h"

#include "memilio/math/eigen.h"
#include <array>
#include <random>

namespace mio
{
class Person;

/**
 * LocationId identifies a Location uniquely. It consists of the LocationType of the Location and an Index.
 * The index corresponds to the index into the structure m_locations from world, where all Locations are saved.
 */
struct LocationId
{
    uint32_t index;
    LocationType type;
};

/**
 * all locations in the simulated world where persons gather.
 */
class Location
{
public:
   /**
     * construct a Location of a certain type.
     * @param type the type of the location
     * @param index the index of the location
     */
    Location(LocationType type, uint32_t index);

    /**
     * get the type of this location.
     */
    LocationType get_type() const 
    {
        return m_type;
    }

    /**
     *get the index of this location.
     */
    unsigned get_index() const
    {
        return m_index;
    }

    /** 
     * a person interacts with the population at this location, may change infection state.
     * @param person the person that interacts with the population
     * @param dt length of the current simulation time step
     * @param global_params global infection parameters
     * @return new infection state of the person
     */
    InfectionState interact(const Person& person, TimeSpan dt, const GlobalInfectionParameters& global_params) const;

    /** 
     * add a person to the population at this location.
     * @param person the person arriving
    */
    void add_person(const Person& person);

    /** 
     * remove a person from the population of this location.
     * @param person the person leaving
     */
    void remove_person(const Person& person);

    /** 
     * notification that one person in this location changed infection state.
     * @param person the person that changed infection state
     * @param old_state the previous infection state of the person
     */
    void changed_state(const Person& person, InfectionState old_infection_state);

    /** 
     * prepare the location for the next simulation step.
     * @param dt the duration of the simulation step
     * @param global_params global infection parameters
     */
    void begin_step(TimeSpan dt, const GlobalInfectionParameters& global_params);

    /** 
     * number of persons at this location in one infection state.
     * @return number of persons at this location that are in the specified infection state
     */
    int get_subpopulation(InfectionState s) const;

    /** 
     * number of persons at this location for all infection states.
     * vector is indexed by InfectionState.
     * @return number of persons in all infection states.
     * */
    Eigen::Ref<const Eigen::VectorXi> get_subpopulations() const;

    /**
     * @return parameters of the infection that are specific to this location
     */
    LocalInfectionParameters& get_infection_parameters()
    {
        return m_parameters;
    }
    
    const LocalInfectionParameters& get_infection_parameters() const
    {
        return m_parameters;
    }

    void set_testing_scheme (TimeSpan interval, double probability)
    {
        m_testing_scheme = TestingScheme(interval, probability);
    }

    const TestingScheme& get_testing_scheme() const
    {
        return m_testing_scheme;
    }

private:
    void change_subpopulation(InfectionState s, int delta);

private: 
    LocationType m_type;
    uint32_t m_index;
    int m_num_persons = 0;
    std::array<int, size_t(InfectionState::Count)> m_subpopulations;
    LocalInfectionParameters m_parameters;
    CustomIndexArray<double, AbmAgeGroup, mio::VaccinationState> m_cached_exposure_rate;
    TestingScheme m_testing_scheme;
};
} // namespace mio

#endif
