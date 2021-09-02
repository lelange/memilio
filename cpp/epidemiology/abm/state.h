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
#ifndef EPI_ABM_STATE_H
#define EPI_ABM_STATE_H

#include <cstdint>

namespace epi
{

/** 
 * infection state in ABM.
 * can be used as 0-based index
 */
enum class InfectionState : std::uint32_t
{
    Susceptible = 0,
    Exposed,
    Carrier,
    Infected_Detected,
    Infected_Undetected,
    Infected_Severe,
    Infected_Critical,
    Recovered_Carrier,
    Recovered_Infected,
    Dead,
    //TODO: Add description of the different infection states
    Count //last!!
};

/**
 * vacination state in ABM.
 * can be used as 0-based index.
 */
enum class VacinationState : std::uint32_t
{
    Unvacinated = 0,
    Vacinated,
    
    Count //last!!
};

} // namespace epi

#endif