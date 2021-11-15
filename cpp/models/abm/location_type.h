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
#ifndef EPI_ABM_LOCATION_TYPE_H
#define EPI_ABM_LOCATION_TYPE_H

#include <cstdint>
#include <limits>

namespace mio
{

/**
 * type of a location.
 */
enum class LocationType : std::uint32_t
{
    Home = 0,
    School,
    Work,
    SocialEvent, // TODO: differentiate different kinds
    BasicsShop, // groceries and other necessities
    Hospital,
    ICU,

    Count //last!
};

static constexpr uint32_t INVALID_LOCATION_INDEX = std::numeric_limits<uint32_t>::max();

} // namespace mio

#endif