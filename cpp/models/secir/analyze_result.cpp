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
#include "secir/analyze_result.h"

#include <algorithm>
#include <cassert>

namespace mio
{

/**
 * TODO: extrapolate first and last point
 */
TimeSeries<double> interpolate_simulation_result(const TimeSeries<double>& simulation_result)
{
    assert(simulation_result.get_num_time_points() > 0 && "TimeSeries must not be empty.");

    const auto t0      = simulation_result.get_time(0);
    const auto tmax    = simulation_result.get_last_time();
    const auto day0    = static_cast<int>(floor(t0));
    const auto day_max = static_cast<int>(ceil(tmax));

    auto day = day0;
    TimeSeries<double> interpolated(simulation_result.get_num_elements());
    interpolated.reserve(day_max - day0 + 1);
    interpolated.add_time_point(day, simulation_result.get_value(0));
    day++;

    //interpolate between pair of time points that lie on either side of each integer day
    for (int i = 0; i < simulation_result.get_num_time_points() - 1;) {
        //only go to next pair of time points if no time point is added.
        //otherwise check the same time points again
        //in case there is more than one day between the two time points
        if (simulation_result.get_time(i) < day && simulation_result.get_time(i + 1) >= day) {
            auto weight = (day - simulation_result.get_time(i)) /
                          (simulation_result.get_time(i + 1) - simulation_result.get_time(i));
            interpolated.add_time_point(day, simulation_result[i] +
                                                 (simulation_result[i + 1] - simulation_result[i]) * weight);
            ++day;
        }
        else {
            ++i;
        }
    }

    if (day_max > tmax) {
        interpolated.add_time_point(day, simulation_result.get_last_value());
    }

    return interpolated;
}

std::vector<std::vector<TimeSeries<double>>>
sum_nodes(const std::vector<std::vector<TimeSeries<double>>>& ensemble_result)
{
    auto num_runs        = ensemble_result.size();
    auto num_nodes       = ensemble_result[0].size();
    auto num_time_points = ensemble_result[0][0].get_num_time_points();
    auto num_elements    = ensemble_result[0][0].get_num_elements();

    std::vector<std::vector<TimeSeries<double>>> sum_result(
        num_runs, std::vector<TimeSeries<double>>(1, TimeSeries<double>::zero(num_time_points, num_elements)));

    for (size_t run = 0; run < num_runs; run++) {
        for (Eigen::Index time = 0; time < num_time_points; time++) {
            sum_result[run][0].get_time(time) = ensemble_result[run][0].get_time(time);
            for (size_t node = 0; node < num_nodes; node++) {
                sum_result[run][0][time] += ensemble_result[run][node][time];
            }
        }
    }
    return sum_result;
}

std::vector<TimeSeries<double>> ensemble_mean(const std::vector<std::vector<TimeSeries<double>>>& ensemble_result)
{
    auto num_runs        = ensemble_result.size();
    auto num_nodes       = ensemble_result[0].size();
    auto num_time_points = ensemble_result[0][0].get_num_time_points();
    auto num_elements    = ensemble_result[0][0].get_num_elements();

    std::vector<TimeSeries<double>> mean(num_nodes, TimeSeries<double>::zero(num_time_points, num_elements));

    for (size_t run = 0; run < num_runs; run++) {
        assert(ensemble_result[run].size() == num_nodes && "ensemble results not uniform.");
        for (size_t node = 0; node < num_nodes; node++) {
            assert(ensemble_result[run][node].get_num_time_points() == num_time_points &&
                   "ensemble results not uniform.");
            for (Eigen::Index time = 0; time < num_time_points; time++) {
                assert(ensemble_result[run][node].get_num_elements() == num_elements &&
                       "ensemble results not uniform.");
                mean[node].get_time(time) = ensemble_result[run][node].get_time(time);
                mean[node][time] += ensemble_result[run][node][time] / num_runs;
            }
        }
    }

    return mean;
}

std::vector<TimeSeries<double>> ensemble_percentile(const std::vector<std::vector<TimeSeries<double>>>& ensemble_result,
                                                    double p)
{
    assert(p > 0.0 && p < 1.0 && "Invalid percentile value.");

    auto num_runs        = ensemble_result.size();
    auto num_nodes       = ensemble_result[0].size();
    auto num_time_points = ensemble_result[0][0].get_num_time_points();
    auto num_elements    = ensemble_result[0][0].get_num_elements();

    std::vector<TimeSeries<double>> percentile(num_nodes, TimeSeries<double>::zero(num_time_points, num_elements));

    std::vector<double> single_element_ensemble(num_runs); //reused for each element
    for (size_t node = 0; node < num_nodes; node++) {
        for (Eigen::Index time = 0; time < num_time_points; time++) {
            percentile[node].get_time(time) = ensemble_result[0][node].get_time(time);
            for (Eigen::Index elem = 0; elem < num_elements; elem++) {
                std::transform(ensemble_result.begin(), ensemble_result.end(), single_element_ensemble.begin(),
                               [=](auto& run) {
                                   return run[node][time][elem];
                               });
                std::sort(single_element_ensemble.begin(), single_element_ensemble.end());
                percentile[node][time][elem] = single_element_ensemble[static_cast<size_t>(num_runs * p)];
            }
        }
    }
    return percentile;
}

double result_distance_2norm(const std::vector<mio::TimeSeries<double>>& result1,
                             const std::vector<mio::TimeSeries<double>>& result2)
{
    assert(result1.size() == result2.size());
    assert(result1.size() > 0);
    assert(result1[0].get_num_time_points() > 0);
    assert(result1[0].get_num_elements() > 0);

    auto norm_sqr = 0.0;
    for (auto iter_node1 = result1.begin(), iter_node2 = result2.begin(); iter_node1 < result1.end();
         ++iter_node1, ++iter_node2) {
        for (Eigen::Index time_idx = 0; time_idx < iter_node1->get_num_time_points(); ++time_idx) {
            auto v1 = (*iter_node1)[time_idx];
            auto v2 = (*iter_node2)[time_idx];
            norm_sqr += ((v1 - v2).array() * (v1 - v2).array()).sum();
        }
    }
    return std::sqrt(norm_sqr);
}

double result_distance_2norm(const std::vector<mio::TimeSeries<double>>& result1,
                             const std::vector<mio::TimeSeries<double>>& result2, InfectionState compartment)
{
    assert(result1.size() == result2.size());
    assert(result1.size() > 0);
    assert(result1[0].get_num_time_points() > 0);
    assert(result1[0].get_num_elements() > 0);

    auto num_compartments = Eigen::Index(InfectionState::Count);
    auto num_age_groups   = result1[0].get_num_elements() / num_compartments;

    auto norm_sqr = 0.0;
    for (auto iter_node1 = result1.begin(), iter_node2 = result2.begin(); iter_node1 < result1.end();
         ++iter_node1, ++iter_node2) {
        for (Eigen::Index time_idx = 0; time_idx < iter_node1->get_num_time_points(); ++time_idx) {
            auto v1 = (*iter_node1)[time_idx];
            auto v2 = (*iter_node2)[time_idx];
            for (Eigen::Index age_idx = 0; age_idx < num_age_groups; ++age_idx) {
                auto d1 = v1[age_idx * num_compartments + Eigen::Index(compartment)];
                auto d2 = v2[age_idx * num_compartments + Eigen::Index(compartment)];
                norm_sqr += (d1 - d2) * (d1 - d2);
            }
        }
    }
    return std::sqrt(norm_sqr);
}

} // namespace mio
