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
#include "memilio/compartments/simulation.h"
#include "secir/analyze_result.h"
#include "matchers.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(TestInterpolateTimeSeries, timePointsAreLinSpaced)
{
    mio::TimeSeries<double> ts(10);
    auto zeros = mio::TimeSeries<double>::Vector::Zero(10);
    ts.add_time_point(0.0, zeros);
    ts.add_time_point(0.1, zeros);
    ts.add_time_point(0.4, zeros);
    ts.add_time_point(1.2, zeros);
    ts.add_time_point(3.7, zeros);
    ts.add_time_point(3.9, zeros);
    ts.add_time_point(3.901, zeros);
    ts.add_time_point(4.5, zeros);

    auto interpolated = mio::interpolate_simulation_result(ts);

    ASSERT_THAT(interpolated.get_times(), ElementsAreLinspace(0.0, 5.0, 6));
}

TEST(TestInterpolateTimeSeries, timeSeriesCanBeginAtAnyDay)
{
    mio::TimeSeries<double> ts(10);
    auto zeros = mio::TimeSeries<double>::Vector::Zero(10);
    ts.add_time_point(-5.9, zeros);
    ts.add_time_point(-5.7, zeros);
    ts.add_time_point(-4.5, zeros);
    ts.add_time_point(-3.1, zeros);
    ts.add_time_point(-2.7, zeros);
    ts.add_time_point(-2.5, zeros);

    auto interpolated = mio::interpolate_simulation_result(ts);

    ASSERT_THAT(interpolated.get_times(), ElementsAreLinspace(-6.0, -2.0, 5));
}

TEST(TestInterpolateTimeSeries, simpleValues)
{
    using Vec = mio::TimeSeries<double>::Vector;
    mio::TimeSeries<double> ts(1);
    ts.add_time_point(0.0, Vec::Constant(1, 0.1));
    ts.add_time_point(0.5, Vec::Constant(1, 0.2));
    ts.add_time_point(1.5, Vec::Constant(1, 0.3));
    ts.add_time_point(2.5, Vec::Constant(1, 0.4));
    ts.add_time_point(3.5, Vec::Constant(1, 0.5));
    ts.add_time_point(4.5, Vec::Constant(1, 0.6));
    ts.add_time_point(5.5, Vec::Constant(1, 0.7));

    auto interpolated = mio::interpolate_simulation_result(ts);

    ASSERT_THAT(interpolated.get_times(), ElementsAreLinspace(0.0, 6.0, 7));
    ASSERT_THAT(interpolated,
                testing::ElementsAre(MatrixNear(Vec::Constant(1, 0.1)), MatrixNear(Vec::Constant(1, 0.25)),
                                     MatrixNear(Vec::Constant(1, 0.35)), MatrixNear(Vec::Constant(1, 0.45)),
                                     MatrixNear(Vec::Constant(1, 0.55)), MatrixNear(Vec::Constant(1, 0.65)),
                                     MatrixNear(Vec::Constant(1, 0.7))));
}

TEST(TestInterpolateTimeSeries, aFewMoreComplexValues)
{
    using Vec = mio::TimeSeries<double>::Vector;
    mio::TimeSeries<double> ts(2);
    ts.add_time_point(0.0, (Vec(2) << 1.0, 2.0).finished());
    ts.add_time_point(1.5, (Vec(2) << 3.0, 10.0).finished());
    ts.add_time_point(2.1, (Vec(2) << 5.0, 3.0).finished());

    auto interpolated = mio::interpolate_simulation_result(ts);

    ASSERT_THAT(interpolated.get_times(), ElementsAreLinspace(0.0, 3.0, 4));
    ASSERT_THAT(interpolated,
                testing::ElementsAre(MatrixNear((Vec(2) << 1, 2).finished()),
                                     MatrixNear((Vec(2) << 1.0 + 2.0 * 2 / 3, 2.0 + 8.0 * 2 / 3).finished()),
                                     MatrixNear((Vec(2) << 3.0 + 2.0 * 5 / 6, 10.0 - 7.0 * 5 / 6).finished()),
                                     MatrixNear((Vec(2) << 5.0, 3.0).finished())));
}

TEST(TestInterpolateTimeSeries, timePointsCanMatchDayExactly)
{
    using Vec = mio::TimeSeries<double>::Vector;
    mio::TimeSeries<double> ts(1);
    ts.add_time_point(0.0, Vec::Constant(1, 0.0));
    ts.add_time_point(0.5, Vec::Constant(1, 1.0));
    ts.add_time_point(1.0, Vec::Constant(1, 2.0));
    ts.add_time_point(2.1, Vec::Constant(1, 3.0));
    ts.add_time_point(3.0, Vec::Constant(1, 4.0));

    auto interpolated = mio::interpolate_simulation_result(ts);

    ASSERT_THAT(interpolated.get_times(), ElementsAreLinspace(0.0, 3.0, 4));
    ASSERT_THAT(interpolated[1], MatrixNear(Vec::Constant(1, 2.0)));
    ASSERT_THAT(interpolated[2], MatrixNear(Vec::Constant(1, 2.0 + 10. / 11.)));
}

TEST(TestInterpolateGraph, basic)
{
    using Model      = mio::SecirModel;
    using Simulation = mio::Simulation<Model>;
    auto g           = mio::Graph<mio::SimulationNode<Simulation>, mio::MigrationEdge>();
    g.add_node(0, Model(1), 0.5);
    g.add_node(1, Model(1), 0.5);
    for (auto& n : g.nodes()) {
        n.property.evolve(0.5, 4.0);
    }

    auto interpolated = mio::interpolate_simulation_result(g);
    ASSERT_EQ(interpolated.size(), 2);
    for (auto& n : interpolated) {
        //interpolation of time series tested separately.
        //so only checking that each node was interpolated.
        ASSERT_THAT(n.get_times(), ElementsAreLinspace(0.0, 5.0, 6));
    }
}

TEST(TestInterpolateEnsemble, basic)
{
    using Vec = mio::TimeSeries<double>::Vector;
    std::vector<mio::TimeSeries<double>> ts;
    ts.emplace_back(1);
    ts.back().add_time_point(0.0, Vec::Constant(1, 0.0));
    ts.back().add_time_point(0.5, Vec::Constant(1, 1.0));
    ts.back().add_time_point(2.0, Vec::Constant(1, 2.0));
    ts.emplace_back(1);
    ts.back().add_time_point(0.0, Vec::Constant(1, 0.0));
    ts.back().add_time_point(1.5, Vec::Constant(1, 1.0));
    ts.back().add_time_point(2.0, Vec::Constant(1, 2.0));

    auto interpolated = mio::interpolate_ensemble_results(ts);

    ASSERT_EQ(interpolated.size(), ts.size());
    ASSERT_THAT(interpolated[0].get_times(), ElementsAreLinspace(0.0, 2.0, 3));
    ASSERT_THAT(interpolated[0][1], MatrixNear(Vec::Constant(1, 1.0 + 1.0 * 1 / 3)));
    ASSERT_THAT(interpolated[1].get_times(), ElementsAreLinspace(0.0, 2.0, 3));
    ASSERT_THAT(interpolated[1][1], MatrixNear(Vec::Constant(1, 0.0 + 1.0 * 2 / 3)));
}

TEST(TestEnsembleSum, basic)
{
    using Vec = mio::TimeSeries<double>::Vector;

    std::vector<std::vector<mio::TimeSeries<double>>> ensemble;

    //run 1
    ensemble.emplace_back(3, mio::TimeSeries<double>(1));
    //node 1
    ensemble.back()[0].add_time_point(3.0, Vec::Constant(1, 0.0));
    ensemble.back()[0].add_time_point(4.0, Vec::Constant(1, 1.0));
    ensemble.back()[0].add_time_point(5.0, Vec::Constant(1, 2.0));
    //node 2
    ensemble.back()[1].add_time_point(3.0, Vec::Constant(1, 3.0));
    ensemble.back()[1].add_time_point(4.0, Vec::Constant(1, 4.0));
    ensemble.back()[1].add_time_point(5.0, Vec::Constant(1, 5.0));
    //node 3
    ensemble.back()[2].add_time_point(3.0, Vec::Constant(1, 6.0));
    ensemble.back()[2].add_time_point(4.0, Vec::Constant(1, 7.0));
    ensemble.back()[2].add_time_point(5.0, Vec::Constant(1, 8.0));

    //run 2
    ensemble.emplace_back(3, mio::TimeSeries<double>(1));
    //node 1
    ensemble.back()[0].add_time_point(3.0, Vec::Constant(1, 0.5));
    ensemble.back()[0].add_time_point(4.0, Vec::Constant(1, 2.0));
    ensemble.back()[0].add_time_point(5.0, Vec::Constant(1, 5.0));
    //node 2
    ensemble.back()[1].add_time_point(3.0, Vec::Constant(1, 7.5));
    ensemble.back()[1].add_time_point(4.0, Vec::Constant(1, 9.5));
    ensemble.back()[1].add_time_point(5.0, Vec::Constant(1, 1.0));
    //node 3
    ensemble.back()[2].add_time_point(3.0, Vec::Constant(1, 1.5));
    ensemble.back()[2].add_time_point(4.0, Vec::Constant(1, 2.5));
    ensemble.back()[2].add_time_point(5.0, Vec::Constant(1, 3.0));

    auto sum = mio::sum_nodes(ensemble);

    ASSERT_EQ(sum.size(), 2);
    ASSERT_THAT(sum[0][0].get_times(), testing::ElementsAre(3.0, 4.0, 5.0));
    ASSERT_THAT(sum[0][0], testing::ElementsAre(MatrixNear(Vec::Constant(1, 9.0)), MatrixNear(Vec::Constant(1, 12.0)),
                                                MatrixNear(Vec::Constant(1, 15.0))));
    ASSERT_THAT(sum[1][0].get_times(), testing::ElementsAre(3.0, 4.0, 5.0));
    ASSERT_THAT(sum[1][0], testing::ElementsAre(MatrixNear(Vec::Constant(1, 9.5)), MatrixNear(Vec::Constant(1, 14.0)),
                                                MatrixNear(Vec::Constant(1, 9.0))));
}

TEST(TestEnsembleMean, basic)
{
    using Vec = mio::TimeSeries<double>::Vector;

    std::vector<std::vector<mio::TimeSeries<double>>> ensemble;

    //run 1
    ensemble.emplace_back(2, mio::TimeSeries<double>(1));
    //node 1
    ensemble.back()[0].add_time_point(3.0, Vec::Constant(1, 0.0));
    ensemble.back()[0].add_time_point(4.0, Vec::Constant(1, 1.0));
    ensemble.back()[0].add_time_point(5.0, Vec::Constant(1, 2.0));
    //node 2
    ensemble.back()[1].add_time_point(3.0, Vec::Constant(1, 0.0));
    ensemble.back()[1].add_time_point(4.0, Vec::Constant(1, 1.0));
    ensemble.back()[1].add_time_point(5.0, Vec::Constant(1, 2.0));

    //run 2
    ensemble.emplace_back(2, mio::TimeSeries<double>(1));
    //node 1
    ensemble.back()[0].add_time_point(3.0, Vec::Constant(1, 0.5));
    ensemble.back()[0].add_time_point(4.0, Vec::Constant(1, 3.0));
    ensemble.back()[0].add_time_point(5.0, Vec::Constant(1, 0.0));
    //node 2
    ensemble.back()[1].add_time_point(3.0, Vec::Constant(1, 1.5));
    ensemble.back()[1].add_time_point(4.0, Vec::Constant(1, 0.5));
    ensemble.back()[1].add_time_point(5.0, Vec::Constant(1, 1.0));

    auto mean = mio::ensemble_mean(ensemble);

    ASSERT_EQ(mean.size(), 2);
    ASSERT_THAT(mean[0].get_times(), testing::ElementsAre(3.0, 4.0, 5.0));
    ASSERT_THAT(mean[0], testing::ElementsAre(MatrixNear(Vec::Constant(1, 0.25)), MatrixNear(Vec::Constant(1, 2.0)),
                                              MatrixNear(Vec::Constant(1, 1.0))));
    ASSERT_THAT(mean[1].get_times(), testing::ElementsAre(3.0, 4.0, 5.0));
    ASSERT_THAT(mean[1], testing::ElementsAre(MatrixNear(Vec::Constant(1, 0.75)), MatrixNear(Vec::Constant(1, 0.75)),
                                              MatrixNear(Vec::Constant(1, 1.5))));
}

TEST(TestEnsemblePercentile, basic)
{
    using Vec = mio::TimeSeries<double>::Vector;

    std::vector<std::vector<mio::TimeSeries<double>>> ensemble;

    //run 1
    ensemble.emplace_back(2, mio::TimeSeries<double>(2));
    //node 1
    ensemble.back()[0].add_time_point(1.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[0].add_time_point(2.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[0].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());
    //node 2
    ensemble.back()[1].add_time_point(1.0, (Vec(2) << 0.2, 0.0).finished());
    ensemble.back()[1].add_time_point(2.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[1].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());

    //run 2
    ensemble.emplace_back(2, mio::TimeSeries<double>(2));
    //node 1
    ensemble.back()[0].add_time_point(1.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[0].add_time_point(2.0, (Vec(2) << 0.0, 1.0).finished());
    ensemble.back()[0].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());
    //node 2
    ensemble.back()[1].add_time_point(1.0, (Vec(2) << 0.1, 0.0).finished());
    ensemble.back()[1].add_time_point(2.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[1].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());

    //run 3
    ensemble.emplace_back(2, mio::TimeSeries<double>(2));
    //node 1
    ensemble.back()[0].add_time_point(1.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[0].add_time_point(2.0, (Vec(2) << 0.0, 2.0).finished());
    ensemble.back()[0].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());
    //node 2
    ensemble.back()[1].add_time_point(1.0, (Vec(2) << 0.3, 0.0).finished());
    ensemble.back()[1].add_time_point(2.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[1].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());

    //run 4
    ensemble.emplace_back(2, mio::TimeSeries<double>(2));
    //node 1
    ensemble.back()[0].add_time_point(1.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[0].add_time_point(2.0, (Vec(2) << 0.0, 3.0).finished());
    ensemble.back()[0].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());
    //node 2
    ensemble.back()[1].add_time_point(1.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[1].add_time_point(2.0, (Vec(2) << 0.0, 0.0).finished());
    ensemble.back()[1].add_time_point(3.0, (Vec(2) << 0.0, 0.0).finished());

    auto q1 = mio::ensemble_percentile(ensemble, 0.2);
    auto q2 = mio::ensemble_percentile(ensemble, 0.4);
    auto q3 = mio::ensemble_percentile(ensemble, 0.7);
    auto q4 = mio::ensemble_percentile(ensemble, 0.9);

    //checking only a few elements
    ASSERT_EQ(q1.size(), 2);
    ASSERT_THAT(q1[0].get_times(), testing::ElementsAre(1.0, 2.0, 3.0));
    ASSERT_EQ(q1[0][1][1], 0.0);
    ASSERT_EQ(q1[1][0][0], 0.0);

    ASSERT_EQ(q2.size(), 2);
    ASSERT_THAT(q2[0].get_times(), testing::ElementsAre(1.0, 2.0, 3.0));
    ASSERT_EQ(q2[0][1][1], 1.0);
    ASSERT_EQ(q2[1][0][0], 0.1);

    ASSERT_EQ(q3.size(), 2);
    ASSERT_THAT(q3[0].get_times(), testing::ElementsAre(1.0, 2.0, 3.0));
    ASSERT_EQ(q3[0][1][1], 2.0);
    ASSERT_EQ(q3[1][0][0], 0.2);

    ASSERT_EQ(q4.size(), 2);
    ASSERT_THAT(q4[0].get_times(), testing::ElementsAre(1.0, 2.0, 3.0));
    ASSERT_EQ(q4[0][1][1], 3.0);
    ASSERT_EQ(q4[1][0][0], 0.3);
}

TEST(TestEnsembleParamsPercentile, basic)
{
    mio::SecirModel model(2);
    mio::SecirModel model2(2);

    auto& params = model.parameters;
    params.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)] = 3;
    params.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)] = 5;
    params.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)] = 0.2;
    params.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)] = 0.5;
    model.populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]      = 10;
    model.populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}] = 10;

    auto& params2 = model2.parameters;
    params2.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)] = 5;
    params2.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)] = 2;
    params2.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)] = 0.4;
    params2.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)] = 0.2;
    model2.populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]      = 20;
    model2.populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}] = 12;

    auto g = std::vector<mio::SecirModel>({model, model2});

    params.set<mio::Seasonality>(0.4);
    params.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)] = 4;
    params.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)] = 6;
    params.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)] = 0.3;
    params.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)] = 0.6;
    model.populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]      = 11;
    model.populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}] = 11;

    params2.set<mio::Seasonality>(0.4);
    params2.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)] = 6;
    params2.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)] = 1;
    params2.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)] = 0.5;
    params2.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)] = 0.3;
    model2.populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]      = 22;
    model2.populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}] = 14;

    auto g2 = std::vector<mio::SecirModel>({model, model2});

    auto ensemble_params = std::vector<std::vector<mio::SecirModel>>({g, g2});

    auto ensemble_p49_params = mio::ensemble_params_percentile(ensemble_params, 0.49);
    auto ensemble_p51_params = mio::ensemble_params_percentile(ensemble_params, 0.51);

    EXPECT_EQ(ensemble_p49_params[0].parameters.get<mio::Seasonality>(), 0.0);
    EXPECT_EQ(ensemble_p49_params[1].parameters.get<mio::Seasonality>(), 0.0);

    EXPECT_EQ(ensemble_p51_params[0].parameters.get<mio::Seasonality>(), 0.4);
    EXPECT_EQ(ensemble_p51_params[1].parameters.get<mio::Seasonality>(), 0.4);

    EXPECT_EQ(ensemble_p49_params[0].parameters.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)], 3.0);
    EXPECT_EQ(ensemble_p49_params[1].parameters.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)], 5.0);

    EXPECT_EQ(ensemble_p51_params[0].parameters.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)], 4.0);
    EXPECT_EQ(ensemble_p51_params[1].parameters.get<mio::ICUToDeathTime>()[mio::AgeGroup(0)], 6.0);

    EXPECT_EQ(ensemble_p49_params[0].parameters.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)], 5.0);
    EXPECT_EQ(ensemble_p49_params[1].parameters.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)], 1.0);

    EXPECT_EQ(ensemble_p51_params[0].parameters.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)], 6.0);
    EXPECT_EQ(ensemble_p51_params[1].parameters.get<mio::HospitalizedToICUTime>()[mio::AgeGroup(1)], 2.0);

    EXPECT_EQ(ensemble_p49_params[0].parameters.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)], 0.2);
    EXPECT_EQ(ensemble_p49_params[1].parameters.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)], 0.4);

    EXPECT_EQ(ensemble_p51_params[0].parameters.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)], 0.3);
    EXPECT_EQ(ensemble_p51_params[1].parameters.get<mio::RelativeCarrierInfectability>()[mio::AgeGroup(0)], 0.5);

    EXPECT_EQ(ensemble_p49_params[0].parameters.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)], 0.5);
    EXPECT_EQ(ensemble_p49_params[1].parameters.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)], 0.2);

    EXPECT_EQ(ensemble_p51_params[0].parameters.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)], 0.6);
    EXPECT_EQ(ensemble_p51_params[1].parameters.get<mio::ICUCasesPerHospitalized>()[mio::AgeGroup(1)], 0.3);

    EXPECT_EQ((ensemble_p49_params[0].populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]), 10);
    EXPECT_EQ((ensemble_p49_params[1].populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]), 20);

    EXPECT_EQ((ensemble_p51_params[0].populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]), 11);
    EXPECT_EQ((ensemble_p51_params[1].populations[{(mio::AgeGroup)0, mio::InfectionState::Exposed}]), 22);

    EXPECT_EQ((ensemble_p49_params[0].populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}]), 10);
    EXPECT_EQ((ensemble_p49_params[1].populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}]), 12);

    EXPECT_EQ((ensemble_p51_params[0].populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}]), 11);
    EXPECT_EQ((ensemble_p51_params[1].populations[{(mio::AgeGroup)1, mio::InfectionState::Hospitalized}]), 14);
}

TEST(TestDistance, same_result_zero_distance)
{
    auto n = Eigen::Index(mio::InfectionState::Count);
    std::vector<mio::TimeSeries<double>> v1(2, mio::TimeSeries<double>(n));
    v1[0].add_time_point(0.0, Eigen::VectorXd::Constant(n, 2.3));
    v1[0].add_time_point(1.0, Eigen::VectorXd::Constant(n, 2.3123));
    v1[1].add_time_point(0.0, Eigen::VectorXd::Constant(n, 3.123));
    v1[1].add_time_point(1.0, Eigen::VectorXd::Constant(n, 15151.3123));

    ASSERT_EQ(mio::result_distance_2norm(v1, v1), 0.0);
    ASSERT_EQ(mio::result_distance_2norm(v1, v1, mio::InfectionState::Exposed), 0.0);
}

TEST(TestDistance, all_compartments)
{
    auto n = Eigen::Index(mio::InfectionState::Count);
    std::vector<mio::TimeSeries<double>> v1(2, mio::TimeSeries<double>(n));
    v1[0].add_time_point(0.0, Eigen::VectorXd::Constant(n, 1.0));
    v1[0].add_time_point(1.0, Eigen::VectorXd::Constant(n, 2.0));
    v1[1].add_time_point(0.0, Eigen::VectorXd::Constant(n, 3.0));
    v1[1].add_time_point(1.0, Eigen::VectorXd::Constant(n, 4.0));

    std::vector<mio::TimeSeries<double>> v2(2, mio::TimeSeries<double>(n));
    v2[0].add_time_point(0.0, Eigen::VectorXd::Constant(n, 3.0));
    v2[0].add_time_point(1.0, Eigen::VectorXd::Constant(n, 1.0));
    v2[1].add_time_point(0.0, Eigen::VectorXd::Constant(n, 3.0));
    v2[1].add_time_point(1.0, Eigen::VectorXd::Constant(n, 10.0));

    ASSERT_EQ(mio::result_distance_2norm(v1, v2), std::sqrt(double(n) * (4.0 + 1.0 + 0.0 + 36.0)));
}

TEST(TestDistance, one_compartment)
{
    auto n = Eigen::Index(mio::InfectionState::Count);
    auto e = Eigen::Index(mio::InfectionState::Exposed);
    std::vector<mio::TimeSeries<double>> v1(2, mio::TimeSeries<double>(n));
    v1[0].add_time_point(0.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 1.0;
    v1[0].add_time_point(1.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 2.0;
    v1[1].add_time_point(0.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 3.0;
    v1[1].add_time_point(1.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 4.0;

    std::vector<mio::TimeSeries<double>> v2(2, mio::TimeSeries<double>(n));
    v2[0].add_time_point(0.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 3.0;
    v2[0].add_time_point(1.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 1.0;
    v2[1].add_time_point(0.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 3.0;
    v2[1].add_time_point(1.0, Eigen::VectorXd::Constant(n, 0.0))[e] = 10.0;

    ASSERT_EQ(mio::result_distance_2norm(v1, v2, mio::InfectionState::Exposed), std::sqrt(4.0 + 1.0 + 0.0 + 36.0));
}