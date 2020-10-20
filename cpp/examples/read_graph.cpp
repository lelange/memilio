#include <epidemiology/migration/migration.h>
#include <epidemiology_io/secir_parameters_io.h>
#include <epidemiology_io/twitter_migration_io.h>

#include <iostream>

int main(int argc, char** argv)
{
    const auto t0   = 0.;
    const auto tmax = 10.;
    const auto dt   = 1.; //time step of migration, not integration

    double tinc    = 5.2, // R_2^(-1)+R_3^(-1)
        tinfmild   = 6, // 4-14  (=R4^(-1))
        tserint    = 4.2, // 4-4.4 // R_2^(-1)+0.5*R_3^(-1)
        thosp2home = 12, // 7-16 (=R5^(-1))
        thome2hosp = 5, // 2.5-7 (=R6^(-1))
        thosp2icu  = 2, // 1-3.5 (=R7^(-1))
        ticu2home  = 8, // 5-16 (=R8^(-1))
        tinfasy    = 6.2, // (=R9^(-1)=R_3^(-1)+0.5*R_4^(-1))
        ticu2death = 5; // 3.5-7 (=R5^(-1))

    double tinfasy2 = 1.0 / (0.5 / (tinfmild - tserint) + 0.5 / tinfmild);
    if (fabs(tinfasy2 - tinfasy) > 0) {
        epi::log_warning("----> TODO / To consider: In the HZI paper, tinfasy (the asymptomatic infectious time) or "
                         "R9^(-1)=R_3^(-1)+0.5*R_4^(-1) is directly given by R_3 and R_4 and maybe should not be an "
                         "'additional parameter'");
    }

    double cont_freq = 10, // see Polymod study
        inf_prob = 0.05, carr_infec = 0.67,
           alpha = 0.09, // 0.01-0.16
        beta     = 0.25, // 0.05-0.5
        delta    = 0.3, // 0.15-0.77
        rho      = 0.2, // 0.1-0.35
        theta    = 0.25; // 0.15-0.4

    double nb_total_t0 = 10000, nb_exp_t0 = 100, nb_inf_t0 = 50, nb_car_t0 = 50, nb_hosp_t0 = 20, nb_icu_t0 = 10,
           nb_rec_t0 = 10, nb_dead_t0 = 0;

    epi::SecirModel<epi::AgeGroup1> model;
    int nb_groups = model.parameters.get_num_groups();
    double fact   = 1.0 / (double)nb_groups;

    auto& params = model.parameters;

    params.set_icu_capacity(std::numeric_limits<double>::max());
    params.set_start_day(0);
    params.set_seasonality(0);

    for (size_t i = 0; i < nb_groups; i++) {
        params.times[i].set_incubation(tinc);
        params.times[i].set_infectious_mild(tinfmild);
        params.times[i].set_serialinterval(tserint);
        params.times[i].set_hospitalized_to_home(thosp2home);
        params.times[i].set_home_to_hospitalized(thome2hosp);
        params.times[i].set_hospitalized_to_icu(thosp2icu);
        params.times[i].set_icu_to_home(ticu2home);
        params.times[i].set_infectious_asymp(tinfasy);
        params.times[i].set_icu_to_death(ticu2death);

        model.populations.set(fact * nb_exp_t0, (epi::AgeGroup1)i, epi::InfectionType::E);
        model.populations.set(fact * nb_car_t0, (epi::AgeGroup1)i, epi::InfectionType::C);
        model.populations.set(fact * nb_inf_t0, (epi::AgeGroup1)i, epi::InfectionType::I);
        model.populations.set(fact * nb_hosp_t0, (epi::AgeGroup1)i, epi::InfectionType::H);
        model.populations.set(fact * nb_icu_t0, (epi::AgeGroup1)i, epi::InfectionType::U);
        model.populations.set(fact * nb_rec_t0, (epi::AgeGroup1)i, epi::InfectionType::R);
        model.populations.set(fact * nb_dead_t0, (epi::AgeGroup1)i, epi::InfectionType::D);
        model.populations.set_difference_from_group_total(fact * nb_total_t0, (epi::AgeGroup1)i, (epi::AgeGroup1)i,
                                                          epi::InfectionType::S);

        params.probabilities[i].set_infection_from_contact(inf_prob);
        params.probabilities[i].set_carrier_infectability(carr_infec);
        params.probabilities[i].set_asymp_per_infectious(alpha);
        params.probabilities[i].set_risk_from_symptomatic(beta);
        params.probabilities[i].set_hospitalized_per_infectious(rho);
        params.probabilities[i].set_icu_per_hospitalized(theta);
        params.probabilities[i].set_dead_per_icu(delta);
    }

    epi::ContactFrequencyMatrix& cont_freq_matrix = params.get_contact_patterns();
    epi::Damping dummy(30., 0.3);
    for (int i = 0; i < nb_groups; i++) {
        for (int j = i; j < nb_groups; j++) {
            cont_freq_matrix.set_cont_freq(fact * cont_freq, i, j);
        }
    }

    std::cout << "Readimg Migration File..." << std::flush;
    Eigen::MatrixXi twitter_migration_2018 = epi::read_migration("2018_lk_matrix.txt");
    std::cout << "Done" << std::endl;

    std::cout << "Intializing Graph..." << std::flush;
    epi::Graph<epi::SecirModel<epi::AgeGroup1>, epi::MigrationEdge> graph;
    for (int node = 0; node < twitter_migration_2018.rows(); node++) {
        graph.add_node(model);
    }
    for (int row = 0; row < twitter_migration_2018.rows(); row++) {
        for (int col = 0; col < twitter_migration_2018.cols(); col++) {
            graph.add_edge(row, col, Eigen::VectorXd::Constant(8 * nb_groups, twitter_migration_2018(row, col)));
        }
    }
    std::cout << "Done" << std::endl;

    std::cout << "Writing XML Files..." << std::flush;
    epi::write_graph(graph);
    std::cout << "Done" << std::endl;

#ifndef EPI_NO_IO
    std::cout << "Reading XML Files..." << std::flush;
    epi::Graph<epi::SecirModel<epi::AgeGroup1>, epi::MigrationEdge> graph_read =
        epi::read_graph<epi::SecirModel<epi::AgeGroup1>>();
    std::cout << "Done" << std::endl;

    std::cout << "Running Simulations..." << std::flush;
    auto study = epi::ParameterStudy<epi::SecirModel<epi::AgeGroup1>>(graph_read, t0, tmax, 1.0, 2);
    std::cout << "Done" << std::endl;
#endif
    return 0;
}
