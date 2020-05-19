#ifndef MIGRATION_H
#define MIGRATION_H

#include "epidemiology/integrator.h"
#include <memory>

namespace epi
{

using MigrationFunction = std::function<void(size_t, double, Eigen::MatrixXd&)>;

/**
 * @brief Integrate ode models for multiple groups with discrete regular migration between the groups.
 * @param t0 Start time of integration.
 * @param tmax End time of integration.
 * @param dt (Initial) step size of integration.
 * @param integrators Integrators for each model to integrate, at least one.
 * @param migration_function Function that defines the migration between groups at a specified time for each variable of the model.
 * @param[in,out] result_groups On entry: initial value. On exit: result at each time step. variables for each model are interleaved e.g. [x11, x12, ..., x21, x22, ...] where xij is the j-th variable of i-th model
 */
std::vector<double> ode_integrate_with_migration(double t0, double tmax, double dt,
                                                 const std::vector<std::unique_ptr<IntegratorBase>>& integrators,
                                                 MigrationFunction migration_function,
                                                 std::vector<Eigen::VectorXd>& result);
} // namespace epi

#endif //MIGRATION_H