#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /*
   * RMSE = sqrt((1/n) * sum((xt_est - xt_true)^2))
   */

  VectorXd rmse(4);
	rmse << 0,0,0,0;

  // Confirm input vectors are useful
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		return rmse;
	}

	// Accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); i++) {
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	// Calculate the mean
	rmse = rmse/estimations.size();

	// Calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /*
   * The Jacobian is a matrix of all partial derivatives.
   *
   * We expect 3 measurement components: rho, phi, rho_dot (range, bearing, range rate),
   *  and our state has 4 components for 2D position and velocity: px, py, vx, vy
   *
   * Thus in this specific case we expect a 3x4 matrix.
   */

  MatrixXd J(3,4);
	J << 0,0,0,0,
       0,0,0,0,
       0,0,0,0;

	// Assign state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// Compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	// Check division by zero
	if(fabs(c1) < 0.0001){
		return J;
	}

	// Compute the Jacobian matrix
	J << (px/c2), (py/c2), 0, 0,
    -(py/c1), (px/c1), 0, 0,
    py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return J;
}
