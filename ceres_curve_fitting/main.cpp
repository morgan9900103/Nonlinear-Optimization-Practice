#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

/*  example for cost function modelling
*   f(x1, x2, y1, y2) = x1*y1 + x2*y2 - 10
*
*   struct F{
*       template <typename T>
*       bool operator()(const T* x, const T* y, T* output) {
*           output[0] = x[0]*y[0] + x[1]*y[1] - T(10.0);
*           rturn true;
}       }
*   }
*/

// cost function model
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    template<typename T>
    bool operator()(const T* const abc, T* residual) const {
        // y - exp(ax^2 + bx + c)
        residual[0] = T(_y) - ceres::exp(abc[0]*T(_x) *T(_x) + abc[1]*T(_x) + abc[2]);
        return true;
    }
    const double _x, _y;
};

int main(int argc, char** argv) {
    double a = 1.0, b = 2.0, c = 1.0;   // True values
    int N = 100;                        // Sample number
    double w_sigma = 1.0;               // Noise variance
    cv::RNG rng;                        // OpenCV random number generator
    double abc[3] = {0, 0, 0};          // abc parameters estimation

    vector<double> x_data, y_data;      // data
    cout << "generating data: " << endl;
    for(int i = 0; i < N; i++) {
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(exp(a*x*x + b*x + c) + rng.gaussian(w_sigma));
        cout << x_data[i] << " " << y_data[i] << endl;
    }

    // Model the Least Square problem
    ceres::Problem problem;
    for(int i = 0; i < N; i++) {
        // Problem::AddResidualBlock(Costfunction* cost_function, LossFunction* loss_function, const vector<double*> parameter_block)
        problem.AddResidualBlock(
            // auto differentiate cost function
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),
            nullptr,    // loss function
            abc         // parameter block
        );
    }

    // Solver
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // Starting optimization
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds." << endl;

    // Output the result
    cout << summary.BriefReport() << endl;
    // cout << summary.FullReport() << endl;
    cout << "estimated a, b, c = ";
    for(auto& a:abc)
        cout << a << " ";
    cout << endl;

    return 0;
}
