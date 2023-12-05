
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_c58991824c05009ae5f71c090a89028b : public Expression
  {
     public:
       double d;
std::shared_ptr<dolfin::GenericFunction> generic_function_sigma_it;
std::shared_ptr<dolfin::GenericFunction> generic_function_sigma_il;
std::shared_ptr<dolfin::GenericFunction> generic_function_fib;


       dolfin_expression_c58991824c05009ae5f71c090a89028b()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          double fib[2];

            generic_function_fib->eval(Eigen::Map<Eigen::Matrix<double, 2, 1>>(fib), x);
          double sigma_il;
            generic_function_sigma_il->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&sigma_il), x);
          double sigma_it;
            generic_function_sigma_it->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&sigma_it), x);
          values[0] = sigma_it * Identity(d) + (sigma_il - sigma_it) * outer(fib(msh), fib(msh);

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "d") { d = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "d") return d;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {
          if (name == "sigma_it") { generic_function_sigma_it = _value; return; }          if (name == "sigma_il") { generic_function_sigma_il = _value; return; }          if (name == "fib") { generic_function_fib = _value; return; }
       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {
          if (name == "sigma_it") return generic_function_sigma_it;          if (name == "sigma_il") return generic_function_sigma_il;          if (name == "fib") return generic_function_fib;
       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_c58991824c05009ae5f71c090a89028b()
{
  return new dolfin::dolfin_expression_c58991824c05009ae5f71c090a89028b;
}

