
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
  class dolfin_expression_e9a126c7e35f6e16e27169505b64922f : public Expression
  {
     public:
       double TAU_IN;
double TAU_OUT;
double V_MIN;
double V_MAX;


       dolfin_expression_e9a126c7e35f6e16e27169505b64922f()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = -w/TAU_IN*pow((V_m - V_MIN),2)*(V_MAX - V_m)/(V_MAX - V_MIN) + 1/TAU_OUT*(V_m - V_MIN)/(V_MAX - V_MIN);

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "TAU_IN") { TAU_IN = _value; return; }          if (name == "TAU_OUT") { TAU_OUT = _value; return; }          if (name == "V_MIN") { V_MIN = _value; return; }          if (name == "V_MAX") { V_MAX = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "TAU_IN") return TAU_IN;          if (name == "TAU_OUT") return TAU_OUT;          if (name == "V_MIN") return V_MIN;          if (name == "V_MAX") return V_MAX;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_e9a126c7e35f6e16e27169505b64922f()
{
  return new dolfin::dolfin_expression_e9a126c7e35f6e16e27169505b64922f;
}

