// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef TW_HPP_
#define TW_HPP_

#include <twin-checker/CheckerApi.h>
#include <list>
#include <memory>
#include <string>
#include <cstring>
#include <Kokkos_Core.hpp>
#include "idefix.hpp"
#include "arrays.hpp"

#define TWTOSTR(var) (#var)
#define TWCHECK_KOKKOS_ARRAY_IMPL(fid, var) \
  do { \
    twin_register_site((fid+__LINE__), __FILE__, strlen(__FILE__)); \
    Kokkos::fence(); \
    auto tmp = Kokkos::create_mirror_view(var); \
    Kokkos::deep_copy(tmp, (var)); \
    Kokkos::fence(); \
    twin_check_double_fixable_array(tmp.data(), tmp.span(), TWTOSTR(var), \
      strlen(TWTOSTR(var)), (fid+__LINE__), __LINE__); \
    Kokkos::deep_copy((var), tmp); \
    Kokkos::fence(); \
  } while(0)

#define TWCHECK_KOKKOS_ARRAY(var) \
  TWCHECK_KOKKOS_ARRAY_IMPL((reinterpret_cast<size_t>(__FILE__ ":") * 11111 + __LINE__),(var))

#define TWCHECK_IMPL(fid, var) \
  do { \
    twin_register_site((fid), __FILE__, strlen(__FILE__)); \
    Kokkos::fence(); \
    twin_check_fixable_ptr(&var, TWTOSTR(var), strlen(TWTOSTR(var)), (fid), __LINE__); \
  } while(0)

#define TWCHECK(var) \
  TWCHECK_IMPL((reinterpret_cast<size_t>(__FILE__ ":") * 11111 + __LINE__), (var))

#define TWREGVAR(varname, var) \
  gbl_idefix_for_var_registry.addVariable(varname, var)

#define TWCHECKALL(key) \
  gbl_idefix_for_var_registry.check(reinterpret_cast<size_t>(key __FILE__), key __FILE__, __LINE__);

static inline void twin_check_fixable_ptr(int * ptr, const char * var, size_t varlen,
  int64_t fid, int line) {
  twin_check_int_fixable_ptr(ptr, var, varlen, fid, line);
}

static inline void twin_check_fixable_ptr(double * ptr, const char * var, size_t varlen,
  int64_t fid, int line) {
  twin_check_double_fixable_ptr(ptr, var, varlen, fid, line);
}

class TwVarChecker {
 public:
  TwVarChecker(void) {}
  virtual ~TwVarChecker(void) {}
  virtual void check(size_t fid, const char * file, int line) = 0;
};

template <class T>
class TwVarCheckerImpl : public TwVarChecker {
 public:
  TwVarCheckerImpl(const std::string & varname, T & value)
    :varname(varname)
    ,value(value)
  {}
  virtual ~TwVarCheckerImpl(void) {}
  virtual void check(size_t fid, const char * file, int line) {
    twin_register_site((fid+line), file, strlen(file));
    Kokkos::fence();
    auto tmp = Kokkos::create_mirror_view(this->value);
    Kokkos::deep_copy(tmp, this->value);
    Kokkos::fence();
    twin_check_double_fixable_array(tmp.data(), tmp.span(), varname.c_str(),
      strlen(varname.c_str()), (fid+line), line);
    Kokkos::deep_copy(this->value, tmp);
    Kokkos::fence();
  }
 private:
  std::string varname;
  T value;
};

class TwRegistry {
 public:
  template <class T> void addVariable(const std::string & varname, T & value) {
    this->vars.push_back(new TwVarCheckerImpl<T>(varname, value));
  }
  void check(size_t fid, const char * file, int line) {
    for (auto & var : this->vars) {
      var->check(fid, file, line);
    }
  }
  void clear(void) {
    for (auto & var : vars)
      delete var;
    this->vars.clear();
  }
 private:
  std::list< TwVarChecker * > vars;
};

extern TwRegistry gbl_idefix_for_var_registry;

template <class T>
IdefixArray1D<T> IdefixCheckedArray1D(const std::string & name, int64_t si) {
  auto res = IdefixArray1D<T>(name, si);
  gbl_idefix_for_var_registry.addVariable(name, res);
  return res;
}

template <class T>
IdefixArray2D<T> IdefixCheckedArray2D(const std::string & name, int64_t sj, int64_t si) {
  auto res = IdefixArray2D<T>(name, sj, si);
  gbl_idefix_for_var_registry.addVariable(name, res);
  return res;
}

template <class T>
IdefixArray3D<T> IdefixCheckedArray3D(const std::string & name, int64_t sk, int64_t sj,
  int64_t si) {
  auto res = IdefixArray3D<T>(name, sk, sj, si);
  gbl_idefix_for_var_registry.addVariable(name, res);
  return res;
}

template <class T>
IdefixArray4D<T> IdefixCheckedArray4D(const std::string & name, int64_t sl, int64_t sk,
  int64_t sj, int64_t si) {
  auto res = IdefixArray4D<T>(name, sl, sk, sj, si);
  gbl_idefix_for_var_registry.addVariable(name, res);
  return res;
}

#endif // TW_HPP_
