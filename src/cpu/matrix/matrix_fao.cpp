#include "gsl/gsl_vector.h"
#include "gsl/gsl_blas.h"
#include "util.h"
#include "equil_helper.h"
#include "matrix/matrix.h"
#include "matrix/matrix_fao.h"
#include <random>
#include <cstdlib>

namespace pogs {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// File scoped constants.
// const NormTypes kNormEquilibrate = kNorm2;
// const NormTypes kNormNormalize   = kNormFro;
const float MIN_SCALE = 1e-3;
const float MAX_SCALE = 1e3;

// template <typename T>
// struct CpuData {
//   const T *orig_data;
//   const POGS_INT *orig_ptr, *orig_ind;
//   CpuData(const T *data, const POGS_INT *ptr, const POGS_INT *ind)
//       : orig_data(data), orig_ptr(ptr), orig_ind(ind) { }
// };

// CBLAS_TRANSPOSE_t OpToCblasOp(char trans) {
//   ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
//   return trans == 'n' || trans == 'N' ? CblasNoTrans : CblasTrans;
// }

template <typename T>
void AddOrCopy(gsl::vector<T> *y, const gsl::vector<T> *x, bool clear);

// template <typename T>
// T NormEst(NormTypes norm_type, const MatrixFAO<T>& A);

}  // namespace

////////////////////////////////////////////////////////////////////////////////
/////////////////////// MatrixFAO Implementation /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
MatrixFAO<T>::MatrixFAO(T *dag_output, size_t m, T *dag_input, size_t n,
                        void (*Amul)(void *), void (*ATmul)(void *),
                        void *dag, size_t samples,
                        size_t equil_steps) :  Matrix<T>(m, n),
                        _samples(samples), _equil_steps(equil_steps) {
  this->_dag_output = gsl::vector_view_array<T>(dag_output, m);
  this->_dag_input = gsl::vector_view_array<T>(dag_input, n);
  this->_Amul = Amul;
  this->_ATmul = ATmul;
  this->_dag = dag;
  this->_done_equil = false;
  // this->_samples = samples;
  // this->_equil_steps = equil_steps;
  // TODO why do I need to set this here?
  this->_m = m;
  this->_n = n;
  // TODO move this.
  srand(1);
}

template <typename T>
MatrixFAO<T>::~MatrixFAO() {
  // TODO
}

template <typename T>
int MatrixFAO<T>::Init() {
  DEBUG_ASSERT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;
  return 0;
}

template <typename T>
int MatrixFAO<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;
  // for (size_t i = 0; i < this->_m; ++i) {
  //   printf("D[%zu] = %e\n", i, this->_d.data[i]);
  // }
  // for (size_t i = 0; i < this->_n; ++i) {
  //   printf("E[%zu] = %e\n", i, this->_e.data[i]);
  // }
  gsl::vector<T> *dag_input =
    const_cast< gsl::vector<T> *>(&this->_dag_input);
  gsl::vector<T> *dag_output =
    const_cast< gsl::vector<T> *>(&this->_dag_output);
  gsl::vector<T> x_vec, y_vec;
  // TODO factor out common code.
  if (trans == 'n' || trans == 'N') {
    x_vec = gsl::vector_view_array<T>(x, this->_n);
    y_vec = gsl::vector_view_array<T>(y, this->_m);
    gsl::vector_memcpy<T>(dag_input, &x_vec);
    // Multiply by E.
    if (this->_done_equil) {
      gsl::vector_mul<T>(dag_input, &(this->_e));
    }
    this->_Amul(this->_dag);
    gsl::vector_scale<T>(dag_output, alpha);
    // Multiply by D.
    if (this->_done_equil) {
      gsl::vector_mul<T>(dag_output, &(this->_d));
    }
    gsl::vector_scale(&y_vec, beta);
    gsl::vector_add<T>(&y_vec, dag_output);
  } else {
    x_vec = gsl::vector_view_array<T>(x, this->_m);
    y_vec = gsl::vector_view_array<T>(y, this->_n);
    gsl::vector_memcpy<T>(dag_output, &x_vec);
    // Multiply by D.
    if (this->_done_equil) {
      gsl::vector_mul<T>(dag_output, &(this->_d));
    }
    this->_ATmul(this->_dag);
    gsl::vector_scale<T>(dag_input, alpha);
    // Multiply by E.
    if (this->_done_equil) {
      gsl::vector_mul<T>(dag_input, &(this->_e));
    }
    gsl::vector_scale<T>(&y_vec, beta);
    gsl::vector_add<T>(&y_vec, dag_input);
  }
  return 0;
}

template <typename T>
int MatrixFAO<T>::Equil(T *d, T *e,
                        const std::function<void(T*)> &constrain_d,
                        const std::function<void(T*)> &constrain_e) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  gsl::vector<T> d_vec = gsl::vector_view_array<T>(d, this->_m);
  gsl::vector<T> e_vec = gsl::vector_view_array<T>(e, this->_n);
  gsl::vector_set_all<T>(&d_vec, 1.0);
  gsl::vector_set_all<T>(&e_vec, 1.0);
  // Perform randomized Sinkhorn-Knopp equilibration.
  // alpha = n, beta = m, gamma = 0.
  T alpha = static_cast<T>(this->_n);
  T beta = static_cast<T>(this->_m);
  // T gamma = static_cast<T>(0.);
  gsl::vector<T> rnsATD = gsl::vector_alloc<T>(this->_n);
  for (size_t i=0; i < this->_equil_steps; ++i) {
    // Set D = alpha*(|A|^2diag(E)^2 + alpha^2*gamma*1)^{-1/2}.
    RandRnsAE(&d_vec, &e_vec);
    // gsl::vector_add_constant<T>(&d_vec, alpha*alpha*gamma);
    std::transform(d_vec.data, d_vec.data + this->_m, d_vec.data, SqrtF<T>());
    // Round D's entries to be in [MIN_SCALE, MAX_SCALE].
    gsl::vector_bound<T>(&d_vec, MIN_SCALE, MAX_SCALE);
    std::transform(d_vec.data, d_vec.data + this->_m, d_vec.data, ReciprF<T>());
    // gsl::vector_scale<T>(&d_vec, alpha);
    // Set E = beta*(|A^T|^2diag(D)^2 + gamma*beta^2*1)^{-1/2}.
    RandRnsATD(&e_vec, &d_vec);
    // Save the row norms squared of A^TD.
    gsl::vector_memcpy<T>(&rnsATD, &e_vec);
    // gsl::vector_add_constant<T>(&e_vec, beta*beta*gamma);
    std::transform(e_vec.data, e_vec.data + this->_n, e_vec.data, SqrtF<T>());
    // Round E's entries to be in [MIN_SCALE, MAX_SCALE].
    gsl::vector_bound<T>(&e_vec, MIN_SCALE, MAX_SCALE);
    std::transform(e_vec.data, e_vec.data + this->_n, e_vec.data, ReciprF<T>());
    // gsl::vector_scale(&e_vec, beta);
  }

  // Scale A to have Frobenius norm of 1.
  gsl::vector_mul<T>(&rnsATD, &e_vec);
  T normA;
  gsl::blas_dot<T>(&rnsATD, &e_vec, &normA);
  printf("normA = %e\n", normA);
  // Scale d and e to account for normalization of A.
  gsl::vector_scale(&d_vec, 1 / std::sqrt(normA));
  gsl::vector_scale(&e_vec, 1 / std::sqrt(normA));
  printf("norm(d) = %e\n", gsl::blas_nrm2(&d_vec));
  printf("norm(e) = %e\n", gsl::blas_nrm2(&e_vec));

  // gsl::vector_set_all<T>(&d_vec, 1.0);
  // gsl::vector_set_all<T>(&e_vec, 1.0);
  // Save D and E.
  this->_done_equil = true;
  this->_d = d_vec;
  this->_e = e_vec;
  printf("D[0] = %e\n", this->_d.data[0]);
  printf("E[0] = %e\n", this->_e.data[0]);

  DEBUG_PRINTF("norm A = %e, normd = %e, norme = %e\n", normA,
      gsl::blas_nrm2(&d_vec), gsl::blas_nrm2(&e_vec));

  return 0;
}


// Populates s with random +1, -1 entries.
template <typename T>
void MatrixFAO<T>::GenRandS(gsl::vector<T> *s) const {
  for (size_t i = 0; i < s->size; ++i) {
    int draw = 2*(rand() % 2) - 1;
    gsl::vector_set<T>(s, i, draw);
  }
}


// Estimate the row norm squared of AE.
template <typename T>
void MatrixFAO<T>::RandRnsAE(gsl::vector<T> *output,
  const gsl::vector<T> *e) const {
  gsl::vector_scale<T>(output, 0);
  gsl::vector<T> *dag_input =
    const_cast< gsl::vector<T> *>(&this->_dag_input);
  gsl::vector<T> *dag_output =
    const_cast< gsl::vector<T> *>(&this->_dag_output);
  for (size_t i = 0; i < this->_samples; ++i) {
    GenRandS(dag_input);
    gsl::vector_mul<T>(dag_input, e);
    this->_Amul(this->_dag);
    gsl::vector_mul<T>(dag_output, dag_output);
    // printf("AE output[0] = %e\n", output->data[0]);
    gsl::vector_add<T>(output, dag_output);
  }
  printf("before div norm output AE = %e\n", gsl::blas_nrm2(output));
  gsl::vector_scale<T>(output, 1.0/static_cast<T>(this->_samples));
  printf("norm output AE = %e\n", gsl::blas_nrm2(output));
}

// Estimate the row norm squared of A^TD.
template <typename T>
void MatrixFAO<T>::RandRnsATD(gsl::vector<T> *output,
  const gsl::vector<T> *d) const {
  gsl::vector_scale<T>(output, 0);
  gsl::vector<T> *dag_input =
    const_cast< gsl::vector<T> *>(&this->_dag_input);
  gsl::vector<T> *dag_output =
    const_cast< gsl::vector<T> *>(&this->_dag_output);
  for (size_t i = 0; i < this->_samples; ++i) {
    GenRandS(dag_output);
    gsl::vector_mul<T>(dag_output, d);
    this->_ATmul(this->_dag);
    gsl::vector_mul<T>(dag_input, dag_input);
    // printf("ATD output[0] = %e\n", output->data[0]);
    gsl::vector_add<T>(output, dag_input);
  }
  printf("before div norm output ATD = %e\n", gsl::blas_nrm2(output));
  gsl::vector_scale<T>(output, 1.0/static_cast<T>(this->_samples));
  printf("norm output ATD = %e\n", gsl::blas_nrm2(output));
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Equilibration Helpers //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// // Estimates norm of A. norm_type should either be kNorm2 or kNormFro.
// template <typename T>
// T NormEst(NormTypes norm_type, const MatrixFAO<T>& A) {
//   switch (norm_type) {
//     case kNorm2: {
//       return 1;
//     }
//     case kNormFro: {
//       return 1;
//     }
//     case kNorm1:
//       // 1-norm normalization doens't make make sense since it treats rows and
//       // columns differently.
//     default:
//       ASSERT(false);
//       return static_cast<T>(0.);
//   }
// }

// Either y += x or y = x.
template <typename T>
void AddOrCopy(gsl::vector<T> *y, const gsl::vector<T> *x, bool copy) {
    if (copy) {
      gsl::vector_memcpy<T>(y, x);
    } else {
      gsl::vector_add<T>(y, x);
    }
}

}  // namespace

#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class MatrixFAO<double>;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class MatrixFAO<float>;
#endif

}  // namespace pogs

