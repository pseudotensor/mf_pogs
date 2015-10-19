#include <random>
#include <cstdlib>

#include "equil_helper.cuh"
#include "cml/cml_blas.cuh"
#include "cml/cml_vector.cuh"
#include "cml/cml_rand.cuh"
#include "matrix/matrix.h"
#include "matrix/matrix_fao.h"
#include "equil_helper.cuh"
#include "util.h"


namespace pogs {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

template<typename T>
struct GpuData {
  cml::vector<T> _dag_output, _dag_input;
  cml::vector<T> _d, _e;
  cublasHandle_t hdl;
  GpuData() {
    cublasCreate(&hdl);
    CUDA_CHECK_ERR();
  }
  ~GpuData() {
    // TODO get this to work.
    // cublasDestroy(hdl);
    CUDA_CHECK_ERR();
  }
};


// File scoped constants.
// const NormTypes kNormEquilibrate = kNorm2;
// const NormTypes kNormNormalize   = kNormFro;
const float MIN_SCALE = 1e-3;
const float MAX_SCALE = 1e3;

// template <typename T>
// struct GpuData {
//   const T *orig_data;
//   const POGS_INT *orig_ptr, *orig_ind;
//   GpuData(const T *data, const POGS_INT *ptr, const POGS_INT *ind)
//       : orig_data(data), orig_ptr(ptr), orig_ind(ind) { }
// };

// CBLAS_TRANSPOSE_t OpToCblasOp(char trans) {
//   ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
//   return trans == 'n' || trans == 'N' ? CblasNoTrans : CblasTrans;
// }

template <typename T>
void AddOrCopy(cml::vector<T> *y, const cml::vector<T> *x, bool clear);

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
  GpuData<T> *info = new GpuData<T>();

  info->_dag_output = cml::vector_view_array<T>(dag_output, m);
  info->_dag_input = cml::vector_view_array<T>(dag_input, n);
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

  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixFAO<T>::~MatrixFAO() {
  // TODO
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  // delete info;
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

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->hdl;
  // for (size_t i = 0; i < this->_m; ++i) {
  //   printf("D[%zu] = %e\n", i, this->_d.data[i]);
  // }
  // for (size_t i = 0; i < this->_n; ++i) {
  //   printf("E[%zu] = %e\n", i, this->_e.data[i]);
  // }
  cml::vector<T> *dag_input =
    const_cast< cml::vector<T> *>(&info->_dag_input);
  cml::vector<T> *dag_output =
    const_cast< cml::vector<T> *>(&info->_dag_output);
  cml::vector<T> x_vec, y_vec;
  // TODO factor out common code.
  if (trans == 'n' || trans == 'N') {
    x_vec = cml::vector_view_array<T>(x, this->_n);
    y_vec = cml::vector_view_array<T>(y, this->_m);

    // printf("before Amul norm(x) = %e\n", cml::blas_nrm2(hdl, &x_vec));
    // printf("before Amul norm(y) = %e\n", cml::blas_nrm2(hdl, &y_vec));

    cml::vector_memcpy<T>(dag_input, &x_vec);
    // Multiply by E.
    if (this->_done_equil) {
      cml::vector_mul<T>(dag_input, &(info->_e));
    }
    this->_Amul(this->_dag);
    cml::vector_scale<T>(dag_output, alpha);
    // Multiply by D.
    if (this->_done_equil) {
      cml::vector_mul<T>(dag_output, &(info->_d));
    }
    cml::vector_scale(&y_vec, beta);
    cml::blas_axpy(hdl, 1., dag_output, &y_vec);
  } else {
    x_vec = cml::vector_view_array<T>(x, this->_m);
    y_vec = cml::vector_view_array<T>(y, this->_n);

    // printf("before ATmul norm(x) = %e\n", cml::blas_nrm2(hdl, &x_vec));
    // printf("before ATmul norm(y) = %e\n", cml::blas_nrm2(hdl, &y_vec));

    cml::vector_memcpy<T>(dag_output, &x_vec);
    // Multiply by D.
    if (this->_done_equil) {
      cml::vector_mul<T>(dag_output, &(info->_d));
    }
    this->_ATmul(this->_dag);
    cml::vector_scale<T>(dag_input, alpha);
    // Multiply by E.
    if (this->_done_equil) {
      cml::vector_mul<T>(dag_input, &(info->_e));
    }
    cml::vector_scale<T>(&y_vec, beta);
    cml::blas_axpy(hdl, 1., dag_input, &y_vec);
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // printf("after mul norm(x) = %e\n", cml::blas_nrm2(hdl, &x_vec));
  // printf("after mul norm(y) = %e\n", cml::blas_nrm2(hdl, &y_vec));

  return 0;
}

// Populates s with random +1, -1 entries.
template <typename T>
void GenRandS(cml::vector<T> *s) {
  std::vector<T> v(s->size);
  for (size_t i = 0; i < s->size; ++i) {
    v[i] = 2*(rand() % 2) - 1;
    // TODO, this isn't really optimal
    //cml::vector_set<T>(s, i, draw);
  }
  cudaMemcpy(s->data, v.data(), s->size * sizeof(T), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();
}

template <typename T>
struct BoundF {
  T lb, ub;
  BoundF(T lb, T ub) : lb(lb), ub(ub) { }
  __host__ __device__ double operator()(double x) {
    return fmax(lb, fmin(ub, x));
  }
  __host__ __device__ float operator()(float x) {
    return fmaxf(lb, fminf(ub, x));
  }
};

template <typename T>
int MatrixFAO<T>::Equil(T *d, T *e,
                        const std::function<void(T*)> &constrain_d,
                        const std::function<void(T*)> &constrain_e) {
  DEBUG_ASSERT(this->_done_init);
  CUDA_CHECK_ERR();
  if (!this->_done_init)
    return 1;


  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->hdl;
  CUDA_CHECK_ERR();

  cml::vector<T> *dag_input =
    const_cast< cml::vector<T> *>(&info->_dag_input);
  cml::vector<T> *dag_output =
    const_cast< cml::vector<T> *>(&info->_dag_output);
  cml::vector<T> d_vec = cml::vector_view_array<T>(d, this->_m);
  cml::vector<T> e_vec = cml::vector_view_array<T>(e, this->_n);
  cml::vector_set_all<T>(&d_vec, 1.0);
  cml::vector_set_all<T>(&e_vec, 1.0);
  if (this->_equil_steps == 0) {
    return 0;
  }

  // Perform randomized Sinkhorn-Knopp equilibration.
  // alpha = (m+n)/m, beta = (m+n)/n, gamma = 0.
  T alpha = static_cast<T>(this->_m + this->_n)/static_cast<T>(this->_m);
  T beta = static_cast<T>(this->_m + this->_n)/static_cast<T>(this->_n);
  T gamma = static_cast<T>(1e-4);
  cml::vector<T> rnsATD = cml::vector_alloc<T>(this->_n);
  for (size_t i=0; i < this->_equil_steps; ++i) {
    // Set D = alpha*(|A|^2diag(E)^2 + alpha^2*gamma*1)^{-1/2}.
    {
       cml::vector_scale<T>(&d_vec, 0);
       for (size_t i = 0; i < this->_samples; ++i) {
         GenRandS(dag_input);
         cml::vector_mul<T>(dag_input, &e_vec);
         this->_Amul(this->_dag);
         cml::vector_mul<T>(dag_output, dag_output);
         cml::blas_axpy(hdl, 1., dag_output, &d_vec);
         cudaDeviceSynchronize();
         CUDA_CHECK_ERR();
       }
       // printf("before div norm d_vec AE = %e\n", cml::blas_nrm2(hdl, &d_vec));
       cml::vector_scale<T>(&d_vec, 1.0/static_cast<T>(this->_samples));
       // printf("norm d_vec AE = %e\n", cml::blas_nrm2(hdl, &d_vec));
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    //RandRnsAE(&d_vec, &e_vec);
    // cml::vector_add_constant<T>(&d_vec, alpha*alpha*gamma);
    cml::vector_add_constant<T>(&d_vec, alpha*gamma);

    thrust::transform(thrust::device_pointer_cast(d_vec.data),
           thrust::device_pointer_cast(d_vec.data + this->_m),
           thrust::device_pointer_cast(d_vec.data), SqrtF<T>());
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    // Round D's entries to be in [MIN_SCALE, MAX_SCALE].
    thrust::transform(thrust::device_pointer_cast(d_vec.data),
           thrust::device_pointer_cast(d_vec.data + this->_m),
           thrust::device_pointer_cast(d_vec.data),
           BoundF<T>(MIN_SCALE, MAX_SCALE));
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    thrust::transform(thrust::device_pointer_cast(d_vec.data),
           thrust::device_pointer_cast(d_vec.data + this->_m),
           thrust::device_pointer_cast(d_vec.data), ReciprF<T>());
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    // Force D to respect cone boundaries.
    constrain_d(d_vec.data);
    // cml::vector_scale<T>(&d_vec, alpha);
    // Set E = beta*(|A^T|^2diag(D)^2 + gamma*beta^2*1)^{-1/2}.
    {
        cml::vector_scale<T>(&e_vec, 0);
        for (size_t i = 0; i < this->_samples; ++i) {
          GenRandS(dag_output);
          cml::vector_mul<T>(dag_output, &d_vec);
          this->_ATmul(this->_dag);
          cml::vector_mul<T>(dag_input, dag_input);
          // printf("ATD e_vec[0] = %e\n", e_vec->data[0]);
          cml::blas_axpy(hdl, 1., dag_input, &e_vec);
          //cml::vector_add<T>(e_vec, dag_input);
          cudaDeviceSynchronize();
        }
        // printf("before div norm e_vec ATD = %e\n", cml::blas_nrm2(hdl, &e_vec));
        cml::vector_scale<T>(&e_vec, 1.0/static_cast<T>(this->_samples));
        // printf("norm e_vec ATD = %e\n", cml::blas_nrm2(hdl, &e_vec));
    }
    //RandRnsATD(&e_vec, &d_vec);
    // Save the row norms squared of A^TD.
    cml::vector_memcpy<T>(&rnsATD, &e_vec);
    // cml::vector_add_constant<T>(&e_vec, beta*beta*gamma);
    cml::vector_add_constant<T>(&e_vec, beta*gamma);
    thrust::transform(thrust::device_pointer_cast(e_vec.data),
           thrust::device_pointer_cast(e_vec.data + this->_n),
           thrust::device_pointer_cast(e_vec.data), SqrtF<T>());
    // Round E's entries to be in [MIN_SCALE, MAX_SCALE].
    thrust::transform(thrust::device_pointer_cast(e_vec.data),
           thrust::device_pointer_cast(e_vec.data + this->_n),
           thrust::device_pointer_cast(e_vec.data),
           BoundF<T>(MIN_SCALE, MAX_SCALE));
    thrust::transform(thrust::device_pointer_cast(e_vec.data),
           thrust::device_pointer_cast(e_vec.data + this->_n),
           thrust::device_pointer_cast(e_vec.data), ReciprF<T>());
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    // Force E to respect cone boundaries.
    constrain_e(e_vec.data);
    // cml::vector_scale(&e_vec, beta);
  }

  // Scale A to have Frobenius norm of 1.
  cml::vector_mul<T>(&rnsATD, &e_vec);
  T normA;
  cml::blas_dot(hdl, &rnsATD, &e_vec, &normA);
  printf("normA = %e before div\n", normA);
  // POGS divides by sqrt(min(m,n)) for some reason.
  normA = std::sqrt(normA)/std::sqrt(std::min(this->_m, this->_n));
  // T normA = Norm2Est(hdl, this);
  // Scale d and e to account for normalization of A.
  cml::vector_scale(&d_vec, 1 / std::sqrt(normA));
  cml::vector_scale(&e_vec, 1 / std::sqrt(normA));
  printf("normA = %e, norm(d) = %e, norm(e) = %e\n", normA,
    cml::blas_nrm2(hdl, &d_vec), cml::blas_nrm2(hdl, &e_vec));

  // Save D and E.
  this->_done_equil = true;
  info->_d = d_vec;
  info->_e = e_vec;

  DEBUG_PRINTF("norm A = %e, normd = %e, norme = %e\n", normA,
      cml::blas_nrm2(hdl, &d_vec), cml::blas_nrm2(hdl, &e_vec));

  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  return 0;
}





// Estimate the row norm squared of AE.
// template <typename T>
// void MatrixFAO<T>::RandRnsAE(MatrixFAO<T> *mat, cml::vector<T> *output,
//   const cml::vector<T> *e) const {
//
//   GpuData<T> *info = reinterpret_cast<GpuData<T>*>(mat->_info);
//
//   cml::vector_scale<T>(output, 0);
//   cml::vector<T> *dag_input =
//     const_cast< cml::vector<T> *>(&info->_dag_input);
//   cml::vector<T> *dag_output =
//     const_cast< cml::vector<T> *>(&info->_dag_output);
//   for (size_t i = 0; i < this->_samples; ++i) {
//     GenRandS(dag_input);
//     cml::vector_mul<T>(dag_input, e);
//     this->_Amul(this->_dag);
//     cml::vector_mul<T>(dag_output, dag_output);
//     // printf("AE output[0] = %e\n", output->data[0]);
//     cml::vector_add<T>(output, dag_output);
//   }
//   printf("before div norm output AE = %e\n", cml::blas_nrm2(output));
//   cml::vector_scale<T>(output, 1.0/static_cast<T>(this->_samples));
//   printf("norm output AE = %e\n", cml::blas_nrm2(output));
// }

// Estimate the row norm squared of A^TD.
// template <typename T>
// void MatrixFAO<T>::RandRnsATD(cml::vector<T> *output,
//   const cml::vector<T> *d) const {
//
//   GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
//
//   cml::vector_scale<T>(output, 0);
//   cml::vector<T> *dag_input =
//     const_cast< cml::vector<T> *>(&info->_dag_input);
//   cml::vector<T> *dag_output =
//     const_cast< cml::vector<T> *>(&info->_dag_output);
//   for (size_t i = 0; i < this->_samples; ++i) {
//     GenRandS(dag_output);
//     cml::vector_mul<T>(dag_output, d);
//     this->_ATmul(this->_dag);
//     cml::vector_mul<T>(dag_input, dag_input);
//     // printf("ATD output[0] = %e\n", output->data[0]);
//     cml::vector_add<T>(output, dag_input);
//   }
//   printf("before div norm output ATD = %e\n", cml::blas_nrm2(output));
//   cml::vector_scale<T>(output, 1.0/static_cast<T>(this->_samples));
//   printf("norm output ATD = %e\n", cml::blas_nrm2(output));
// }

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
// template <typename T>
// void AddOrCopy(cml::vector<T> *y, const cml::vector<T> *x, bool copy) {
//     if (copy) {
//       cml::vector_memcpy<T>(y, x);
//     } else {
//       cml::vector_add<T>(y, x);
//     }
// }

}  // namespace

#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class MatrixFAO<double>;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class MatrixFAO<float>;
#endif

}  // namespace pogs

