#include <cstddef>
#include <cmath>
#include <immintrin.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <array>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <tuple>

namespace py = pybind11;

template <typename T = float,
          bool UseAVX2 = true,
          size_t MaxStackSegments = 32>
void pts_to_lines(
    const T* pts_x, const T* pts_z, size_t n_pts,
    const T* segments_data, size_t n_segments,
    T* distances = nullptr,
    T* closest_x = nullptr, T* closest_z = nullptr,
    int* segment_indices = nullptr,
    T* projection_scalars = nullptr);



template <typename T>
auto project_on_polyline(
    py::array_t<T, py::array::c_style | py::array::forcecast> pts,
    py::array_t<T, py::array::c_style | py::array::forcecast> segments_arr,
    bool return_distance = true,
    bool return_projection = true,
    bool return_param = true,
    bool return_index = true)
{
    // Check input dimensions
    if (pts.ndim() != 2 || pts.shape(1) != 2) {
        throw std::runtime_error("pts must be shape (N, 2)");
    }
    if (segments_arr.ndim() != 3) {
        throw std::runtime_error("segments_arr must be 3-dimensional");
    }
    if (segments_arr.shape(1) != 2 || segments_arr.shape(2) != 2) {
        throw std::runtime_error("segments_arr must be shape (M, 2, 2)");
    }

    const size_t n_pts = pts.shape(0);
    const size_t n_segments = segments_arr.shape(0);

    if (n_segments == 0) {
        throw std::runtime_error("segments_arr must contain at least one segment");
    }

    // Get direct pointers to input data
    const T* pts_ptr = static_cast<const T*>(pts.data());
    const T* segments_data_ptr = static_cast<const T*>(segments_arr.data());

    // Temporary storage for de-interleaved points
    std::vector<T> pts_x(n_pts), pts_z(n_pts);
    const T* pts_x_ptr;
    const T* pts_z_ptr;

    // De-interleave pts into separate x and z arrays
    // as pts_to_lines expects separate contiguous arrays for x and z coordinates.
    auto pts_unchecked = pts.unchecked<2>();
    for (size_t i = 0; i < n_pts; ++i) {
        pts_x[i] = pts_unchecked(i, 0);
        pts_z[i] = pts_unchecked(i, 1);
    }
    pts_x_ptr = pts_x.data();
    pts_z_ptr = pts_z.data();

    // Prepare output arrays based on what's requested
    py::array_t<T> distances, projection_x, projection_z, params;
    py::array_t<int32_t> indices;

    T* dist_ptr = nullptr;
    T* proj_x_ptr = nullptr;
    T* proj_z_ptr = nullptr;
    T* param_ptr = nullptr;
    int* idx_ptr = nullptr;

    if (return_distance) {
        distances = py::array_t<T>(n_pts);
        dist_ptr = static_cast<T*>(distances.mutable_unchecked<1>().mutable_data(0));
    }

    if (return_projection) {
        projection_x = py::array_t<T>(n_pts);
        projection_z = py::array_t<T>(n_pts);
        proj_x_ptr = static_cast<T*>(projection_x.mutable_unchecked<1>().mutable_data(0));
        proj_z_ptr = static_cast<T*>(projection_z.mutable_unchecked<1>().mutable_data(0));
    }

    if (return_param) {
        params = py::array_t<T>(n_pts);
        param_ptr = static_cast<T*>(params.mutable_unchecked<1>().mutable_data(0));
    }

    if (return_index) {
        indices = py::array_t<int32_t>(n_pts);
        idx_ptr = static_cast<int32_t*>(indices.mutable_unchecked<1>().mutable_data(0));
    }

    // Simple single call - pts_to_lines will use nullptr checks for runtime dispatch
    pts_to_lines<T, true, 32>(
        pts_x_ptr, pts_z_ptr, n_pts,
        segments_data_ptr, n_segments,
        dist_ptr, proj_x_ptr, proj_z_ptr, idx_ptr, param_ptr);

    // Build return dictionary based on requested outputs
    py::dict result;

    if (return_distance) {
        result["distances"] = distances;
    }

    if (return_projection) {
        // Stack x and z into (N, 2) array
        py::array_t<T> projection({int(n_pts), int(2)});
        auto proj_unchecked = projection.mutable_unchecked<2>();
        for (size_t i = 0; i < n_pts; ++i) {
            proj_unchecked(i, 0) = proj_x_ptr[i];
            proj_unchecked(i, 1) = proj_z_ptr[i];
        }
        result["projections"] = projection;
    }

    if (return_param) {
        result["params"] = params;
    }

    if (return_index) {
        result["indices"] = indices;
    }

    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Point to polyline projection utilities";

    m.def("project_on_polyline",
          [](py::array pts, py::array seg,
             bool return_distance, bool return_projection,
             bool return_param, bool return_index) {
              // Check if inputs are float32 or float64
              if (pts.dtype().is(py::dtype::of<float>()) &&
                  seg.dtype().is(py::dtype::of<float>())) {
                  return project_on_polyline<float>(
                      pts.cast<py::array_t<float>>(),
                      seg.cast<py::array_t<float>>(),
                      return_distance, return_projection,
                      return_param, return_index);
              } else if (pts.dtype().is(py::dtype::of<double>()) &&
                         seg.dtype().is(py::dtype::of<double>())) {
                  return project_on_polyline<double>(
                      pts.cast<py::array_t<double>>(),
                      seg.cast<py::array_t<double>>(),
                      return_distance, return_projection,
                      return_param, return_index);
              } else if (pts.dtype().is(py::dtype::of<double>()) ||
                         seg.dtype().is(py::dtype::of<double>())) {
                  // If either is float64, convert both to float64
                  return project_on_polyline<double>(
                      pts.cast<py::array_t<double>>(),
                      seg.cast<py::array_t<double>>(),
                      return_distance, return_projection,
                      return_param, return_index);
              } else {
                  // Default to float32 for other types
                  return project_on_polyline<float>(
                      pts.cast<py::array_t<float>>(),
                      seg.cast<py::array_t<float>>(),
                      return_distance, return_projection,
                      return_param, return_index);
              }
          },
          py::arg("pts"),
          py::arg("seg"),
          py::arg("return_distance") = true,
          py::arg("return_projection") = true,
          py::arg("return_param") = true,
          py::arg("return_index") = true,
          R"pbdoc(
          Project points onto a polyline and compute various properties.

          The function automatically uses float32 or float64 precision based on input dtypes.
          If either input is float64, the computation is done in float64.

          Parameters
          ----------
          pts : array_like, shape (N, 2)
              Points to project, with columns [x, z]
          seg : array_like, shape (M, 2)
              Polyline vertices, with columns [x, z]
          return_distance : bool, optional
              If True, return distances from points to polyline
          return_projection : bool, optional
              If True, return projected points on polyline
          return_param : bool, optional
              If True, return projection parameter t in [0, 1] along each segment
          return_index : bool, optional
              If True, return index of closest segment

          Returns
          -------
          dict
              Dictionary containing requested outputs:
              - 'distance': array of shape (N,) with distances
              - 'projection': array of shape (N, 2) with projected points
              - 'param': array of shape (N,) with projection parameters
              - 'index': array of shape (N,) with segment indices
          )pbdoc");
}

template <typename T,
          bool UseAVX2,
          size_t MaxStackSegments>
void pts_to_lines(
    const T* pts_x, const T* pts_z, size_t n_pts,
    const T* segments_data, size_t n_segments,
    T* distances,
    T* closest_x, T* closest_z,
    int* segment_indices,
    T* projection_scalars)
{
    const auto want_projection = closest_x != nullptr && closest_z != nullptr;
    const auto want_distance = distances != nullptr;
    const auto want_index = segment_indices != nullptr;
    const auto want_param = projection_scalars != nullptr;

    if (n_segments == 0) {
        return;
    }
    constexpr T AB_NORM_THRESHOLD = T{1e-9};

    T stack_workspace[MaxStackSegments * 3];
    std::unique_ptr<T[]> heap_workspace;

    T* abx;
    T* abz;
    T* inv_ab_norm_sq;

    if (n_segments <= MaxStackSegments) {
        abx = stack_workspace;
        abz = stack_workspace + MaxStackSegments;
        inv_ab_norm_sq = stack_workspace + 2 * MaxStackSegments;
    } else {
        heap_workspace = std::make_unique<T[]>(n_segments * 3);
        abx = heap_workspace.get();
        abz = abx + n_segments;
        inv_ab_norm_sq = abz + n_segments;
    }

    for (size_t i = 0; i < n_segments; ++i) {
        const T ax = segments_data[i * 4 + 0];
        const T ay = segments_data[i * 4 + 1];
        const T bx = segments_data[i * 4 + 2];
        const T by = segments_data[i * 4 + 3];

        abx[i] = bx - ax;
        abz[i] = by - ay;
        const T ab_norm_sq = std::fma(abx[i], abx[i], abz[i] * abz[i]);
        inv_ab_norm_sq[i] = (ab_norm_sq > AB_NORM_THRESHOLD) ? T{1} / ab_norm_sq : T{0};
    }

    size_t pt_idx = 0;

    if constexpr (UseAVX2 && std::is_same_v<T, float>) {
        constexpr size_t vec_size = 8;
        const size_t n_vec = n_pts / vec_size;

        for (size_t vec_i = 0; vec_i < n_vec; ++vec_i) {
            const size_t base_idx = vec_i * vec_size;

            const __m256 px = _mm256_loadu_ps(&pts_x[base_idx]);
            const __m256 pz = _mm256_loadu_ps(&pts_z[base_idx]);

            __m256 min_dist_sq = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            __m256 min_closest_x = _mm256_setzero_ps();
            __m256 min_closest_z = _mm256_setzero_ps();
            __m256 min_proj_scalar = _mm256_setzero_ps();
            __m256i min_seg_idx = _mm256_set1_epi32(-1);

            for (size_t seg_idx = 0; seg_idx < n_segments; ++seg_idx) {
                const float ax = segments_data[seg_idx * 4 + 0];
                const float az = segments_data[seg_idx * 4 + 1];
                const float seg_abx = abx[seg_idx];
                const float seg_abz = abz[seg_idx];
                const float seg_inv_norm = inv_ab_norm_sq[seg_idx];

                if (seg_inv_norm == 0.0f) continue;

                const __m256 ax_vec = _mm256_set1_ps(ax);
                const __m256 az_vec = _mm256_set1_ps(az);
                const __m256 abx_vec = _mm256_set1_ps(seg_abx);
                const __m256 abz_vec = _mm256_set1_ps(seg_abz);
                const __m256 inv_norm_vec = _mm256_set1_ps(seg_inv_norm);

                const __m256 apx = _mm256_sub_ps(px, ax_vec);
                const __m256 apz = _mm256_sub_ps(pz, az_vec);

                __m256 proj_scalar = _mm256_fmadd_ps(apx, abx_vec, _mm256_mul_ps(apz, abz_vec));
                proj_scalar = _mm256_mul_ps(proj_scalar, inv_norm_vec);

                const __m256 zero = _mm256_setzero_ps();
                const __m256 one = _mm256_set1_ps(1.0f);
                proj_scalar = _mm256_max_ps(zero, _mm256_min_ps(one, proj_scalar));

                const __m256 closest_x = _mm256_fmadd_ps(proj_scalar, abx_vec, ax_vec);
                const __m256 closest_z = _mm256_fmadd_ps(proj_scalar, abz_vec, az_vec);

                const __m256 dx = _mm256_sub_ps(px, closest_x);
                const __m256 dz = _mm256_sub_ps(pz, closest_z);
                const __m256 dist_sq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dz, dz));

                const __m256 is_smaller = _mm256_cmp_ps(dist_sq, min_dist_sq, _CMP_LT_OQ);
                min_dist_sq = _mm256_blendv_ps(min_dist_sq, dist_sq, is_smaller);

                if (want_projection) {
                    min_closest_x = _mm256_blendv_ps(min_closest_x, closest_x, is_smaller);
                    min_closest_z = _mm256_blendv_ps(min_closest_z, closest_z, is_smaller);
                }

                if (want_param) {
                    min_proj_scalar = _mm256_blendv_ps(min_proj_scalar, proj_scalar, is_smaller);
                }

                if (want_index) {
                    const __m256i seg_idx_vec = _mm256_set1_epi32(static_cast<int>(seg_idx));
                    const __m256i mask = _mm256_castps_si256(is_smaller);
                    min_seg_idx = _mm256_blendv_epi8(min_seg_idx, seg_idx_vec, mask);
                }
            }

            if (want_distance) {
                _mm256_storeu_ps(&distances[base_idx], _mm256_sqrt_ps(min_dist_sq));
            }

            if (want_projection) {
                _mm256_storeu_ps(&closest_x[base_idx], min_closest_x);
                _mm256_storeu_ps(&closest_z[base_idx], min_closest_z);
            }

            if (want_index) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&segment_indices[base_idx]), min_seg_idx);
            }

            if (want_param) {
                _mm256_storeu_ps(&projection_scalars[base_idx], min_proj_scalar);
            }
        }

        pt_idx = n_vec * vec_size;
    }
    else if constexpr (UseAVX2 && std::is_same_v<T, double>) {
        constexpr size_t vec_size = 4;
        const size_t n_vec = n_pts / vec_size;

        for (size_t vec_i = 0; vec_i < n_vec; ++vec_i) {
            const size_t base_idx = vec_i * vec_size;

            const __m256d px = _mm256_loadu_pd(&pts_x[base_idx]);
            const __m256d pz = _mm256_loadu_pd(&pts_z[base_idx]);

            __m256d min_dist_sq = _mm256_set1_pd(std::numeric_limits<double>::infinity());
            __m256d min_closest_x = _mm256_setzero_pd();
            __m256d min_closest_z = _mm256_setzero_pd();
            __m256d min_proj_scalar = _mm256_setzero_pd();
            __m128i min_seg_idx = _mm_set1_epi32(-1);

            for (size_t seg_idx = 0; seg_idx < n_segments; ++seg_idx) {
                const double ax = segments_data[seg_idx * 4 + 0];
                const double az = segments_data[seg_idx * 4 + 1];
                const double seg_abx = abx[seg_idx];
                const double seg_abz = abz[seg_idx];
                const double seg_inv_norm = inv_ab_norm_sq[seg_idx];

                if (seg_inv_norm == 0.0) continue;

                const __m256d ax_vec = _mm256_set1_pd(ax);
                const __m256d az_vec = _mm256_set1_pd(az);
                const __m256d abx_vec = _mm256_set1_pd(seg_abx);
                const __m256d abz_vec = _mm256_set1_pd(seg_abz);
                const __m256d inv_norm_vec = _mm256_set1_pd(seg_inv_norm);

                const __m256d apx = _mm256_sub_pd(px, ax_vec);
                const __m256d apz = _mm256_sub_pd(pz, az_vec);

                __m256d proj_scalar = _mm256_fmadd_pd(apx, abx_vec, _mm256_mul_pd(apz, abz_vec));
                proj_scalar = _mm256_mul_pd(proj_scalar, inv_norm_vec);

                const __m256d zero = _mm256_setzero_pd();
                const __m256d one = _mm256_set1_pd(1.0);
                proj_scalar = _mm256_max_pd(zero, _mm256_min_pd(one, proj_scalar));

                const __m256d closest_x = _mm256_fmadd_pd(proj_scalar, abx_vec, ax_vec);
                const __m256d closest_z = _mm256_fmadd_pd(proj_scalar, abz_vec, az_vec);

                const __m256d dx = _mm256_sub_pd(px, closest_x);
                const __m256d dz = _mm256_sub_pd(pz, closest_z);
                const __m256d dist_sq = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dz, dz));

                const __m256d is_smaller = _mm256_cmp_pd(dist_sq, min_dist_sq, _CMP_LT_OQ);
                min_dist_sq = _mm256_blendv_pd(min_dist_sq, dist_sq, is_smaller);

                if (want_projection) {
                    min_closest_x = _mm256_blendv_pd(min_closest_x, closest_x, is_smaller);
                    min_closest_z = _mm256_blendv_pd(min_closest_z, closest_z, is_smaller);
                }

                if (want_param) {
                    min_proj_scalar = _mm256_blendv_pd(min_proj_scalar, proj_scalar, is_smaller);
                }

                if (want_index) {
                    const __m128i seg_idx_vec = _mm_set1_epi32(static_cast<int>(seg_idx));
                    const __m256i is_smaller_int = _mm256_castpd_si256(is_smaller);
                    const __m256i permute_indices = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
                    const __m256i permuted_mask = _mm256_permutevar8x32_epi32(is_smaller_int, permute_indices);
                    const __m128i mask_128 = _mm256_castsi256_si128(permuted_mask);
                    min_seg_idx = _mm_blendv_epi8(min_seg_idx, seg_idx_vec, mask_128);
                }
            }

            if (want_distance) {
                _mm256_storeu_pd(&distances[base_idx], _mm256_sqrt_pd(min_dist_sq));
            }

            if (want_projection) {
                _mm256_storeu_pd(&closest_x[base_idx], min_closest_x);
                _mm256_storeu_pd(&closest_z[base_idx], min_closest_z);
            }

            if (want_index) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&segment_indices[base_idx]), min_seg_idx);
            }

            if (want_param) {
                _mm256_storeu_pd(&projection_scalars[base_idx], min_proj_scalar);
            }
        }

        pt_idx = n_vec * vec_size;
    }

    for (; pt_idx < n_pts; ++pt_idx) {
        const T px = pts_x[pt_idx];
        const T pz = pts_z[pt_idx];

        T min_dist_sq = std::numeric_limits<T>::max();
        T min_closest_x = T{0};
        T min_closest_z = T{0};
        T min_proj_scalar = T{0};
        int min_seg_idx = -1;

        for (size_t seg_idx = 0; seg_idx < n_segments; ++seg_idx) {
            const T ax = segments_data[seg_idx * 4 + 0];
            const T az = segments_data[seg_idx * 4 + 1];
            const T seg_abx = abx[seg_idx];
            const T seg_abz = abz[seg_idx];
            const T seg_inv_norm = inv_ab_norm_sq[seg_idx];

            if (seg_inv_norm == T{0}) continue;

            const T apx = px - ax;
            const T apz = pz - az;
            const T dot_prod = std::fma(apx, seg_abx, apz * seg_abz);
            T proj_scalar = dot_prod * seg_inv_norm;
            proj_scalar = std::clamp(proj_scalar, T{0}, T{1});

            const T closest_x = std::fma(proj_scalar, seg_abx, ax);
            const T closest_z = std::fma(proj_scalar, seg_abz, az);

            const T dx = px - closest_x;
            const T dz = pz - closest_z;
            const T dist_sq = std::fma(dx, dx, dz * dz);

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                min_closest_x = closest_x;
                min_closest_z = closest_z;
                min_seg_idx = static_cast<int>(seg_idx);
                min_proj_scalar = proj_scalar;
            }
        }

        if (want_distance) {
            distances[pt_idx] = std::sqrt(min_dist_sq);
        }
        if (want_projection) {
            closest_x[pt_idx] = min_closest_x;
            closest_z[pt_idx] = min_closest_z;
        }
        if (want_index) {
            segment_indices[pt_idx] = min_seg_idx;
        }
        if (want_param) {
            projection_scalars[pt_idx] = min_proj_scalar;
        }
    }
}
