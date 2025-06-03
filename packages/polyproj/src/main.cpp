#include <algorithm>
#include <cmath>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Project points onto a polyline (segments) and return a dict of results.
// points: shape (N,2), segments: shape (M,2,2)
py::dict
project_on_polyline(py::array_t<double> points, py::array_t<double> segments,
                    bool return_distance = true, bool return_projection = false,
                    bool return_param = false, bool return_index = false) {
  // Validate and unpack points
  auto buf_pts = points.request();
  if (buf_pts.ndim != 2 || buf_pts.shape[1] != 2)
    throw std::runtime_error("`points` must be a 2D array with shape (N,2)");
  py::ssize_t N = buf_pts.shape[0];
  double *pts_ptr = static_cast<double *>(buf_pts.ptr);

  // Validate and unpack segments
  auto buf_segs = segments.request();
  if (buf_segs.ndim != 3 || buf_segs.shape[1] != 2 || buf_segs.shape[2] != 2)
    throw std::runtime_error(
        "`segments` must be a 3D array with shape (M,2,2)");
  py::ssize_t M = buf_segs.shape[0];
  double *seg_ptr = static_cast<double *>(buf_segs.ptr);

  // Prepare storage for best results per point
  std::vector<double> best_dist_sq(N, std::numeric_limits<double>::infinity());
  std::vector<double> best_proj;
  std::vector<double> best_params;
  std::vector<py::ssize_t> best_indices;

  if (return_projection)
    best_proj.resize(N * 2);
  if (return_param)
    best_params.resize(N);
  if (return_index)
    best_indices.resize(N);

  // Brute-force over segments then points
  for (size_t seg_idx = 0; seg_idx < M; ++seg_idx) {
    // Flattened layout: seg_ptr[seg_idx*4 + (2*i + j)]
    const double ax = seg_ptr[seg_idx * 4 + 0];
    const double ay = seg_ptr[seg_idx * 4 + 1];
    const double bx = seg_ptr[seg_idx * 4 + 2];
    const double by = seg_ptr[seg_idx * 4 + 3];

    const double vx = bx - ax;
    const double vy = by - ay;
    const double v_norm_sq = vx * vx + vy * vy;
    const bool is_point = (v_norm_sq == 0.0);

    for (size_t i = 0; i < N; ++i) {
      const double px = pts_ptr[i * 2 + 0];
      const double py = pts_ptr[i * 2 + 1];
      double t, cx, cy;

      if (is_point) {
        t = 0.0;
        cx = ax;
        cy = ay;
      } else {
        const double apx = px - ax;
        const double apy = py - ay;
        double raw = (apx * vx + apy * vy) / v_norm_sq;
        // clamp
        if (raw < 0.0)
          raw = 0.0;
        else if (raw > 1.0)
          raw = 1.0;
        t = raw;
        cx = ax + t * vx;
        cy = ay + t * vy;
      }

      const double dx = px - cx;
      const double dy = py - cy;
      const double dist_sq = dx * dx + dy * dy;

      if (dist_sq < best_dist_sq[i]) {
        best_dist_sq[i] = dist_sq;
        if (return_projection) {
          best_proj[i * 2 + 0] = cx;
          best_proj[i * 2 + 1] = cy;
        }
        if (return_param) {
          best_params[i] = t;
        }
        if (return_index) {
          best_indices[i] = static_cast<int>(seg_idx);
        }
      }
    }
  }

  // Build result dict
  py::dict result;

  // distances
  if (return_distance) {
    py::array_t<double> arr({N});
    auto buf = arr.request();
    double *dst = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < N; ++i)
      dst[i] = std::sqrt(best_dist_sq[i]);
    result["distances"] = arr;
  } else {
    result["distances"] = py::none();
  }

  // projections
  if (return_projection) {
    std::vector<py::ssize_t> shape = {N , 2};
    py::array_t<double> arr(shape);
    auto buf = arr.request();
    double *dst = static_cast<double *>(buf.ptr);
    std::copy(best_proj.begin(), best_proj.end(), dst);
    result["projections"] = arr;
  } else {
    result["projections"] = py::none();
  }

  // params
  if (return_param) {
    py::array_t<double> arr(N);
    auto buf = arr.request();
    double *dst = static_cast<double *>(buf.ptr);
    std::copy(best_params.begin(), best_params.end(), dst);
    result["params"] = arr;
  } else {
    result["params"] = py::none();
  }

  // indices
  if (return_index) {
    py::array_t<py::ssize_t> arr(N);
    auto buf = arr.request();
    py::ssize_t *dst = static_cast<py::ssize_t *>(buf.ptr);
    std::copy(best_indices.begin(), best_indices.end(), dst);
    result["indices"] = arr;
  } else {
    result["indices"] = py::none();
  }

  return result;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "pybind11 bindings for polyproj core functions";
  m.def("project_on_polyline", &project_on_polyline, py::arg("points"),
        py::arg("segments"), py::arg("return_distance") = true,
        py::arg("return_projection") = false, py::arg("return_param") = false,
        py::arg("return_index") = false,
        R"pbdoc(
Project points onto a polyline.

Arguments:
  points            -- array of shape (N,2)
  segments          -- array of shape (M,2,2)
  return_distance   -- include distances array
  return_projection -- include projected points array
  return_param      -- include param t array
  return_index      -- include segment index array

Returns:
  dict with keys "distances", "projections", "params", "indices"
  (each is an array if requested, otherwise None)
)pbdoc");
}
