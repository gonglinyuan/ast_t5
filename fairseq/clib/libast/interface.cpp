//
// Created by Anonymized on 5/1/23.
//

#include <torch/extension.h>

#include "libast.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void
configure_interface(long tok_l, long tok_r, long tok_0, long tok_py_func_def, long tok_py_block, long tok_py_expr_stmt,
                    long tok_py_string, long tok_py_func_body, std::vector<long> identifier_ndtypes,
                    long sentinel_idx_begin, long max_positions, long ndtype_begin) {
    configured = true;
    TOK_L = tok_l;
    TOK_R = tok_r;
    TOK_0 = tok_0;
    TOK_PY_FUNC_DEF = tok_py_func_def;
    TOK_PY_BLOCK = tok_py_block;
    TOK_PY_EXPR_STMT = tok_py_expr_stmt;
    TOK_PY_STRING = tok_py_string;
    TOK_PY_FUNC_BODY = tok_py_func_body;
    IDENTIFIER_NDTYPES = identifier_ndtypes;
    SENTINEL_IDX_BEGIN = sentinel_idx_begin;
    MAX_POSITIONS = max_positions;
    NDTYPE_BEGIN = ndtype_begin;
}

std::tuple<py::array_t<unsigned char>, std::vector<long>, std::vector<long>, std::vector<long>, py::array_t<long>, py::array_t<long>>
generate_masks_interface(py::array_t<long> seq_arr, py::array_t<long> ndtype_arr, double mask_prob,
                         long flat_multiple_len, long threshold_lb, long threshold_ub, double obf_prob,
                         double obf_ratio, double t2c_prob, double t2c_ratio, long seed) {
    // Initialize input buffer
    py::buffer_info seq_info = seq_arr.request(), ndtype_info = ndtype_arr.request();
    if (seq_info.ndim != 1 || seq_info.itemsize != sizeof(long)) {
        throw std::runtime_error("Incompatible seq buffer format!");
    }
    if (ndtype_info.ndim != 1 || ndtype_info.itemsize != sizeof(long)) {
        throw std::runtime_error("Incompatible ndtype buffer format!");
    }
    std::vector<long> seq(static_cast<long *>(seq_info.ptr), static_cast<long *>(seq_info.ptr) + seq_info.size);
    std::vector<long> binarized_ndtypes(static_cast<long *>(ndtype_info.ptr),
                                        static_cast<long *>(ndtype_info.ptr) + ndtype_info.size);

    std::vector<unsigned char> mask;
    std::vector<long> mask_init_idx, sentinels, span_ndtypes;
    std::tie(mask, mask_init_idx, sentinels, span_ndtypes) = generate_masks(seq, binarized_ndtypes, mask_prob,
                                                                            flat_multiple_len, threshold_lb,
                                                                            threshold_ub, obf_prob, obf_ratio, t2c_prob,
                                                                            t2c_ratio, seed);
    std::vector<long> source = generate_source_seq(seq, mask, mask_init_idx, sentinels, span_ndtypes);
    std::vector<long> target = generate_target_seq(seq, mask, mask_init_idx, sentinels);

    // Initialize output buffer
    auto mask_arr = py::array_t<unsigned char>(static_cast<long>(mask.size()));
    py::buffer_info mask_info = mask_arr.request();
    std::copy(mask.begin(), mask.end(), static_cast<unsigned char *>(mask_info.ptr));
    auto source_arr = py::array_t<long>(static_cast<long>(source.size()));
    py::buffer_info source_info = source_arr.request();
    std::copy(source.begin(), source.end(), static_cast<long *>(source_info.ptr));
    auto target_arr = py::array_t<long>(static_cast<long>(target.size()));
    py::buffer_info target_info = target_arr.request();
    std::copy(target.begin(), target.end(), static_cast<long *>(target_info.ptr));

    return std::make_tuple(mask_arr, mask_init_idx, sentinels, span_ndtypes, source_arr, target_arr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("configured") = configured;
    m.def("configure", &configure_interface);
    m.def("generate_masks", &generate_masks_interface);
}