//
// Created by Anonymized on 5/1/23.
//

#ifndef UNTITLED_LIBAST_HPP
#define UNTITLED_LIBAST_HPP

#include <vector>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <utility>
#include <random>
#include <iterator>
#include <memory>
#include <chrono>
#include <limits>
#include <unordered_map>

bool configured = false;
long TOK_L = -1;
long TOK_R = -1;
long TOK_0 = -1;
long TOK_PY_FUNC_DEF = -1;
long TOK_PY_BLOCK = -1;
long TOK_PY_EXPR_STMT = -1;
long TOK_PY_STRING = -1;
long TOK_PY_FUNC_BODY = -1;
long FLAT_MULTIPLE_LEN = -1;
long THRESHOLD_LB = -1;
long THRESHOLD_UB = -1;
double OBF_PROB = 0.0;
double OBF_RATIO = 0.0;
double T2C_PROB = 0.0;
double T2C_RATIO = 0.0;
long SENTINEL_IDX_BEGIN = -1;
long MAX_POSITIONS = -1;
long NDTYPE_BEGIN = -1;

std::vector<long> IDENTIFIER_NDTYPES;


std::mt19937 rng;


class Tree {
public:
    long l;
    long r;
    long ndtype;
    std::vector<std::shared_ptr<Tree>> child;
    bool is_leaf;
    bool mask;
    long mask_sum;

    explicit Tree(long l, long ndtype, bool leaf) : l(l), ndtype(ndtype), is_leaf(leaf), mask(false), mask_sum(0) {
        if (leaf) {
            r = l + 1;
        } else {
            r = l;
        }
    }

    long get_size() const {
        return r - l;
    }

    void add_child(const std::shared_ptr<Tree> &child_) {
        assert(r == child_->l);
        child.push_back(child_);
        r = child_->r;
    }

    void fill(long r_) {
        assert(r <= r_);
        for (long i = r; i < r_; ++i) {
            add_child(std::make_shared<Tree>(i, -1, true));
        }
    }

    void add_child_maybe_fill(const std::shared_ptr<Tree> &child_) {
        fill(child_->l);
        add_child(child_);
    }

    void do_mask() {
        mask = true;
        mask_sum = get_size();
    }

    void update_mask_sum() {
        mask_sum = 0;
        for (const auto &ch: child) {
            mask_sum += ch->mask_sum;
        }
    }
};


std::string tree_to_string(const std::shared_ptr<Tree> &node, long indent = 0, bool include_mask = false) {
    std::stringstream result;
    std::string inner;

    if (!node->is_leaf) {
        std::stringstream inner_ss;
        inner_ss << "[\n";
        for (const auto &child: node->child) {
            inner_ss << tree_to_string(child, indent + 1, include_mask) << "\n";
        }
        inner_ss << std::string(indent * 2, ' ') << "]";
        inner = inner_ss.str();
    } else {
        inner = "None";
    }

    if (!include_mask) {
        result << std::string(indent * 2, ' ') << "Tree(" << node->l << ", " << node->r << ", " << node->ndtype << ", "
               << inner << ")";
    } else {
        result << std::string(indent * 2, ' ') << "Tree(" << node->l << ", " << node->r << ", " << node->ndtype << ", "
               << node->mask << ", " << node->mask_sum << ", " << inner << ")";
    }
    return result.str();
}


std::shared_ptr<Tree> build_tree(const std::vector<long> &binarized_ndtypes, long real_len) {
    assert(TOK_0 != -1 && TOK_L != -1 && TOK_R != -1);

    auto root = std::make_shared<Tree>(0, -1, false);
    std::vector<std::shared_ptr<Tree>> stack = {root};
    size_t n = binarized_ndtypes.size();

    long pos;
    for (size_t i = 0; i < n - 1; i += 3) {
        pos = binarized_ndtypes[i] - TOK_0;
        bool is_in;

        if (binarized_ndtypes[i + 1] == TOK_L) {
            is_in = true;
        } else if (binarized_ndtypes[i + 1] == TOK_R) {
            is_in = false;
        } else {
            std::cout << "TOK_L: " << TOK_L << "; TOK_R: " << TOK_R << "; n: " << n << std::endl;
            for (size_t j = 0; j < n - 1; ++j) {
                std::cout << binarized_ndtypes[j] << " ";
            }
            std::cout << std::endl;
            throw std::runtime_error("Unrecognized ndtype token " + std::to_string(binarized_ndtypes[i + 1]));
        }

        if (is_in) {
            auto cur_node = std::make_shared<Tree>(pos, binarized_ndtypes[i + 2], false);
            stack.push_back(cur_node);
        } else {
            if (stack.size() <= 1) {
                root->fill(pos);
                continue;
            }
            auto cur_node = stack.back();

            if (cur_node->ndtype == TOK_PY_FUNC_BODY) {
                // end of a function body, exit twice
                stack.pop_back();
                cur_node->fill(pos);
                stack.back()->add_child_maybe_fill(cur_node);
                cur_node = stack.back();
            }

            stack.pop_back();
            cur_node->fill(pos);
            stack.back()->add_child_maybe_fill(cur_node);

            if (T2C_RATIO > 0.0 && cur_node->ndtype == TOK_PY_EXPR_STMT && cur_node->child.size() == 1 &&
                cur_node->child[0]->ndtype == TOK_PY_STRING && stack.back()->ndtype == TOK_PY_BLOCK &&
                stack.size() >= 2 && stack[stack.size() - 2]->ndtype == TOK_PY_FUNC_DEF) {
                // is a docstring: function_def -> block -> [expression_statement] -> string
                auto func_body = std::make_shared<Tree>(pos, TOK_PY_FUNC_BODY, false);
                stack.push_back(func_body);
            }
        }
    }

    while (stack.size() > 1) {
        auto cur_node = stack.back();
        stack.pop_back();
        cur_node->fill(pos);
        if (cur_node->get_size() > 0) {
            stack.back()->add_child_maybe_fill(cur_node);
        }
    }

    root->fill(real_len);
    return root;
}

double InvJm(int n, int x, int N, int m) {
    return (1.0 - double(x) / (double(m) + 1.0)) /
           (1.0 - (double(n) - 1.0 - double(x)) / (double(N) - 1.0 - double(m)));
}

double hyperquick(int n, int x, int N, int M) {
    if (x < 0 || x < n + M - N) {
        return 0.0;
    }
    if (x > n || x > M) {
        return 1.0;
    }
    const double TOL = 1e-12;
    double s = 1.0;
    for (int k = x; k <= M - 2; ++k) {
        s = s * InvJm(n, x, N, k) + 1.0;
    }
    double ak = s;
    double bk = s;
    int k = M - 2;
    double epsk = 2 * TOL;
    while ((k < (N - (n - x) - 1)) && epsk > TOL) {
        double ck = ak / bk;
        k = k + 1;
        double jjm = InvJm(n, x, N, k);
        bk = bk * jjm + 1.0;
        ak = ak * jjm;
        epsk = (N - (n - x) - 1 - k) * (ck - ak / bk);
    }
    return 1 - (ak / bk - epsk / 2);
}

std::discrete_distribution<long> build_hypergeometric(long n_total, long n_good, long n_sample) {
    std::vector<double> p;
    p.push_back(hyperquick((int) n_sample, 0, (int) n_total, (int) n_good));
    for (long i = 1; i <= std::min(n_sample, n_good); ++i) {
        p.push_back(hyperquick((int) n_sample, (int) i, (int) n_total, (int) n_good));
    }
    for (long i = std::min(n_sample, n_good); i >= 1; --i) {
        p[i] = p[i] - p[i - 1];
    }
    return {p.begin(), p.end()};
}

std::vector<long> generate_multivariate_hypergeometric(std::vector<long> colors, long n_sample) {
    std::vector<long> result = std::vector<long>(colors.size(), 0l);
    long total = 0;
    for (long color: colors) {
        total += color;
    }
    if ((total == 0) || (n_sample == 0)) {
        return result;
    }

    bool more_than_half = n_sample > (total / 2);
    if (more_than_half) {
        n_sample = total - n_sample;
    }

    long num_to_sample = n_sample;
    long remaining = total;
    for (long j = 0; (num_to_sample > 0) && (j + 1 < colors.size()); ++j) {
        auto dist = build_hypergeometric(remaining, colors[j], num_to_sample);
        long r = dist(rng);
        result[j] = r;
        num_to_sample -= r;
        remaining -= colors[j];
    }
    if (num_to_sample > 0) {
        result[colors.size() - 1] = num_to_sample;
    }

    if (more_than_half) {
        for (size_t k = 0; k < colors.size(); ++k) {
            result[k] = colors[k] - result[k];
        }
    }
    return result;
}

std::vector<long> weighted_shuffle(std::vector<double> weights) {
    std::vector<long> results;
    for (long i = 0; i < weights.size(); ++i) {
        auto dist = std::discrete_distribution<long>(weights.begin(), weights.end());
        long child_i = dist(rng);
        results.push_back(child_i);
        weights[child_i] = 0.0;
    }
    return results;
}

void locality_mask(long n, long num_masked_tokens, std::vector<bool> &out) {
    assert(num_masked_tokens <= n);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    std::vector<long> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    for (long i = 0; i < n && num_masked_tokens; ++i) {
        if (i + 1 < n) {
            std::uniform_int_distribution<long> dist_i_n(i, n - 1);
            std::swap(indices[i], indices[dist_i_n(rng)]);
        }
        std::uniform_int_distribution<long> dist_len(1, FLAT_MULTIPLE_LEN - 1);
        long start = indices[i], span_len = dist_len(rng);
        for (long j = 0; j < span_len && num_masked_tokens; ++j) {
            if (!out[(start + j) % n]) {
                out[(start + j) % n] = true;
                num_masked_tokens--;
            }
        }
    }
    assert(num_masked_tokens == 0);
}


std::pair<double, double> mean_std_fn(double x) {
    double f1 = std::max(x - 0.5798248, 0.0), f2 = std::max(x - 0.83561796, 0.0), f3 = std::max(x - 0.90088475, 0.0);
    double mean = (-0.28175632 + 2.39978213 * x + (-0.32083743) * f1 + (-0.31381148) * f2 + (-0.31374825) * f3);
    double std_dev = std::exp(
            -2.41134106 + (-0.30088176) * x + (-0.29200343) * f1 + (-0.27931809) * f2 + (-0.28136072) * f3);
    return std::make_pair(mean, std_dev);
}

std::vector<double> calculate_shuffle_weights(std::vector<long> child_sizes, long num_masked_tokens) {
    long child_size_mi = std::numeric_limits<long>::max(), child_size_mx = std::numeric_limits<long>::min(), child_size_su = 0, rank_of_m = 0;
    size_t largest_child = -1, smallest_child = -1;
    for (size_t i = 0; i < child_sizes.size(); i++) {
        if (child_sizes[i] < child_size_mi) {
            child_size_mi = child_sizes[i];
            smallest_child = i;
        }
        if (child_sizes[i] > child_size_mx) {
            child_size_mx = child_sizes[i];
            largest_child = i;
        }
        child_size_su += child_sizes[i];
        rank_of_m += static_cast<long>(child_sizes[i] >= num_masked_tokens);
    }

    std::vector<double> weights(child_sizes.size());
    if (num_masked_tokens <= child_size_mi) {
        for (size_t i = 0; i < child_sizes.size(); i++) {
            weights[i] = static_cast<double>(child_sizes[i]);
        }
    } else if (num_masked_tokens >= child_size_su - child_size_mx && num_masked_tokens <= child_size_mx) {
        for (size_t i = 0; i < child_sizes.size(); i++) {
            weights[i] = 1.0;
        }
        weights[largest_child] = static_cast<double>(child_size_su) / static_cast<double>(num_masked_tokens) - 1.0;
    } else if (child_sizes.size() == 2) {
        assert(num_masked_tokens >= child_size_mx);
        for (size_t i = 0; i < child_sizes.size(); i++) {
            weights[i] = 1.0;
        }
        weights[largest_child] =
                static_cast<double>(child_size_su - child_size_mx) / static_cast<double>(child_size_mx);
    } else if (child_sizes.size() == 3 && rank_of_m == 2) {
        for (size_t i = 0; i < child_sizes.size(); i++) {
            weights[i] = (static_cast<double>(child_size_su) / static_cast<double>(num_masked_tokens) - 1.0) *
                         static_cast<double>(child_sizes[i]) / static_cast<double>(child_size_su - child_size_mi);
        }
        weights[smallest_child] = 1.0;
    } else {
        for (size_t i = 0; i < child_sizes.size(); i++) {
            double x = static_cast<double>(child_sizes[i]) / static_cast<double>(child_size_su);
            double x_mean, x_std;
            std::tie(x_mean, x_std) = mean_std_fn(x);
            weights[i] = std::exp(std::normal_distribution<>(x_mean, x_std)(rng)) + 0.001;
        }
    }

    double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto &w: weights) {
        w /= weight_sum;
    }
    return weights;
}

void generate_mask(const std::shared_ptr<Tree> &node, long num_masked_tokens) {
    long remaining_size = node->get_size() - node->mask_sum;
    assert(1 <= num_masked_tokens && num_masked_tokens <= remaining_size);

    if (node->is_leaf) { // is a leaf node
        assert(remaining_size == num_masked_tokens && num_masked_tokens == 1);
        node->do_mask();
        return;
    }

    assert(!node->child.empty());
    std::vector<long> child_sizes(node->child.size());
    bool is_flat = true;
    for (size_t i = 0; i < node->child.size(); i++) {
        child_sizes[i] = node->child[i]->get_size() - node->child[i]->mask_sum;
        if (node->child[i]->get_size() > 1) {
            is_flat = false;
        }
    }

    if (is_flat) {
        std::vector<size_t> nonzero_child_indices;
        for (size_t i = 0; i < node->child.size(); i++) {
            if (child_sizes[i] > 0) {
                nonzero_child_indices.push_back(i);
            }
        }
        assert(nonzero_child_indices.size() == remaining_size);
        std::vector<bool> is_mask(remaining_size, false);
        locality_mask(remaining_size, num_masked_tokens, is_mask);
        for (size_t jj = 0; jj < is_mask.size(); ++jj) {
            size_t j = nonzero_child_indices[jj];
            assert(node->l + j == node->child[j]->l);
            if (is_mask[jj]) {
                node->child[j]->do_mask();
            }
        }
        node->update_mask_sum();
        return;
    }

    std::vector<long> child_to_mask = generate_multivariate_hypergeometric(child_sizes, num_masked_tokens);

    std::uniform_int_distribution<long> dist_uniform(THRESHOLD_LB, THRESHOLD_UB);
    long threshold = dist_uniform(rng);
    long num_masked_tokens_small = 0;
    std::vector<long> small_child_indices, small_child_original_sizes;
    for (size_t i = 0; i < node->child.size(); i++) {
        if (child_sizes[i] == 0) {
            continue;
        }
        if (child_to_mask[i] <= threshold) {
            num_masked_tokens_small += child_to_mask[i];
            small_child_indices.push_back(static_cast<long>(i));
            small_child_original_sizes.push_back(node->child[i]->get_size());
        } else {
            generate_mask(node->child[i], child_to_mask[i]);
        }
    }

    auto weights = calculate_shuffle_weights(small_child_original_sizes, num_masked_tokens_small);
    auto sorted_small_child_indices = weighted_shuffle(weights);

    for (long ii: sorted_small_child_indices) {
        long i = small_child_indices[ii];
        assert(child_to_mask[i] <= threshold &&
               small_child_original_sizes[ii] == child_sizes[i] + node->child[i]->mask_sum);
        if (num_masked_tokens_small >= child_sizes[i]) {
            node->child[i]->do_mask();
            num_masked_tokens_small -= child_sizes[i];
        } else if (num_masked_tokens_small > 0) {
            generate_mask(node->child[i], num_masked_tokens_small);
            num_masked_tokens_small = 0;
        } else {
            break;
        }
    }
    node->update_mask_sum();
}

std::vector<std::tuple<long, long, long>> collect_mask_raw(const std::shared_ptr<Tree> &node) {
    std::vector<std::tuple<long, long, long>> result;
    if (node->mask) {
        result.emplace_back(node->l, node->r, node->ndtype);
        return result;
    }
    if (node->is_leaf) {
        return result;
    }
    for (const auto &child: node->child) {
        auto child_result = collect_mask_raw(child);
        result.insert(result.end(), std::make_move_iterator(child_result.begin()),
                      std::make_move_iterator(child_result.end()));
    }
    return result;
}


void collect_mask_flat(const std::shared_ptr<Tree> &node, unsigned char *out) {
    if (node->mask) {
        for (long i = node->l; i < node->r; ++i) {
            out[i] = 1;
        }
    }
    if (node->is_leaf) {
        return;
    }
    for (const auto &child: node->child) {
        collect_mask_flat(child, out);
    }
}

std::vector<std::tuple<long, long, long>>
merge_adjacent_spans(const std::vector<std::tuple<long, long, long>> &mask_spans) {
    std::vector<std::tuple<long, long, long>> tmp;
    for (const auto &span: mask_spans) {
        long l, r, ndtype;
        std::tie(l, r, ndtype) = span;

        if (!tmp.empty()) {
            long last_l, last_r, last_ndtype;
            std::tie(last_l, last_r, last_ndtype) = tmp.back();
            if (last_r == l) {
                tmp.back() = std::make_tuple(last_l, r, -1);
                continue;
            }
        }
        tmp.push_back(span);
    }

    std::vector<std::tuple<long, long, long>> result;
    for (const auto &span: tmp) {
        long l, r, ndtype;
        std::tie(l, r, ndtype) = span;
        result.emplace_back(l, r, ndtype);
    }

    return result;
}

std::vector<std::tuple<long, long, long>> collect_mask(const std::shared_ptr<Tree> &node) {
    return merge_adjacent_spans(collect_mask_raw(node));
}

unsigned long mod_pow(unsigned long a, unsigned long b, unsigned long p) {
    unsigned long res = 1;
    a %= p;

    while (b > 0) {
        if (b & 1) {
            res = (res * a) % p;
        }
        a = (a * a) % p;
        b >>= 1;
    }
    return res;
}

unsigned long mod_inverse(unsigned long a, unsigned long p) {
    return mod_pow(a, p - 2, p);
}

class SeqHasher {
private:
    const unsigned long seed_1 = 1997, seed_2 = 1993, mo_1 = 1000000007, mo_2 = 1000000009;
    unsigned long n;
    std::vector<unsigned long> p_1, p_inv_1, h_sum_1;
    std::vector<unsigned long> p_2, p_inv_2, h_sum_2;
public:
    explicit SeqHasher(const std::vector<long> &seq) : n(seq.size()) {
        p_1.resize(n + 1);
        p_inv_1.resize(n + 1);
        h_sum_1.resize(n + 1);
        p_2.resize(n + 1);
        p_inv_2.resize(n + 1);
        h_sum_2.resize(n + 1);
        p_1[0] = 1;
        p_2[0] = 1;
        for (unsigned long i = 1; i <= n; ++i) {
            p_1[i] = (p_1[i - 1] * seed_1) % mo_1;
            p_2[i] = (p_2[i - 1] * seed_2) % mo_2;
        }
        h_sum_1[n] = 0;
        h_sum_2[n] = 0;
        p_inv_1[n] = mod_inverse(p_1[n], mo_1);
        p_inv_2[n] = mod_inverse(p_2[n], mo_2);
        for (long i = static_cast<long>(n) - 1; i >= 0; i--) {
            p_inv_1[i] = (p_inv_1[i + 1] * seed_1) % mo_1;
            p_inv_2[i] = (p_inv_2[i + 1] * seed_2) % mo_2;
            h_sum_1[i] = (static_cast<unsigned long>(seq[i]) * p_1[i] + h_sum_1[i + 1]) % mo_1;
            h_sum_2[i] = (static_cast<unsigned long>(seq[i]) * p_2[i] + h_sum_2[i + 1]) % mo_2;
        }
    }

    unsigned long hash(long l, long r) const {
        unsigned long r1 = (h_sum_1[l] - h_sum_1[r] + mo_1) * p_inv_1[l] % mo_1, r2 =
                (h_sum_2[l] - h_sum_2[r] + mo_2) * p_inv_2[l] % mo_2;
        return r1 | (r2 << 32);
    }
};


void collect_identifiers(const std::shared_ptr<Tree> &node, const SeqHasher &hasher,
                         std::unordered_map<unsigned long, std::pair<long, std::vector<long>>> &out) {
    if (node->is_leaf) {
        return;
    }
    if (std::find(IDENTIFIER_NDTYPES.begin(), IDENTIFIER_NDTYPES.end(), node->ndtype) != IDENTIFIER_NDTYPES.end()) {
        unsigned long key = hasher.hash(node->l, node->r);
        if (out.find(key) == out.end()) {
            out[key] = std::make_pair<long, std::vector<long>>(node->get_size(), {node->l});
        } else {
            assert(node->get_size() == out[key].first);
            out[key].second.push_back(node->l);
        }
        return;
    }
    for (const auto &child: node->child) {
        collect_identifiers(child, hasher, out);
    }
}

std::unordered_map<unsigned long, std::pair<long, std::vector<long>>>
collect_identifiers(const std::shared_ptr<Tree> &node, const SeqHasher &hasher) {
    std::unordered_map<unsigned long, std::pair<long, std::vector<long>>> ret;
    collect_identifiers(node, hasher, ret);
    return ret;
}

std::vector<std::pair<long, long>>
calculate_obfuscation(const std::shared_ptr<Tree> &node, const SeqHasher &hasher, long &num_obf_tokens) {
    auto identifier_map = collect_identifiers(node, hasher);
    std::vector<std::pair<long, long>> ret;
    std::vector<long> span_lens;
    std::vector<long> weight;
    long weight_sum = 0;
    std::vector<std::vector<std::pair<long, long>>> spans;
    for (const auto &entry: identifier_map) {
        long span_len;
        std::vector<long> start_pos_s;
        std::tie(span_len, start_pos_s) = entry.second;
        span_lens.push_back(span_len);
        weight.push_back(span_len * static_cast<long>(start_pos_s.size()));
        weight_sum += span_len * static_cast<long>(start_pos_s.size());
        std::vector<std::pair<long, long>> cur_spans;
        cur_spans.reserve(start_pos_s.size());
        for (long start_pos: start_pos_s) {
            cur_spans.emplace_back(start_pos, start_pos + span_len);
        }
        spans.push_back(cur_spans);
    }
    while (weight_sum > 0 && num_obf_tokens > 0) {
        std::discrete_distribution<long> dist(weight.begin(), weight.end());
        long idx = dist(rng);
        if (weight[idx] > num_obf_tokens) {
            assert(span_lens[idx] * static_cast<long>(spans[idx].size()) == weight[idx]);
            long cur_mask_num = num_obf_tokens / span_lens[idx];
            if (cur_mask_num > 0) {
                num_obf_tokens -= span_lens[idx] * cur_mask_num;
                std::uniform_int_distribution<long> dist_uniform(0,
                                                                 static_cast<long>(spans[idx].size()) - cur_mask_num);
                long start_j = dist_uniform(rng);
                ret.insert(ret.end(), spans[idx].begin() + start_j, spans[idx].begin() + start_j + cur_mask_num);
                weight_sum -= weight[idx];
                weight[idx] = 0;
            }
            break;
        } else {
            num_obf_tokens -= weight[idx];
            ret.insert(ret.end(), spans[idx].begin(), spans[idx].end());
            weight_sum -= weight[idx];
            weight[idx] = 0;
        }
    }
    std::sort(ret.begin(), ret.end());
    return ret;
}

void apply_obfuscation(const std::shared_ptr<Tree> &node, const std::vector<std::pair<long, long>> &obf_spans,
                       std::vector<std::pair<long, long>>::iterator &obf_it) {
    if (obf_it == obf_spans.end()) {
        return;
    }
    if (node->l == obf_it->first && node->r == obf_it->second) {
        node->do_mask();
        ++obf_it;
        return;
    }
    if (node->is_leaf) {
        return;
    }
    for (const auto &child: node->child) {
        apply_obfuscation(child, obf_spans, obf_it);
    }
    node->update_mask_sum();
}

void obfuscate(const std::shared_ptr<Tree> &tree, const SeqHasher &hasher, long num_obf_tokens) {
    auto obf_spans = calculate_obfuscation(tree, hasher, num_obf_tokens);
    auto obf_it = obf_spans.begin();
    apply_obfuscation(tree, obf_spans, obf_it);
    assert(obf_it == obf_spans.end());
}

void collect_func_bodies(const std::shared_ptr<Tree> &node, std::vector<std::pair<long, long>> &out) {
    if (node->is_leaf) {
        return;
    }
    if (node->ndtype == TOK_PY_FUNC_BODY) {
        out.emplace_back(node->l, node->r);
    }
    for (const auto &child: node->child) {
        collect_func_bodies(child, out);
    }
}

void mask_func_bodies(const std::shared_ptr<Tree> &tree, long num_t2c_tokens) {
    std::vector<std::pair<long, long>> func_bodies, t2c_spans, filtered_t2c_spans;
    collect_func_bodies(tree, func_bodies);
    std::vector<long> func_body_sizes(func_bodies.size(), 0);
    for (size_t i = 0; i < func_bodies.size(); ++i) {
        func_body_sizes[i] = func_bodies[i].second - func_bodies[i].first;
    }
    auto weights = calculate_shuffle_weights(func_body_sizes, num_t2c_tokens);
    auto func_body_indices = weighted_shuffle(weights);
    for (long ii: func_body_indices) {
        if (func_body_sizes[ii] <= num_t2c_tokens) {
            t2c_spans.push_back(func_bodies[ii]);
            num_t2c_tokens -= func_body_sizes[ii];
        } else {
            break;
        }
    }
    std::sort(t2c_spans.begin(), t2c_spans.end());
    for (auto span : t2c_spans) {
        bool skip = false;
        for (auto span2 : filtered_t2c_spans) {
            if (span2.first <= span.first && span.second <= span2.second) {
                skip = true;
                break;
            }
        }
        if (!skip) {
            filtered_t2c_spans.push_back(span);
        }
    }
    auto t2c_it = filtered_t2c_spans.begin();
    apply_obfuscation(tree, filtered_t2c_spans, t2c_it);
    assert(t2c_it == filtered_t2c_spans.end());
}

std::tuple<std::vector<long>, std::vector<long>, std::vector<long>>
finalize_masks(const std::vector<std::tuple<long, long, long>> &spans, const SeqHasher &hasher, unsigned char *out) {
    std::unordered_map<unsigned long, long> sentinel_id;
    std::vector<long> mask_init_idx, sentinels, span_ndtypes;
    for (const auto &span: spans) {
        long l, r, ndtype;
        std::tie(l, r, ndtype) = span;
        mask_init_idx.push_back(l);
        unsigned long key = hasher.hash(l, r);
        auto sentinel_id_it = sentinel_id.find(key);
        if (sentinel_id_it != sentinel_id.end()) {
            sentinels.push_back(sentinel_id_it->second);
        } else {
            long cur_sentinel_id = static_cast<long>(sentinel_id.size());
            sentinels.push_back(cur_sentinel_id);
            sentinel_id[key] = cur_sentinel_id;
        }
        span_ndtypes.push_back(ndtype);
        for (long j = l; j < r; ++j) {
            *(out + j) = 1;
        }
    }
    return std::make_tuple(mask_init_idx, sentinels, span_ndtypes);
}

std::vector<long> generate_source_seq(const std::vector<long> &seq, const std::vector<unsigned char> &mask,
                                      const std::vector<long> &mask_init_idx, const std::vector<long> &sentinels,
                                      const std::vector<long> &span_ndtypes, bool include_ndtypes = false) {
    std::vector<long> out;
    for (long i = 0, j = 0; j < seq.size(); ++j) {
        if (mask[j]) {
            if (i < mask_init_idx.size() && mask_init_idx[i] == j) {
                out.push_back(sentinels[i] + SENTINEL_IDX_BEGIN + 1);
                if (include_ndtypes && span_ndtypes[i] != -1) {
                    out.push_back(span_ndtypes[i] - NDTYPE_BEGIN + SENTINEL_IDX_BEGIN + MAX_POSITIONS);
                }
                ++i;
            }
        } else {
            out.push_back(seq[j]);
        }
    }
    return out;
}

std::vector<long> generate_target_seq(const std::vector<long> &seq, const std::vector<unsigned char> &mask,
                                      const std::vector<long> &mask_init_idx, const std::vector<long> &sentinels) {
    std::vector<long> out;
    long next_sentinel = 0;
    bool skip = false;
    for (long i = 0, j = 0; j < seq.size(); ++j) {
        if (mask[j]) {
            if (i < mask_init_idx.size() && mask_init_idx[i] == j) {
                if (sentinels[i] == next_sentinel) {
                    out.push_back(sentinels[i] + SENTINEL_IDX_BEGIN + 1);
                    next_sentinel = sentinels[i] + 1;
                    skip = false;
                } else {
                    assert(sentinels[i] < next_sentinel);
                    skip = true;
                }
                ++i;
            }
            if (!skip) {
                out.push_back(seq[j]);
            }
        }
    }
    return out;
}

std::tuple<std::vector<unsigned char>, std::vector<long>, std::vector<long>, std::vector<long>>
generate_masks(const std::vector<long> &seq, const std::vector<long> &binarized_ndtypes, double mask_prob,
               long flat_multiple_len, long threshold_lb, long threshold_ub, double obf_prob, double obf_ratio,
               double t2c_prob, double t2c_ratio, long seed) {
    std::vector<unsigned char> mask(seq.size(), 0);
    std::vector<long> mask_init_idx, sentinels, span_ndtypes;

    FLAT_MULTIPLE_LEN = flat_multiple_len;
    THRESHOLD_LB = threshold_lb;
    THRESHOLD_UB = threshold_ub;
    OBF_PROB = obf_prob;
    OBF_RATIO = obf_ratio;
    T2C_PROB = t2c_prob;
    T2C_RATIO = t2c_ratio;
    rng.seed(seed);

    SeqHasher hasher(seq);
    auto tree = build_tree(binarized_ndtypes, static_cast<long>(seq.size()));
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    long num_masked_tokens = static_cast<long>(static_cast<double>(tree->get_size()) * mask_prob + dist01(rng));
    assert(OBF_PROB + T2C_PROB <= 1.0);
    double r = dist01(rng);
    if (OBF_RATIO > 0.0 && r < OBF_PROB) {
        long num_obf_tokens = static_cast<long>(static_cast<double>(num_masked_tokens) * OBF_RATIO + dist01(rng));
        obfuscate(tree, hasher, num_obf_tokens);
        if (num_masked_tokens - tree->mask_sum > 0) {
            generate_mask(tree, num_masked_tokens - tree->mask_sum);
        }
    } else if (T2C_RATIO > 0.0 && r < OBF_PROB + T2C_PROB) {
        long num_t2c_tokens = static_cast<long>(static_cast<double>(num_masked_tokens) * T2C_RATIO + dist01(rng));
        mask_func_bodies(tree, num_t2c_tokens);
        if (num_masked_tokens - tree->mask_sum > 0) {
            generate_mask(tree, num_masked_tokens - tree->mask_sum);
        }
    } else {
        if (num_masked_tokens > 0) {
            generate_mask(tree, num_masked_tokens);
        }
    }
    auto mask_spans = collect_mask(tree);
    std::tie(mask_init_idx, sentinels, span_ndtypes) = finalize_masks(mask_spans, hasher, mask.data());
    return std::make_tuple(mask, mask_init_idx, sentinels, span_ndtypes);
}

#endif //UNTITLED_LIBAST_HPP
