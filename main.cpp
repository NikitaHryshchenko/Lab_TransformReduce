#include <algorithm>
#include <chrono>
#include <execution>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

template <typename F>
double measure_ms(F&& f) {
    auto start = Clock::now();
    f();
    auto end = Clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    return diff.count();
}

template <typename It, typename T, typename BinaryOp, typename UnaryOp>
T manual_parallel_transform_reduce(It first, It last,
    T init,
    BinaryOp reduce_op,
    UnaryOp transform_op,
    std::size_t K)
{
    auto n = std::distance(first, last);
    if (n <= 0 || K == 0)
        return init;

    if (static_cast<std::size_t>(n) < K)
        K = static_cast<std::size_t>(n);

    std::vector<T> partials(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    std::size_t total = n;
    std::size_t base = total / K;
    std::size_t rem = total % K;

    It block_start = first;

    for (std::size_t i = 0; i < K; ++i) {
        std::size_t block_size = base + (i < rem ? 1 : 0);
        It block_end = block_start;
        std::advance(block_end, block_size);

        It bs = block_start;
        It be = block_end;

        threads.emplace_back([&partials, i, bs, be, reduce_op, transform_op]() {
            partials[i] = std::transform_reduce(
                bs, be,
                T{}, reduce_op, transform_op
            );
            });

        block_start = block_end;
    }

    for (auto& th : threads)
        th.join();

    T result = init;
    for (auto& v : partials)
        result = reduce_op(result, v);

    return result;
}

int main() {
    using T = double;

    std::cout << std::fixed << std::setprecision(3);

    std::vector<std::size_t> sizes = {
        100000,
        1000000,
        5000000
    };

    auto reduce_op = std::plus<T>{};
    auto transform_op = [](T x) { return x * x; };

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    unsigned hw = std::thread::hardware_concurrency();
    if (!hw) hw = 1;

    std::cout << "Hardware threads: " << hw << "\n";

    volatile T sink = 0;

    for (auto size : sizes) {
        std::cout << "\n===== SIZE = " << size << " =====\n";

        std::vector<T> data(size);
        for (auto& x : data) x = dist(rng);

        {
            T r{};
            double t = measure_ms([&]() {
                r = std::transform_reduce(
                    data.begin(), data.end(),
                    T{}, reduce_op, transform_op
                );
                });
            sink = r;
            std::cout << "transform_reduce (no policy):   " << t << " ms\n";
        }

        {
            T r{};
            double t = measure_ms([&]() {
                r = std::transform_reduce(
                    std::execution::par,
                    data.begin(), data.end(),
                    T{}, reduce_op, transform_op
                );
                });
            sink = r;
            std::cout << "transform_reduce (par):         " << t << " ms\n";
        }

        {
            T r{};
            double t = measure_ms([&]() {
                r = std::transform_reduce(
                    std::execution::par_unseq,
                    data.begin(), data.end(),
                    T{}, reduce_op, transform_op
                );
                });
            sink = r;
            std::cout << "transform_reduce (par_unseq):   " << t << " ms\n";
        }

        std::cout << "\nManual parallel transform_reduce:\n";
        std::cout << "K\tTime_ms\n";

        double best_time = 1e18;
        std::size_t best_K = 1;

        for (std::size_t K = 1; K <= hw * 2; ++K) {
            T r{};
            double t = measure_ms([&]() {
                r = manual_parallel_transform_reduce(
                    data.begin(), data.end(),
                    T{}, reduce_op, transform_op,
                    K
                );
                });

            sink = r;

            std::cout << K << "\t" << t << "\n";

            if (t < best_time) {
                best_time = t;
                best_K = K;
            }
        }

        std::cout << "\nBest K = " << best_K
            << ", K/hw = " << (double)best_K / hw
            << "\n";
    }

    return 0;
}
