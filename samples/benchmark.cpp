/*

Copyright (c) 2019, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

// WAIT / NOTIFY
    //#define __NO_TABLE
    //#define __NO_FUTEX
    //#define __NO_CONDVAR
    //#define __NO_SLEEP
    //#define __NO_IDENT
    // To benchmark against spinning
    //#define __NO_SPIN
    //#define __NO_WAIT

// SEMAPHORE
    //#define __NO_SEM
    //#define __NO_SEM_BACK
    //#define __NO_SEM_FRONT
    //#define __NO_SEM_POLL

#include <cmath>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <set>

#include <chrono>

#include <simt/atomic>

//#include <semaphore>
//#include <latch>
//#include <barrier>

struct mutex {
	void lock() noexcept {
		while (1 == l.exchange(1, simt::std::memory_order_acquire))
#ifndef __NO_WAIT
			l.wait(1, simt::std::memory_order_relaxed)
#endif
            ;
	}
	void unlock() noexcept {
		l.store(0, simt::std::memory_order_release);
#ifndef __NO_WAIT
		l.notify_one();
#endif
	}
	simt::std::atomic<int> l = ATOMIC_VAR_INIT(0);
};

struct ticket_mutex {
	void lock() noexcept {
        auto const my = in.fetch_add(1, simt::std::memory_order_acquire);
        while(1) {
            auto const now = out.load(simt::std::memory_order_acquire);
            if(now == my)
                return;
#ifndef __NO_WAIT
            out.wait(now, simt::std::memory_order_relaxed);
#endif
        }
	}
	void unlock() noexcept {
		out.fetch_add(1, simt::std::memory_order_release);
#ifndef __NO_WAIT
		out.notify_all();
#endif
	}
	alignas(64) simt::std::atomic<int> in = ATOMIC_VAR_INIT(0);
    alignas(64) simt::std::atomic<int> out = ATOMIC_VAR_INIT(0);
};

/*
struct sem_mutex {
	void lock() noexcept {
        c.acquire();
	}
	void unlock() noexcept {
        c.release();
	}
	std::binary_semaphore c = 1;
};
*/

static constexpr int sections = 1 << 20;

using sum_mean_dev_t = std::tuple<int, double, double>;

template<class V>
sum_mean_dev_t sum_mean_dev(V && v) {
    assert(!v.empty());
    auto const sum = std::accumulate(v.begin(), v.end(), 0);
    auto const mean = sum / v.size();
    auto const sq_diff_sum = std::accumulate(v.begin(), v.end(), 0.0, [=](auto left, auto right) -> auto {
        return left + (right - mean) * (right - mean);
    });
    auto const variance = sq_diff_sum / v.size();
    auto const stddev = std::sqrt(variance);
    return std::tie(sum, mean, stddev);
}

template <class F>
sum_mean_dev_t test_body(int threads, F && f) {

    std::vector<int> progress(threads, 0);
	std::vector<std::thread> ts(threads);
	for (int i = 0; i < threads; ++i)
		ts[i] = std::thread([&, i]() {
            progress[i] = f(sections / threads);
        });

	for (auto& t : ts)
		t.join();

    return sum_mean_dev(progress);
}

template <class F>
sum_mean_dev_t test_omp_body(int threads, F && f) {
#ifdef _OPENMP
    std::vector<int> progress(threads, 0);
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < threads; ++i)
        progress[i] = f(sections / threads);
    return sum_mean_dev(progress);
#else
    assert(0); // build with -fopenmp
	return sum_mean_dev_t();
#endif
}

template <class F>
void test(std::string const& name, int threads, F && f, std::atomic<bool>& keep_going, bool use_omp = false) {

    std::thread test_helper([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        keep_going.store(false, std::memory_order_relaxed);
    });

    auto const t1 = std::chrono::steady_clock::now();
    auto const smd = use_omp ? test_omp_body(threads, f)
                             : test_body(threads, f);
    auto const t2 = std::chrono::steady_clock::now();

    test_helper.join();

	double const d = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    std::cout << std::setprecision(2) << std::fixed;
	std::cout << name << " : " << d / std::get<0>(smd) << "ns per step, fairness metric = " 
                         << 100 * (1.0 - std::min(1.0, std::get<2>(smd) / std::get<1>(smd))) << "%." 
                         << std::endl;
}

template<class F>
void test_loop(F && f) {
    static int const max = std::thread::hardware_concurrency();
    static std::vector<std::pair<int, std::string>> const counts = 
        { { 1, "single-threaded" }, 
          { max >> 5, "3% occupancy" },
          { max >> 4, "6% occupancy" },
          { max >> 3, "12% occupancy" },
          { max >> 2, "25% occupancy" },
          { max >> 1, "50% occupancy" },
          { max, "100% occupancy" },
//#if !defined(__NO_SPIN) || !defined(__NO_WAIT)
//          { max * 2, "200% occupancy" } 
//#endif
        };
    std::set<int> done{0};
    for(auto const& c : counts) {
        if(done.find(c.first) != done.end())
            continue;
        f(c);
        done.insert(c.first);
    }
}

template<class M>
void test_mutex(std::string const& name, bool use_omp = false) {
    test_loop([&](auto c) {
        M m;
        std::atomic<bool> keep_going(true);
        auto f = [&](int n) -> int {
            int i = 0;
            while(keep_going.load(std::memory_order_relaxed)) {
                m.lock();
                ++i;
                m.unlock();
            }
            return i;
        };
        test(name + ": " + c.second, c.first, f, keep_going);
    });
};

template<class B>
void test_barrier(std::string const& name, bool use_omp = false) {

    test_loop([&](auto c) {
        B b(c.first);
        std::atomic<bool> keep_going(true); // unused here
        auto f = [&](int n)  -> int {
            for (int i = 0; i < n; ++i)
                b.arrive_and_wait();
            return n;
        };
        test(name + ": " + c.second, c.first, f, keep_going, use_omp);
    });
};

int main() {

    int const max = std::thread::hardware_concurrency();
    std::cout << "System has " << max << " hardware threads." << std::endl;

#ifndef __NO_MUTEX
//    test_mutex<sem_mutex>("Semlock");
    test_mutex<mutex>("Spinlock");
    test_mutex<ticket_mutex>("Ticket");
#endif

#ifndef __NO_BARRIER
//    test_barrier<barrier<>>("Barrier");
#endif

#ifdef _OPENMP
    struct omp_barrier {
        omp_barrier(ptrdiff_t) { }
        void arrive_and_wait() {
            #pragma omp barrier
        }
    };
    test_barrier<omp_barrier>("OMP", true);
#endif
/*
#if defined(_POSIX_THREADS) && !defined(__APPLE__)
    struct posix_barrier {
        posix_barrier(ptrdiff_t count) {
            pthread_barrier_init(&pb, nullptr, count);
        }
        ~posix_barrier() {
            pthread_barrier_destroy(&pb);
        }
        void arrive_and_wait() {
            pthread_barrier_wait(&pb);
        }
        pthread_barrier_t pb;
    };
    test_barrier<posix_barrier>("Pthread");
#endif
*/
	return 0;
}
