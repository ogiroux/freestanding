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

#ifdef __NVCC__
# define _ABI __host__ __device__
# define check(ans) { assert_((ans), __FILE__, __LINE__); }
inline void assert_(cudaError_t code, const char *file, int line) {
  if (code == cudaSuccess)
    return;
  std::cerr << "check failed: " << cudaGetErrorString(code) << " : " << file << '@' << line << std::endl;
  abort();
}
#else
# define _ABI
#endif

template <class T>
struct managed_allocator {
  typedef simt::std::size_t size_type;
  typedef simt::std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;// (deprecated in C++17)(removed in C++20)	T*
  typedef const T* const_pointer;// (deprecated in C++17)(removed in C++20)	const T*
  typedef T& reference;// (deprecated in C++17)(removed in C++20)	T&
  typedef const T& const_reference;// (deprecated in C++17)(removed in C++20)	const T&

  template< class U > struct rebind { typedef managed_allocator<U> other; };
  managed_allocator() = default;
  template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* out = nullptr;
#ifdef __NVCC__
# ifdef __aarch64__
    check(cudaMallocHost(&out, n*sizeof(T), cudaHostAllocMapped));
    void* out2;
    check(cudaHostGetDevicePointer(&out2, out, 0));
    assert(out2==out); //< we can't handle non-uniform addressing
# else
    check(cudaMallocManaged(&out, n*sizeof(T)));
# endif
#else
    out = malloc(n*sizeof(T));
#endif
    return static_cast<T*>(out);
  }
  void deallocate(T* p, std::size_t) noexcept { 
#ifdef __NVCC__ 
# ifdef __aarch64__
    check(cudaFreeHost(p));
# else
    check(cudaFree(p));
# endif
#else
    free(p);
#endif
  }
};
template<class T, class... Args>
T* make_(Args &&... args) {
    managed_allocator<T> ma;
    return new (ma.allocate(1)) T(std::forward<Args>(args)...);
}

struct mutex {
	_ABI void lock() noexcept {
		while (1 == l.exchange(1, simt::std::memory_order_acquire))
#ifndef __NO_WAIT
			l.wait(1, simt::std::memory_order_relaxed)
#endif
            ;
	}
	_ABI void unlock() noexcept {
		l.store(0, simt::std::memory_order_release);
#ifndef __NO_WAIT
		l.notify_one();
#endif
	}
	simt::std::atomic<int> l = ATOMIC_VAR_INIT(0);
};

struct ticket_mutex {
	_ABI void lock() noexcept {
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
	_ABI void unlock() noexcept {
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

#ifdef __NVCC__
template<class F>
__global__ void launcher(F f, int s_per_t, int* p) {
    p[blockIdx.x * blockDim.x + threadIdx.x] = (*f)(s_per_t);
}
#endif

template <class F>
sum_mean_dev_t test_body(int threads, F f) {

    std::vector<int, managed_allocator<int>> progress(threads, 0);

#ifdef __NVCC__
    auto f_ = make_<F>(f);
    launcher<<<threads, 1>>>(f_, sections / threads, &progress[0]);
    cudaDeviceSynchronize();
#else
	std::vector<std::thread> ts(threads);
	for (int i = 0; i < threads; ++i)
		ts[i] = std::thread([&, i]() {
            progress[i] = f(sections / threads);
        });
	for (auto& t : ts)
		t.join();
#endif

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
void test(std::string const& name, int threads, F && f, simt::std::atomic<bool>& keep_going, bool use_omp = false) {

    std::thread test_helper([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        keep_going.store(false, simt::std::memory_order_relaxed);
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
    test_loop([&](std::pair<int, std::string> c) {
        M* m = make_<M>();
        simt::std::atomic<bool> *keep_going = make_<simt::std::atomic<bool>>(true);
        auto f = [=] _ABI (int n) -> int {
            int i = 0;
            while(keep_going->load(simt::std::memory_order_relaxed)) {
                m->lock();
                ++i;
                m->unlock();
            }
            return i;
        };
        test(name + ": " + c.second, c.first, f, *keep_going);
    });
};

template<class B>
void test_barrier(std::string const& name, bool use_omp = false) {

    test_loop([&](std::pair<int, std::string> c) {
        B* b = make_<B>(c.first);
        simt::std::atomic<bool> *keep_going = make_<simt::std::atomic<bool>>(true);
        auto f = [=] _ABI (int n)  -> int {
            for (int i = 0; i < n; ++i)
                b->arrive_and_wait();
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
