g++ -std=c++11 trie_st.cpp -O2 -o trie_st
g++ -std=c++11 trie_mt.cpp -O2 -o trie_mt -pthread
nvcc -I../include -arch=compute_70 -std=c++11 -O2 trie.cu --expt-relaxed-constexpr -o trie
g++ -I../include -I/usr/local/cuda/include -std=c++11 benchmark.cpp -O2 -lpthread -o benchmark
g++ -I../include -I/usr/local/cuda/include -std=c++11 benchmark.cpp ../libcxx/src/barrier.cpp ../libcxx/src/atomic.cpp ../libcxx/src/semaphore.cpp -O2 -lpthread -o benchmark -D_LIBCPP_SIMT -D_LIBCUPP_HAS_TREE_BARRIER
nvcc -I../include -arch=compute_70 -std=c++11 benchmark.cu -O2 -lpthread --expt-relaxed-constexpr --expt-extended-lambda -o benchmark
