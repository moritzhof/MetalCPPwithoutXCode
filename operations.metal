#include <metal_stdlib>
using namespace metal;

struct sum{
    template<typename T, typename U>
    inline T operator()(thread const T& a, thread const T& b) const {
        return a+b;
    }
    
    template<typename T, typename U>
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const {
        return a+b;
    }
};


kernel void add_vector(device const float* a, device const float* b, device float* c, uint index [[thread_position_in_grid]]){
    
    c[index] = a[index]+b[index];
}
