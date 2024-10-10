#include <metal_stdlib>
using namespace metal;

struct sum{
    template<typename T>
    inline T operator()(thread const T& a, thread const T& b) const {
        return a+b;
    }
    
    template<typename T>
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const {
        return a+b;
    }
};


struct sub{
    template<typename T>
    inline T operator()(thread const T& a, thread const T& b) const {
        return a-b;
    }
    
    template<typename T>
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const {
        return a-b;
    }
};


template<typename T, typename OPERATION>
kernel void operation_vector(device const T* a, device const T* b, device T* c, uint index [[thread_position_in_grid]]){
    OPERATION op;
    c[index] = op(a[index],b[index]);
}

template [[host_name("sum_vectors")]] kernel void operation_vector<float, sum>(device const float*, device const float*, device float*, uint);

template [[host_name("sub_vectors")]] kernel void operation_vector<float, sub>(device const float*, device const float*, device float*, uint);
