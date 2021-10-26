__kernel void maximum__elementwise(__global const float *a_g, __global float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = max(a, (float)0.); // Currently, maximum is only used for ReLU
}

__kernel void greater_equal__elementwise(__global const float *a_g, __global float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = a >= b_g[0];
}

__kernel void equal__broadcast(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __global int *res_strides)
{
  int gid = get_global_id(0);

  int i = res_strides[gid];
  int ix = x_strides[gid];
  int iy = y_strides[gid];

  res_g[i] = x_g[ix] == y_g[iy];
}

__kernel void power__broadcast(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __global int *res_strides)
{
  int gid = get_global_id(0);

  int i = res_strides[gid];
  int ix = x_strides[gid];
  int iy = y_strides[gid];

  res_g[i] = pow(x_g[ix], y_g[iy]);
}


__kernel void add__broadcast(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __global int *res_strides)
{
  int gid = get_global_id(0);

  int i = res_strides[gid];
  int ix = x_strides[gid];
  int iy = y_strides[gid];

  res_g[i] = x_g[ix] + y_g[iy];
}

__kernel void amax__reduce(__global const float* buffer,
            __global const int* strides,
            __global const int* reduced_axes_stride,
            __const int reduced_axis_size,
            __global float* result,
            __global const int* result_strides)
{
  float accum = -INFINITY; // identity

  int idx = result_strides[get_global_id(0)];
  int offset = strides[get_global_id(0)];

  // sum over k
  for (int k = 0; k < reduced_axis_size; k++)
  {
    accum = max(accum, buffer[reduced_axes_stride[k] + offset]);
  }

  result[idx] = accum;
}

__kernel void argmax__reduce(__global const float* buffer,
            __global const int* strides,
            __global const int* reduced_axes_stride,
            __const int reduced_axis_size,
            __global float* result,
            __global const int* result_strides)
{
  float _max = -INFINITY; // identity

  int idx = result_strides[get_global_id(0)];
  int offset = strides[get_global_id(0)];

  // sum over k
  int max_idx = 0;
  for (int k = 0; k < reduced_axis_size; k++)
  {
    float val = buffer[reduced_axes_stride[k] + offset];
    if (_max < val)
    {
      _max = val;
      max_idx = k;
    }
  }

  result[idx] = max_idx;
}

__kernel void exp__elementwise(__global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = exp(a);
}

__kernel void log__elementwise(__global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = log(a);
}

//-------------------------------------------------------------------------------
// 
//  kernel: sum_forward
//
//  Purpose: compute sum reduction for multidimensional array along given axis.
//           Each element in dimension which does not belong to the reduced axis has its own thread, 
//           ex. total threads = X + Y + Z if input dimension is [X,Y,Z,W] and W is being reduced.
//           Every thread performs reduction of every N'th element in the input array where
//           N is the corresponding dimension stride. The result is stored in its accumulator
//           which later gets assigned to the result.
//           TODO: extend reduction across all axis using interleaved pairs, see
//                 https://stackoverflow.com/questions/37066164/method-to-do-final-sum-with-reduction
//
//  input: __global const float* buffer: the input array
//         __global const int* strides: strides for the input array (float aligned)
//         __global const int* anchored_axes: indicies for axes which are not being reduced
//         __const int reduced_axis: the axis being reduced
//         __const int reduced_axis_size: length of the reduced axis
//
//  output: __global float* result: the output array
//          __global const int* reult_strides: strides for the output array (float aligned)

__kernel
void sum__reduce(__global const float* buffer,
            __global const int* strides,
            __global const int* reduced_axes_stride,
            __const int reduced_axis_size,
            __global float* result,
            __global const int* result_strides)
{
  float accum = 0; // identity

  int idx = result_strides[get_global_id(0)];
  int offset = strides[get_global_id(0)];

  // sum over k
  for (int k = 0; k < reduced_axis_size; k++)
  {
    accum += buffer[reduced_axes_stride[k] + offset];
  }

  result[idx] = accum;
}

// Copy for non-contiguous memory (only works for int dtype)
__kernel void copy__broadcast(__global const int *a_g,
                          __global const int* a_strides,
                          __global int *res_g,
                          __global const int *res_strides)
{
  int gid = get_global_id(0);
  res_g[res_strides[gid]] = a_g[a_strides[gid]];
}

__kernel
void einsum__einsum(__global const float* x_g,
            __global const float* y_g,
            __global const int* x_strides,
            __global const int* y_strides,
            __global const int* reduced_axes_stride_x,
            __global const int* reduced_axes_stride_y,
            __const int reduced_axes_size,
            __global float* result,
            __global const int* result_strides)
{

  float accum = 0; // identity

  int gid = get_global_id(0);

  int idx = result_strides[gid];
  int ix = x_strides[gid];
  int iy = y_strides[gid];

  // sum over k
  for (int k = 0; k < reduced_axes_size; k++)
  {
    accum += x_g[reduced_axes_stride_x[k] + ix] * y_g[reduced_axes_stride_y[k] + iy];
  }

  result[idx] = accum;
}

__kernel 
void multiply__broadcast(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __global int *res_strides)
{
  int gid = get_global_id(0);

  int i = res_strides[gid];
  int ix = x_strides[gid];
  int iy = y_strides[gid];

  res_g[i] = x_g[ix] * y_g[iy];
}

__kernel 
void bincount__bincount(__global const int *x_g,
                        __global const float *y_g, 
                        __global float *res_g)
{
  int gid = get_global_id(0);
  int i = x_g[gid];

  res_g[i] += y_g[gid];
}
