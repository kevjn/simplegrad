__kernel void relu_forward(__global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = max(a, (float)0.);
}

__kernel void relu_backward(__global const float *dv_g, __global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  float dv = dv_g[gid];
  res_g[gid] = dv * (a >= 0);
}

__kernel void add_forward(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __const int res_dims,
                          __global int *res_strides)
{
  int i = 0, ix = 0, iy = 0;
  //iterate over trailing axes, the stride for non-trailing axes is set to 0 on host.
  for(int dim = 0; dim < res_dims; dim++)
  {
    i += get_global_id(dim) * res_strides[dim];
    ix += get_global_id(dim) * x_strides[dim];
    iy += get_global_id(dim) * y_strides[dim];
  }

  res_g[i] = x_g[ix] + y_g[iy];
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
//         __const int result_ndims: result dimension (this might be same as get_global_size(0))
//
//  output: __global float* result: the output array
//         __global const int* reult_strides: strides for the output array (float aligned)

__kernel
void sum_forward(__global const float* buffer,
            __global const int* strides,
            __global const int* anchored_axes,
            __const int reduced_axis,
            __const int reduced_axis_size,
            __const int result_ndims,
            __global float* result,
            __global const int* result_strides)
{
  float accum = 0; // identity

  // sum over k
  for (int k = 0; k < reduced_axis_size; k++)
  {
    int offset = 0;
    for (int ax = 0; ax < result_ndims; ax++)
    {
      offset += get_global_id(ax) * strides[anchored_axes[ax]];
    }
    accum += buffer[k * strides[reduced_axis] + offset];
  }

  int idx = 0;
  for (int ax = 0; ax < result_ndims; ax++)
  {
    idx += get_global_id(ax) * result_strides[ax];
  }
  result[idx] = accum;
}