__kernel void relu__default(__global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = max(a, (float)0.);
}

__kernel void greater_equal__default(__global const float *a_g, __global float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = a >= b_g[0];
}

__kernel void pow__binary(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __const int res_dims,
                          __global int *res_strides)
{
  int i = 0, ix = 0, iy = 0;
  for(int dim = 0; dim < res_dims; dim++)
  {
    i += get_global_id(dim) * res_strides[dim];
    ix += get_global_id(dim) * x_strides[dim];
    iy += get_global_id(dim) * y_strides[dim];
  }

  res_g[i] = pow(x_g[ix], y_g[iy]);
}


__kernel void add__binary(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __const int res_dims,
                          __global int *res_strides)
{
  int i = 0, ix = 0, iy = 0;
  //iterate over trailing axes, the stride for non-trailing axes is set to 0 on host.
  for(int dim = 0; dim < res_dims; dim++) // res_dims = get_work_dim()
  {
    i += get_global_id(dim) * res_strides[dim];
    ix += get_global_id(dim) * x_strides[dim];
    iy += get_global_id(dim) * y_strides[dim];
  }

  res_g[i] = x_g[ix] + y_g[iy];
}

__kernel void max__reduction(__global const float* buffer,
            __global const int* strides,
            __global const int* anchored_axes,
            __const int reduced_axis,
            __const int reduced_axis_size,
            __global float* result,
            __global const int* result_strides)
{
  float accum = -INFINITY; // identity

  int offset = 0;
  int idx = 0;
  for (int dim = 0; dim < get_work_dim(); dim++)
  {
    offset += get_global_id(dim) * strides[anchored_axes[dim]];
    idx += get_global_id(dim) * result_strides[dim];
  }

  // sum over k
  for (int k = 0; k < reduced_axis_size; k++)
  {
    accum = max(accum, buffer[k * strides[reduced_axis] + offset]);
  }

  result[idx] = accum;
}

__kernel void argmax__reduction(__global const float* buffer,
            __global const int* strides,
            __global const int* anchored_axes,
            __const int reduced_axis,
            __const int reduced_axis_size,
            __global int* result,
            __global const int* result_strides)
{
  float _max = -INFINITY; // identity

  int offset = 0;
  int idx = 0;
  for (int dim = 0; dim < get_work_dim(); dim++)
  {
    offset += get_global_id(dim) * strides[anchored_axes[dim]];
    idx += get_global_id(dim) * result_strides[dim];
  }

  // sum over k
  int max_idx = 0;
  for (int k = 0; k < reduced_axis_size; k++)
  {
    float val = buffer[k * strides[reduced_axis] + offset];
    if (_max < val)
    {
      _max = val;
      max_idx = k;
    }
  }

  result[idx] = max_idx;
}

__kernel void exp__default(__global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = exp(a);
}

__kernel void log__default(__global const float *a_g, __global float *res_g)
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
void sum__reduction(__global const float* buffer,
            __global const int* strides,
            __global const int* anchored_axes,
            __const int reduced_axis,
            __const int reduced_axis_size,
            __global float* result,
            __global const int* result_strides)
{
  float accum = 0; // identity

  int offset = 0;
  int idx = 0;
  for (int dim = 0; dim < get_work_dim(); dim++)
  {
    offset += get_global_id(dim) * strides[anchored_axes[dim]];
    idx += get_global_id(dim) * result_strides[dim];
  }

  // sum over k
  for (int k = 0; k < reduced_axis_size; k++)
  {
    accum += buffer[k * strides[reduced_axis] + offset];
  }

  result[idx] = accum;
}

__kernel void broadcast_to__binary(__global const float *dv_g,
                          __global const int* dv_strides,
                          __global const float *x_g, 
                          __global const int* x_strides,
                          __global float *res_g,
                          __const int res_dims,
                          __global int *res_strides)
{
  // broadcast dv
  int i = 0, idv = 0;
  for(int dim = 0; dim < res_dims; dim++)
  {
    i += get_global_id(dim) * res_strides[dim];
    idv += get_global_id(dim) * dv_strides[dim];
  }
  res_g[i] = dv_g[idv];
}

__kernel
void einsum__einsum(__global const float* x_g,
            __global const float* y_g,
            __global const int* x_strides,
            __global const int* y_strides,
            __const int reduced_axis_stride_x,
            __const int reduced_axis_stride_y,
            __const int reduced_axis_size,
            __global float* result,
            __global const int* result_strides)
{

  float accum = 0; // identity

  int idx = 0, ix = 0, iy = 0;
  for (int dim = 0; dim < get_work_dim(); dim++)
  {
    idx += get_global_id(dim) * result_strides[dim];
    ix += get_global_id(dim) * x_strides[dim];
    iy += get_global_id(dim) * y_strides[dim];
  }

  // sum over k
  for (int k = 0; k < reduced_axis_size; k++)
  {
    accum += x_g[k * reduced_axis_stride_x + ix] * y_g[k * reduced_axis_stride_y + iy];
  }

  result[idx] = accum;
}

__kernel 
void mul__binary(__global const float *x_g,
                          __global const int* x_strides,
                          __global const float *y_g, 
                          __global const int* y_strides,
                          __global float *res_g,
                          __const int res_dims,
                          __global int *res_strides)
{
  int idx = 0, ix = 0, iy = 0;
  for (int dim = 0; dim < get_work_dim(); dim++)
  {
    idx += get_global_id(dim) * res_strides[dim];
    ix += get_global_id(dim) * x_strides[dim];
    iy += get_global_id(dim) * y_strides[dim];
  }

  res_g[idx] = x_g[ix] * y_g[iy];
}