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

__kernel void add_forward(__global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  float b = b_g[gid];
  res_g[gid] = a + b;
}
