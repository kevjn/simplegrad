__kernel void relu_forward(__global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  float a = a_g[gid];
  res_g[gid] = max(a, (float)0.);
}


// inline all operations for the compiler