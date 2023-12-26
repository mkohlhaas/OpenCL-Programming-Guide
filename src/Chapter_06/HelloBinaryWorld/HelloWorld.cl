kernel void hello_kernel(global const float *a,
                         global const float *b,
                         global       float *result)
{
    int i = get_global_id(0);
    result[i] = a[i] + b[i];
}
