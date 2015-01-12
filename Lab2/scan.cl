__kernel void scan_blelloch(__global float * a, __global float * r, __local float * b, __global float * sum, int n)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_idx  = get_group_id(0);
    uint block_size = get_local_size(0);
    uint dp = 1;

    b[lid] = a[gid];

    for(uint s = block_size>>1; s > 0; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < s)
        {
            uint i = dp*(2*lid+1)-1;
            uint j = dp*(2*lid+2)-1;
            b[j] += b[i];
        }

        dp <<= 1;
    }

    if (lid == 0) {
        sum[block_idx] = b[block_size - 1];
        b[block_size - 1] = 0;
    }

    for(uint s = 1; s < block_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(lid < s)
        {
            uint i = dp*(2*lid+1)-1;
            uint j = dp*(2*lid+2)-1;

            float t = b[j];
            b[j] += b[i];
            b[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    r[gid] = b[lid];
}


__kernel void blocks_sum(__global float * a, __global float * r, __global float * sums)
{
    r[get_global_id(0)] = a[get_global_id(0)] + sums[get_group_id(0)];
}