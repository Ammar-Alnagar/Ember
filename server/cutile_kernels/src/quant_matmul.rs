use cutile::prelude::*;
use cutile::half::f16;
use cuda_core::stream::Stream;

#[cutile::module]
mod quant_kernels {
    use cutile::core::*;
    use cutile::half::f16;

    #[cutile::entry()]
    fn q4_matmul_kernel<
        const BLOCK_M: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
        const GROUPSIZE: i32,
        const USE_HALF2: bool,
    >(
        output: &mut Tensor<f16, { [BLOCK_M, BLOCK_N] }>,
        input: &Tensor<f16, { [BLOCK_M, BLOCK_K] }>,
        qweight: &Tensor<u32, { [BLOCK_K, BLOCK_N] }>,
        scales: &Tensor<f16, { [BLOCK_K / GROUPSIZE, BLOCK_N] }>,
        qzeros: &Tensor<u32, { [BLOCK_K / GROUPSIZE, BLOCK_N] }>,
    ) {
        let tid = get_tile_thread_id();
        let bid = get_tile_block_id();
        
        let row = bid.0 * BLOCK_M + tid.0;
        let col = bid.1 * BLOCK_N + tid.1;
        
        if row >= output.shape()[0] || col >= output.shape()[1] {
            return;
        }
        
        let mut acc: f32 = 0.0;
        
        for k_tile in 0..(output.shape()[2] / BLOCK_K) {
            let k_base = k_tile * BLOCK_K;
            
            for k in 0..BLOCK_K {
                let k_idx = k_base + k;
                let group_idx = k_idx / GROUPSIZE;
                
                let x_val = input[[tid.0, k]];
                
                let weight_u32 = qweight[[k, tid.1]];
                let weight_nibble = (weight_u32 >> (4 * (tid.0 % 8))) & 0xF;
                let weight = weight_nibble as i32;
                
                let scale = if USE_HALF2 {
                    scales[[group_idx, tid.1]].to_f32()
                } else {
                    scales[[group_idx, tid.1]].to_f32()
                };
                
                let zero = (qzeros[[group_idx, tid.1]] >> (4 * (tid.0 % 8))) & 0xF;
                let zero = zero as i32 + 1;
                
                let dequantized = (weight - zero) as f32 * scale;
                acc += x_val.to_f32() * dequantized;
            }
        }
        
        output[[tid.0, tid.1]] = f16::from_f32(acc);
    }

    #[cutile::entry()]
    fn q4_matmul_reconstruct_kernel<
        const BLOCK_M: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
        const GROUPSIZE: i32,
    >(
        output: &mut Tensor<f16, { [BLOCK_K, BLOCK_N] }>,
        qweight: &Tensor<u32, { [BLOCK_K, BLOCK_N] }>,
        scales: &Tensor<f16, { [BLOCK_K / GROUPSIZE, BLOCK_N] }>,
        qzeros: &Tensor<u32, { [BLOCK_K / GROUPSIZE, BLOCK_N] }>,
    ) {
        let tid = get_tile_thread_id();
        let bid = get_tile_block_id();
        
        let k = bid.0 * BLOCK_K + tid.0;
        let n = bid.1 * BLOCK_N + tid.1;
        
        if k >= output.shape()[0] || n >= output.shape()[1] {
            return;
        }
        
        let group_idx = k / GROUPSIZE;
        let weight_u32 = qweight[[tid.0, tid.1]];
        let weight_nibble = (weight_u32 >> (4 * (tid.0 % 8))) & 0xF;
        let weight = weight_nibble as i32;
        
        let scale = scales[[group_idx, tid.1]].to_f32();
        let zero = (qzeros[[group_idx, tid.1]] >> (4 * (tid.0 % 8))) & 0xF;
        let zero = zero as i32 + 1;
        
        let dequantized = (weight - zero) as f32 * scale;
        output[[tid.0, tid.1]] = f16::from_f32(dequantized);
    }

    #[cutile::entry()]
    fn column_remap_kernel<
        const HEIGHT: i32,
        const WIDTH: i32,
    >(
        output: &mut Tensor<f16, { [HEIGHT, WIDTH] }>,
        input: &Tensor<f16, { [HEIGHT, WIDTH] }>,
        remap: &Tensor<u32, { [WIDTH] }>,
    ) {
        let tid = get_tile_thread_id();
        
        if tid.0 >= HEIGHT || tid.1 >= WIDTH {
            return;
        }
        
        let src_col = remap[[tid.1]] as usize;
        output[[tid.0, tid.1]] = input[[tid.0, src_col]];
    }
}

pub use quant_kernels::*;

pub async fn q4_matmul(
    stream: &Stream,
    input: &cutile::tensor::Tensor<cutile::DType>,
    qweight: &cutile::tensor::Tensor<cutile::DType>,
    scales: &cutile::tensor::Tensor<cutile::DType>,
    qzeros: &cutile::tensor::Tensor<cutile::DType>,
    groupsize: i32,
    block_size_z: i32,
) -> Result<cutile::tensor::Tensor<cutile::DType>, crate::CutileKernelsError> {
    let input_shape = input.shape();
    let qweight_shape = qweight.shape();
    
    let height = input_shape[0];
    let dim = input_shape[1];
    let width = qweight_shape[1];
    
    let block_m = 32;
    let block_n = 32;
    let block_k = 8;
    
    let output = cutile::api::zeros(&[height, width], cutile::DType::F16);
    let output = output.partition([block_m, block_n]);
    
    let grid_m = (height + block_m - 1) / block_m;
    let grid_n = (width + block_n - 1) / block_n;
    
    let _ = q4_matmul_kernel::<32, 32, 8, 128, true>(
        output,
        input,
        qweight,
        scales,
        qzeros,
    )
    .grid_dim([grid_m as u32, grid_n as u32, 1])
    .block_dim([32, 32, 1])
    .launch(stream)
    .await?;
    
    Ok(output.unpartition().await?)
}

pub async fn q4_matmul_reconstruct(
    stream: &Stream,
    qweight: &cutile::tensor::Tensor<cutile::DType>,
    scales: &cutile::tensor::Tensor<cutile::DType>,
    qzeros: &cutile::tensor::Tensor<cutile::DType>,
    groupsize: i32,
) -> Result<cutile::tensor::Tensor<cutile::DType>, crate::CutileKernelsError> {
    let qweight_shape = qweight.shape();
    let dim = qweight_shape[0];
    let width = qweight_shape[1];
    
    let block_k = 128;
    let block_n = 128;
    
    let output = cutile::api::zeros(&[dim, width], cutile::DType::F16);
    let output = output.partition([block_k, block_n]);
    
    let grid_k = (dim + block_k - 1) / block_k;
    let grid_n = (width + block_n - 1) / block_n;
    
    let _ = q4_matmul_reconstruct_kernel::<128, 128, 128, 128>(
        output,
        qweight,
        scales,
        qzeros,
    )
    .grid_dim([grid_k as u32, grid_n as u32, 1])
    .block_dim([16, 16, 1])
    .launch(stream)
    .await?;
    
    Ok(output.unpartition().await?)
}

pub async fn column_remap(
    stream: &Stream,
    input: &cutile::tensor::Tensor<cutile::DType>,
    remap: &cutile::tensor::Tensor<cutile::DType>,
) -> Result<cutile::tensor::Tensor<cutile::DType>, crate::CutileKernelsError> {
    let input_shape = input.shape();
    let height = input_shape[0];
    let width = input_shape[1];
    
    let output = cutile::api::zeros_like(input);
    let output = output.partition([16, 16]);
    
    let grid_h = (height + 15) / 16;
    let grid_w = (width + 15) / 16;
    
    let _ = column_remap_kernel::<16, 16>(
        output,
        input,
        remap,
    )
    .grid_dim([grid_h as u32, grid_w as u32, 1])
    .block_dim([16, 16, 1])
    .launch(stream)
    .await?;
    
    Ok(output.unpartition().await?)
}