use cutile::prelude::*;
use cuda_core::stream::Stream;
use half::{f16, bf16};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CutileKernelsError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cuda_core::error::DriverError),
    #[error("Tensor shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),
}

#[cutile::module]
mod attention_kernels {
    use cutile::core::*;
    use cutile::half::{f16, bf16};

    #[cutile::entry()]
    fn masked_softmax_f32<
        const B: i32,
        const M: i32,
        const KV: i32,
        const MIN_KV_SHARD: i32,
    >(
        output: &mut Tensor<f32, { [B, M, KV] }>,
        attention_scores: &Tensor<f32, { [B, M, KV] }>,
        mask: &Tensor<bool, { [B, M, KV] }>,
    ) {
        let tid = get_tile_thread_id().0;
        let bid = get_tile_block_id().0;
        
        let effective_kv = (KV - 1) / MIN_KV_SHARD + 1;
        let rows_per_block = B * M / get_grid_dim().0;
        
        let row_id = tid / effective_kv;
        let effective_kv_id = tid % effective_kv;
        
        let kv_start = effective_kv_id * MIN_KV_SHARD;
        let kv_end = (kv_start + MIN_KV_SHARD).min(KV);
        
        let batch_seq_id = bid * rows_per_block + row_id;
        if batch_seq_id >= B * M {
            return;
        }
        
        let batch_id = batch_seq_id / M;
        let seq_id = batch_seq_id % M;
        
        extern __shared__ float smem[];
        let row_offset = row_id * 2;
        
        if effective_kv_id == 0 {
            smem[row_offset] = f32::NEG_INFINITY;
            smem[row_offset + 1] = 0.0;
        }
        __syncthreads();
        
        let mut thread_max = f32::NEG_INFINITY;
        for kv in kv_start..kv_end {
            if !mask[[batch_id, seq_id, kv]] {
                let score = attention_scores[[batch_id, seq_id, kv]];
                thread_max = thread_max.max(score);
            }
        }
        
        if thread_max != f32::NEG_INFINITY {
            atomic_max(&mut smem[row_offset], thread_max);
        }
        
        __syncthreads();
        
        let mut thread_sum = 0.0f32;
        let mut exp_vals = [0.0f32; 4];
        
        for (i, kv) in (kv_start..kv_end).enumerate() {
            if !mask[[batch_id, seq_id, kv]] {
                let score = attention_scores[[batch_id, seq_id, kv]];
                let exp_val = (score - smem[row_offset]).exp();
                exp_vals[i] = exp_val;
                thread_sum += exp_val;
            }
        }
        
        if thread_sum > 0.0 {
            atomic_add(&mut smem[row_offset + 1], thread_sum);
        }
        
        __syncthreads();
        
        let denom = smem[row_offset + 1];
        for (i, kv) in (kv_start..kv_end).enumerate() {
            let val = if denom == 0.0 {
                0.0
            } else {
                exp_vals[i] / denom
            };
            output[[batch_id, seq_id, kv]] = val;
        }
    }

    #[cutile::entry()]
    fn masked_softmax_f16<
        const B: i32,
        const M: i32,
        const KV: i32,
        const MIN_KV_SHARD: i32,
    >(
        output: &mut Tensor<f16, { [B, M, KV] }>,
        attention_scores: &Tensor<f16, { [B, M, KV] }>,
        mask: &Tensor<bool, { [B, M, KV] }>,
    ) {
        let tid = get_tile_thread_id().0;
        let bid = get_tile_block_id().0;
        
        let effective_kv = (KV - 1) / MIN_KV_SHARD + 1;
        let rows_per_block = B * M / get_grid_dim().0;
        
        let row_id = tid / effective_kv;
        let effective_kv_id = tid % effective_kv;
        
        let kv_start = effective_kv_id * MIN_KV_SHARD;
        let kv_end = (kv_start + MIN_KV_SHARD).min(KV);
        
        let batch_seq_id = bid * rows_per_block + row_id;
        if batch_seq_id >= B * M {
            return;
        }
        
        let batch_id = batch_seq_id / M;
        let seq_id = batch_seq_id % M;
        
        extern __shared__ float smem[];
        let row_offset = row_id * 2;
        
        if effective_kv_id == 0 {
            smem[row_offset] = f32::NEG_INFINITY;
            smem[row_offset + 1] = 0.0;
        }
        __syncthreads();
        
        let mut thread_max = f32::NEG_INFINITY;
        for kv in kv_start..kv_end {
            if !mask[[batch_id, seq_id, kv]] {
                let score = attention_scores[[batch_id, seq_id, kv]].to_f32();
                thread_max = thread_max.max(score);
            }
        }
        
        if thread_max != f32::NEG_INFINITY {
            atomic_max(&mut smem[row_offset], thread_max);
        }
        
        __syncthreads();
        
        let mut thread_sum = 0.0f32;
        let mut exp_vals = [0.0f32; 4];
        
        for (i, kv) in (kv_start..kv_end).enumerate() {
            if !mask[[batch_id, seq_id, kv]] {
                let score = attention_scores[[batch_id, seq_id, kv]].to_f32();
                let exp_val = (score - smem[row_offset]).exp();
                exp_vals[i] = exp_val;
                thread_sum += exp_val;
            }
        }
        
        if thread_sum > 0.0 {
            atomic_add(&mut smem[row_offset + 1], thread_sum);
        }
        
        __syncthreads();
        
        let denom = smem[row_offset + 1];
        for (i, kv) in (kv_start..kv_end).enumerate() {
            let val = if denom == 0.0 {
                f16::ZERO
            } else {
                f16::from_f32(exp_vals[i] / denom)
            };
            output[[batch_id, seq_id, kv]] = val;
        }
    }

    #[cutile::entry()]
    fn masked_softmax_bf16<
        const B: i32,
        const M: i32,
        const KV: i32,
        const MIN_KV_SHARD: i32,
    >(
        output: &mut Tensor<bf16, { [B, M, KV] }>,
        attention_scores: &Tensor<bf16, { [B, M, KV] }>,
        mask: &Tensor<bool, { [B, M, KV] }>,
    ) {
        let tid = get_tile_thread_id().0;
        let bid = get_tile_block_id().0;
        
        let effective_kv = (KV - 1) / MIN_KV_SHARD + 1;
        let rows_per_block = B * M / get_grid_dim().0;
        
        let row_id = tid / effective_kv;
        let effective_kv_id = tid % effective_kv;
        
        let kv_start = effective_kv_id * MIN_KV_SHARD;
        let kv_end = (kv_start + MIN_KV_SHARD).min(KV);
        
        let batch_seq_id = bid * rows_per_block + row_id;
        if batch_seq_id >= B * M {
            return;
        }
        
        let batch_id = batch_seq_id / M;
        let seq_id = batch_seq_id % M;
        
        extern __shared__ float smem[];
        let row_offset = row_id * 2;
        
        if effective_kv_id == 0 {
            smem[row_offset] = f32::NEG_INFINITY;
            smem[row_offset + 1] = 0.0;
        }
        __syncthreads();
        
        let mut thread_max = f32::NEG_INFINITY;
        for kv in kv_start..kv_end {
            if !mask[[batch_id, seq_id, kv]] {
                let score = attention_scores[[batch_id, seq_id, kv]].to_f32();
                thread_max = thread_max.max(score);
            }
        }
        
        if thread_max != f32::NEG_INFINITY {
            atomic_max(&mut smem[row_offset], thread_max);
        }
        
        __syncthreads();
        
        let mut thread_sum = 0.0f32;
        let mut exp_vals = [0.0f32; 4];
        
        for (i, kv) in (kv_start..kv_end).enumerate() {
            if !mask[[batch_id, seq_id, kv]] {
                let score = attention_scores[[batch_id, seq_id, kv]].to_f32();
                let exp_val = (score - smem[row_offset]).exp();
                exp_vals[i] = exp_val;
                thread_sum += exp_val;
            }
        }
        
        if thread_sum > 0.0 {
            atomic_add(&mut smem[row_offset + 1], thread_sum);
        }
        
        __syncthreads();
        
        let denom = smem[row_offset + 1];
        for (i, kv) in (kv_start..kv_end).enumerate() {
            let val = if denom == 0.0 {
                bf16::ZERO
            } else {
                bf16::from_f32(exp_vals[i] / denom)
            };
            output[[batch_id, seq_id, kv]] = val;
        }
    }
}

pub use attention_kernels::*;

mod quant_matmul;
pub use quant_matmul::*;

pub async fn masked_softmax(
    stream: &Stream,
    attention_scores: &cutile::tensor::Tensor<cutile::DType>,
    mask: &cutile::tensor::Tensor<cutile::DType>,
) -> Result<cutile::tensor::Tensor<cutile::DType>, CutileKernelsError> {
    let shape = attention_scores.shape();
    if shape.len() != 3 {
        return Err(CutileKernelsError::ShapeMismatch(
            "attention_scores must be 3D [B, M, KV]".to_string(),
        ));
    }
    
    let (b, m, kv) = (shape[0] as i32, shape[1] as i32, shape[2] as i32);
    let min_kv_shard = 4;
    
    let output = cutile::api::zeros_like(attention_scores);
    let output = output.partition([1, 1, min_kv_shard as usize]);
    
    let grid_size = (b * m + 1023) / 1024;
    
    match attention_scores.dtype() {
        cutile::DType::F32 => {
            let _ = masked_softmax_f32::<0, 0, 0, 4>(
                output,
                attention_scores,
                mask,
            )
            .generics(vec!["f32".to_string()])
            .grid_dim(grid_size as u32)
            .block_dim(1024)
            .shared_mem((1024 / 4) * 2 * 4)
            .launch(stream)
            .await?;
        }
        cutile::DType::F16 => {
            let _ = masked_softmax_f16::<0, 0, 0, 4>(
                output,
                attention_scores,
                mask,
            )
            .generics(vec!["f16".to_string()])
            .grid_dim(grid_size as u32)
            .block_dim(1024)
            .shared_mem((1024 / 4) * 2 * 4)
            .launch(stream)
            .await?;
        }
        cutile::DType::BF16 => {
            let _ = masked_softmax_bf16::<0, 0, 0, 4>(
                output,
                attention_scores,
                mask,
            )
            .generics(vec!["bf16".to_string()])
            .grid_dim(grid_size as u32)
            .block_dim(1024)
            .shared_mem((1024 / 4) * 2 * 4)
            .launch(stream)
            .await?;
        }
        _ => return Err(CutileKernelsError::UnsupportedDtype(
            format!("Unsupported dtype: {:?}", attention_scores.dtype()),
        )),
    }
    
    Ok(output.unpartition().await?)
}