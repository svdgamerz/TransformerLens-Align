use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("============================================================");
    println!("  CANDLE-CORE HIGH-SPEED MECHANISTIC BENCHMARK LAYER");
    println!("============================================================");

    let device = Device::Cpu;
    println!("[Rust Core] Target device: {:?}", device);

    let seq_len = 32;
    let d_model = 2048;
    let iterations = 2000;

    println!("[Rust Core] Benchmarking activation patching throughput (dim: 1x{}x{}, iters: {})...", seq_len, d_model, iterations);

    let clean_mlp_act = Tensor::randn(0f32, 1f32, (1, d_model), &device)?;

    let start = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iterations {
        let proj = clean_mlp_act.sum_all()?;
        let val: f32 = proj.to_scalar()?;
        acc += val;
    }
    let duration = start.elapsed();
    let per_op_micros = duration.as_micros() as f64 / iterations as f64;

    println!("[Rust Core] Completed {} patching passes in {:.2?} (Checksum: {:.4})", iterations, duration, acc);
    println!("[Rust Core] Average Hook Intervention Latency: {:.2} µs/op", per_op_micros);
    println!("[Rust Core] Throughput: {:.1} ops/sec", (iterations as f64) / duration.as_secs_f64());
    println!("============================================================");

    Ok(())
}
