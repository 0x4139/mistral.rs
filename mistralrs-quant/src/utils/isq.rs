use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Tensor, D,
};

use crate::{
    get_immediate_isq, pending_layer, ImmediateIsqMatch, ImmediateIsqParams, IsqConsumer,
    IsqRequest, IsqType, PendingIsqLayer, QuantMethod, ShardedVarBuilder, TrackedModule,
};

pub enum QuantizationBehavior {
    Quantize(GgmlDType),
    Skip,
}

pub fn apply_immediate_isq(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    apply_immediate_isq_sharded(layer, vb, Some(crate::Shard::default()))
}

/// Like [`apply_immediate_isq`], recording the rank slice so from-source requantization can
/// re-slice; pass None when the load applied a transform a shard cannot express.
pub fn apply_immediate_isq_sharded(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
    shard: Option<crate::Shard>,
) -> Result<Arc<dyn QuantMethod>> {
    apply_immediate_isq_inner(layer, vb, None, shard)
}

pub fn apply_immediate_isq_with_key(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
    key: Option<String>,
    shard: Option<crate::Shard>,
) -> Result<Arc<dyn QuantMethod>> {
    apply_immediate_isq_inner(layer, vb, key, shard)
}

fn apply_immediate_isq_inner(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
    key: Option<String>,
    shard: Option<crate::Shard>,
) -> Result<Arc<dyn QuantMethod>> {
    let Some(params) = get_immediate_isq() else {
        return Ok(layer);
    };
    let prefix = format!("{}.weight", vb.prefix());
    if let Some(ImmediateIsqMatch {
        ty,
        device,
        promote_default,
    }) = crate::resolve_immediate_isq(&params, &prefix)
    {
        let device = if params.capture == crate::IsqCaptureMode::CaptureAll {
            Device::Cpu
        } else {
            device.unwrap_or_else(|| vb.device().clone())
        };

        // Capture modes keep the layer unquantized; the resolved ty is recorded for later.
        let spawn_ty = match params.capture {
            crate::IsqCaptureMode::Immediate => ty,
            _ => None,
        };
        let module_key = key.unwrap_or_else(|| vb.prefix());
        let layer = spawn_pending_isq(layer, spawn_ty, device, &params, module_key.clone());
        vb.tracker().add_module(TrackedModule {
            key: module_key,
            ct: layer.clone(),
            ty,
            promote_default,
            shard,
        });
        Ok(layer)
    } else {
        Ok(layer)
    }
}

pub(crate) fn spawn_pending_isq(
    layer: Arc<dyn QuantMethod>,
    ty: Option<IsqType>,
    device: Device,
    params: &ImmediateIsqParams,
    module_key: String,
) -> Arc<PendingIsqLayer> {
    let guard = params
        .guard
        .clone()
        .with_module_key(module_key.clone())
        .with_consumer(IsqConsumer::ImmediateLoad);
    let request = IsqRequest {
        ty,
        device: device.clone(),
        has_imatrix: false,
        capture: params.capture,
        consumer: IsqConsumer::ImmediateLoad,
        module_key,
    };
    let rx = match layer.plan_isq(&request) {
        Ok(plan) => params.executor.submit(plan, request.consumer, move || {
            layer
                .clone()
                .apply_isq(ty, device, &AtomicUsize::new(0), None, guard)
        }),
        Err(e) => {
            let (tx, rx) = pending_layer::pending_isq_channel();
            let _ = tx.send(Err(e));
            rx
        }
    };
    Arc::new(PendingIsqLayer::new(rx))
}

/// In-flight parallel requantization; receivers are in the same order as the input modules.
/// Holds the pool so spawned jobs outlive the call.
pub struct RequantizeHandles {
    _executor: crate::IsqExecutor,
    pub receivers: Vec<pending_layer::IsqReceiver>,
}

/// Quantize a rebuilt `[E, out, in]` expert stack to `ty`: GGML types go slab-by-slab so each
/// expert can take its own importance vector; other types quantize the whole stack.
pub fn quantize_expert_stack(
    stack: Tensor,
    ty: IsqType,
    imatrix: Option<Vec<f32>>,
    device: &Device,
    guard: crate::QuantizeOntoGuard,
) -> Result<Arc<dyn QuantMethod>> {
    quantize_expert_stack_with_bias(
        stack,
        None,
        ty,
        imatrix,
        device,
        &AtomicUsize::new(0),
        guard,
    )
}

pub fn quantize_expert_stack_with_bias(
    stack: Tensor,
    bias: Option<Tensor>,
    ty: IsqType,
    imatrix: Option<Vec<f32>>,
    device: &Device,
    n_quantized: &AtomicUsize,
    guard: crate::QuantizeOntoGuard,
) -> Result<Arc<dyn QuantMethod>> {
    let (experts, output, _) = stack.dims3()?;
    if let Some(bias) = &bias {
        if bias.dims() != [experts, output] {
            candle_core::bail!(
                "Stacked expert bias shape {:?} does not match weight shape {:?}; expected [{experts}, {output}].",
                bias.dims(),
                stack.dims()
            );
        }
    }
    if !ty.supports_stacked_gather() {
        candle_core::bail!(
            "Cannot quantize stacked expert weights to {ty}: that target does not support stacked expert gather. Use a Q*K/Q*_0/Q*_1 target, AFQ, or omit ISQ."
        );
    }
    if candle_core::quantized::GgmlDType::try_from(ty).is_ok() {
        n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let w = crate::GgufMatMul::quantize_expert_stack(
            &stack,
            ty,
            imatrix.as_deref(),
            device,
            guard,
        )?;
        let bias = bias
            .map(|bias| bias.to_dtype(DType::F32)?.to_device(device))
            .transpose()?;
        return Ok(Arc::new(crate::GgufMatMul::from_qtensor(w, bias)));
    }
    let unquant = Arc::new(crate::UnquantLinear::new(
        crate::QuantMethodConfig::Unquantized(candle_nn::Linear::new(stack, bias)),
    )?) as Arc<dyn QuantMethod>;
    unquant.apply_isq(Some(ty), device.clone(), n_quantized, imatrix, guard)
}

/// Quantize every tracked module on an executor sized for `pool_ty`.
pub fn requantize_tracked(
    modules: &[TrackedModule],
    pool_ty: IsqType,
    ty_for: impl Fn(&TrackedModule) -> IsqType,
    imatrix_for: &dyn Fn(&str) -> Option<Vec<f32>>,
    consumer: IsqConsumer,
    extra_host_reserve_bytes: usize,
    report: Option<crate::QuantizationReport>,
) -> Result<RequantizeHandles> {
    let config = crate::IsqExecutorConfig::new(Some(pool_ty))
        .with_external_reserved_host_bytes(extra_host_reserve_bytes);
    let (executor, _) = crate::create_isq_executor(config);
    let guard = crate::QuantizeOntoGuard::new();
    let mut receivers = Vec::with_capacity(modules.len());
    for module in modules {
        let layer = module.ct.resolve()?;
        let ty = ty_for(module);
        let imatrix = if ty.supports_imatrix() {
            imatrix_for(&module.key)
        } else {
            if imatrix_for(&module.key).is_some() {
                crate::log::once_log_warn(format!(
                    "{ty} does not consume imatrix weights; quantizing without them."
                ));
            }
            None
        };
        let device = layer.dtype_and_device().1;
        let mut guard = guard
            .clone()
            .with_module_key(module.key.clone())
            .with_requested(ty.to_string())
            .with_consumer(consumer);
        if let Some(report) = &report {
            guard = guard.with_report(report.clone());
        }
        let request = IsqRequest {
            ty: Some(ty),
            device: device.clone(),
            has_imatrix: imatrix.is_some(),
            capture: crate::IsqCaptureMode::Immediate,
            consumer,
            module_key: module.key.clone(),
        };
        let plan = layer.plan_isq(&request)?;
        let rx = executor.submit(plan, consumer, move || {
            layer
                .clone()
                .apply_isq(Some(ty), device, &AtomicUsize::new(0), imatrix, guard)
        });
        receivers.push(rx);
    }
    Ok(RequantizeHandles {
        _executor: executor,
        receivers,
    })
}

/// Return the fallback dtype for the given dtype.
fn get_fallback(dtype: GgmlDType) -> QuantizationBehavior {
    // The normal `Q` quants are a bit more lenient than the `K` quants.
    // => Try to fallback to a similar `Q` quant.
    // If that's not possible, skip this tensor.
    match dtype {
        GgmlDType::Q2K => QuantizationBehavior::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q3K => QuantizationBehavior::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q4K => QuantizationBehavior::Quantize(GgmlDType::Q4_1),
        GgmlDType::Q5K => QuantizationBehavior::Quantize(GgmlDType::Q5_0),
        GgmlDType::Q6K => QuantizationBehavior::Quantize(GgmlDType::Q5_1),
        GgmlDType::Q8K => QuantizationBehavior::Quantize(GgmlDType::Q8_1),
        _ => QuantizationBehavior::Skip,
    }
}

/// Check if the tensor can be quantized with the given dtype.
fn can_quantize(tensor: &Tensor, dtype: GgmlDType) -> bool {
    let dims = tensor.shape().dims();
    // The tensor must not be empty and the last dimension must be a multiple of the block size.
    !dims.is_empty() && dims[dims.len() - 1].is_multiple_of(dtype.block_size())
}

/// Groups whose largest magnitude is below this are quantized as exact zeros
/// (llama.cpp's `GROUP_MAX_EPS`).
pub(crate) const GROUP_MAX_EPS: f32 = 1e-15;

/// Sub-block (scale group) length of an importance-weighted K-quant, or `None`
/// when `dtype` has no imatrix quantizer.
fn imatrix_group_len(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q6K => Some(16),
        GgmlDType::Q4K | GgmlDType::Q5K => Some(32),
        _ => None,
    }
}

/// `QTensor::quantize_imatrix` with llama.cpp's near-zero group guard.
///
/// The importance-weighted K-quant search squares and weights each value. In a
/// group whose values are all tiny (e.g. ~1e-37 "dead" rows) those products
/// underflow to 0 and the fitted scale becomes 0/0 = NaN, which poisons the
/// whole super-block. Groups whose max |x| is below [`GROUP_MAX_EPS`] are
/// flushed to exact zeros first; the quantizer already maps an all-zero group
/// to zero scales and zero quants. Tensors without such groups are passed
/// through untouched, so their output is bit-identical to the unguarded path.
pub fn quantize_imatrix_guarded(
    src: &Tensor,
    imatrix_weights: &[f32],
    dtype: GgmlDType,
) -> Result<QTensor> {
    let n_per_row = src.dim(D::Minus1)?;
    let Some(group) = imatrix_group_len(dtype).filter(|g| n_per_row.is_multiple_of(*g)) else {
        return QTensor::quantize_imatrix(src, imatrix_weights, dtype);
    };
    let mut xs = src.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut flushed = 0usize;
    for g in xs.chunks_exact_mut(group) {
        let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax > 0.0 && amax < GROUP_MAX_EPS {
            g.fill(0.0);
            flushed += 1;
        }
    }
    if flushed == 0 {
        return QTensor::quantize_imatrix(src, imatrix_weights, dtype);
    }
    tracing::debug!(
        "imatrix {dtype:?}: flushed {flushed} near-zero groups (max |x| < {GROUP_MAX_EPS:e}) to zero"
    );
    let guarded = Tensor::from_vec(xs, src.shape(), src.device())?;
    QTensor::quantize_imatrix(&guarded, imatrix_weights, dtype)
}

/// Check if we should quantize the tensor and if so, with which dtype.
pub(crate) fn get_quantization_behaviour(
    tensor: &Tensor,
    dtype: GgmlDType,
) -> QuantizationBehavior {
    if dtype == GgmlDType::F32 {
        return QuantizationBehavior::Skip;
    }

    if can_quantize(tensor, dtype) {
        return QuantizationBehavior::Quantize(dtype);
    }
    let fallback = get_fallback(dtype);
    match fallback {
        QuantizationBehavior::Skip => fallback,
        QuantizationBehavior::Quantize(new_dtype) => get_quantization_behaviour(tensor, new_dtype),
    }
}

pub(crate) fn warn_skip_quantization(
    guard: Option<&crate::QuantizeOntoGuard>,
    module_key: Option<&str>,
    quant: Option<&str>,
    shape: &[usize],
    reason: &str,
) {
    if let Some(report) = guard.and_then(|guard| guard.report()) {
        report.record_skip(
            module_key.unwrap_or("<unknown>"),
            guard
                .and_then(|guard| guard.requested())
                .map(ToString::to_string)
                .or_else(|| quant.map(ToString::to_string)),
            shape.to_vec(),
            reason,
        );
        return;
    }

    let quant = quant.map(|quant| format!("{quant} ")).unwrap_or_default();
    match module_key {
        Some(module_key) => crate::log::once_log_warn(format!(
            "Skipping {quant} quantization of `{module_key}` with tensor shape {shape:?}: {reason}."
        )),
        None => crate::log::once_log_warn(format!(
            "Skipping {quant} quantization of tensor with shape {shape:?}: {reason}."
        )),
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq {
    ($tensor:expr, $device:expr, $dtype:expr, $n_quantized:expr, $guard:expr) => {{
        let quantization_behaviour =
            $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
        let dtype = match quantization_behaviour {
            $crate::utils::isq::QuantizationBehavior::Skip => {
                let shape = $tensor.dims().to_vec();
                let quant = format!("{:?}", $dtype);
                $crate::utils::isq::warn_skip_quantization(
                    Some(&$guard),
                    $guard.module_key(),
                    Some(&quant),
                    &shape,
                    "tensor is not quantizable",
                );
                GgmlDType::F32
            }
            $crate::utils::isq::QuantizationBehavior::Quantize(dtype) => {
                $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                dtype
            }
        };

        // Quantize from a CPU copy: byte extraction from a device-resident QTensor races
        // in-flight device work, and quantization is CPU-bound anyway.
        let cpu_src = $tensor.to_device(&candle_core::Device::Cpu)?;
        let initial = candle_core::quantized::QTensor::quantize(&cpu_src, dtype)?;
        let data = initial.data()?;

        let _acquired_quantize_guard = $guard.acquire(&$device);
        let qstorage = candle_core::quantized::QStorage::from_data(data, &$device, dtype)?;

        Arc::new(candle_core::quantized::QTensor::new(
            qstorage,
            $tensor.shape(),
        )?)
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq_imatrix {
    ($tensor:expr, $imatrix:expr, $device:expr, $dtype:expr, $n_quantized:expr, $guard:expr) => {{
        let quantization_behaviour =
            $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
        let dtype = match quantization_behaviour {
            $crate::utils::isq::QuantizationBehavior::Skip => {
                let shape = $tensor.dims().to_vec();
                let quant = format!("{:?}", $dtype);
                $crate::utils::isq::warn_skip_quantization(
                    Some(&$guard),
                    $guard.module_key(),
                    Some(&quant),
                    &shape,
                    "tensor is not quantizable",
                );
                GgmlDType::F32
            }
            $crate::utils::isq::QuantizationBehavior::Quantize(dtype) => {
                $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                dtype
            }
        };

        // Quantize from a CPU copy: byte extraction from a device-resident QTensor races
        // in-flight device work, and quantization is CPU-bound anyway.
        let cpu_src = $tensor.to_device(&candle_core::Device::Cpu)?;
        // Fallback dtypes (legacy Q, F32) have no imatrix quantizer; quantize plainly.
        let initial = if matches!(
            dtype,
            GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
        ) {
            $crate::utils::isq::quantize_imatrix_guarded(&cpu_src, &$imatrix, dtype)?
        } else {
            candle_core::quantized::QTensor::quantize(&cpu_src, dtype)?
        };
        let data = initial.data()?;

        let _acquired_quantize_guard = $guard.acquire(&$device);
        let qstorage = candle_core::quantized::QStorage::from_data(data, &$device, dtype)?;

        Arc::new(candle_core::quantized::QTensor::new(
            qstorage,
            $tensor.shape(),
        )?)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMATRIX_KQUANTS: [GgmlDType; 5] = [
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ];
    /// Magnitude of the "dead" rows found in the 0.8B/2B `in_proj_qkv` weights.
    const DEAD: f32 = 1.175e-37;

    fn imatrix(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 + (i % 11) as f32 * 0.37).collect()
    }

    fn normal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.731).sin() * 0.05 + (i as f32 * 0.113).cos() * 0.02)
            .collect()
    }

    fn dequant(q: &QTensor) -> Result<Vec<f32>> {
        q.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()
    }

    #[test]
    fn dead_blocks_quantize_to_zero() -> Result<()> {
        let (rows, cols) = (3, 512);
        let xs: Vec<f32> = (0..rows * cols)
            .map(|i| DEAD * (1.0 + (i % 5) as f32 * 0.25) * if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let src = Tensor::from_vec(xs, (rows, cols), &Device::Cpu)?;
        let im = imatrix(cols);
        for dtype in IMATRIX_KQUANTS {
            let q = quantize_imatrix_guarded(&src, &im, dtype)?;
            let ys = dequant(&q)?;
            assert!(
                ys.iter().all(|y| *y == 0.0),
                "{dtype:?}: dead block did not dequantize to exact zeros"
            );
        }
        // the unguarded path is what produced NaN scales for Q4K (issue #61)
        let raw = dequant(&QTensor::quantize_imatrix(&src, &im, GgmlDType::Q4K)?)?;
        assert!(
            raw.iter().any(|y| !y.is_finite()),
            "repro lost: raw Q4K stayed finite"
        );
        Ok(())
    }

    #[test]
    fn mixed_blocks_stay_finite() -> Result<()> {
        let (rows, cols) = (2, 512);
        let base = normal(rows * cols);
        // every other 32-value group of each super-block is dead (a whole scale group
        // for every K-quant: 32 for Q4K/Q5K, two 16-groups for Q2K/Q3K/Q6K)
        let xs: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(i, x)| if (i / 32) % 2 == 0 { DEAD } else { *x })
            .collect();
        let src = Tensor::from_vec(xs.clone(), (rows, cols), &Device::Cpu)?;
        let im = imatrix(cols);
        for dtype in IMATRIX_KQUANTS {
            let ys = dequant(&quantize_imatrix_guarded(&src, &im, dtype)?)?;
            assert!(
                ys.iter().all(|y| y.is_finite()),
                "{dtype:?}: non-finite output"
            );
            let (mut err, mut sig) = (0f32, 0f32);
            for (i, (x, y)) in xs.iter().zip(&ys).enumerate() {
                if (i / 32) % 2 == 0 {
                    assert!(y.abs() < 1e-30, "{dtype:?}: dead value {i} -> {y}");
                } else {
                    err += (x - y) * (x - y);
                    sig += x * x;
                }
            }
            let rel = (err / sig).sqrt();
            assert!(
                rel < 0.6,
                "{dtype:?}: live values badly quantized (rel rmse {rel})"
            );
        }
        Ok(())
    }

    #[test]
    fn normal_blocks_bit_identical() -> Result<()> {
        let (rows, cols) = (4, 768);
        let src = Tensor::from_vec(normal(rows * cols), (rows, cols), &Device::Cpu)?;
        let im = imatrix(cols);
        for dtype in IMATRIX_KQUANTS {
            let guarded = quantize_imatrix_guarded(&src, &im, dtype)?;
            let raw = QTensor::quantize_imatrix(&src, &im, dtype)?;
            assert_eq!(
                guarded.data()?,
                raw.data()?,
                "{dtype:?}: normal block changed"
            );
        }
        Ok(())
    }
}
