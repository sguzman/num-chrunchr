use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of};
use pollster::block_on;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tracing::info;
use wgpu::{
    self, BackendOptions, Backends, BufferDescriptor, BufferUsages, ComputePassDescriptor,
    DeviceDescriptor, InstanceDescriptor, InstanceFlags, MapMode, MemoryBudgetThresholds,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, RequestAdapterOptions, Trace,
    util::DeviceExt,
};

const WORKGROUP_SIZE: u32 = 64;

const SHADER_SOURCE: &str = r#"
struct PrimeEntry {
    value: u32,
    remainder: u32,
};

struct Params {
    len: u32,
    prime_count: u32,
};

@group(0) @binding(0)
var<storage, read_write> primes: array<PrimeEntry>;

@group(0) @binding(1)
var<storage, read> digits: array<u32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.prime_count) {
        return;
    }
    var entry = primes[gid.x];
    let modulus = entry.value;
    var remainder = entry.remainder;
    for (var i: u32 = 0u; i < params.len; i = i + 1u) {
        remainder = ((remainder * 10u) + digits[i]) % modulus;
    }
    entry.remainder = remainder;
    primes[gid.x] = entry;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrimeEntry {
    value: u32,
    remainder: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    len: u32,
    prime_count: u32,
}

impl Params {
    fn new(len: u32, prime_count: u32) -> Self {
        Self { len, prime_count }
    }
}

pub enum BatchModEngine {
    Cpu(CpuBatchModEngine),
    Gpu(GpuBatchModEngine),
}

impl BatchModEngine {
    pub fn try_new(primes: &[u32], prefer_gpu: bool) -> Result<Self> {
        if prefer_gpu {
            let engine = GpuBatchModEngine::new(primes)
                .with_context(|| "GPU batch mod initialization failed")?;
            info!(
                prime_count = primes.len(),
                "GPU batch mod engine initialized"
            );
            Ok(Self::Gpu(engine))
        } else {
            info!(prime_count = primes.len(), "using CPU batch mod engine");
            Ok(Self::Cpu(CpuBatchModEngine::new(primes)))
        }
    }

    pub fn update(&mut self, digits: &[u32]) -> Result<()> {
        match self {
            BatchModEngine::Cpu(engine) => engine.update(digits),
            BatchModEngine::Gpu(engine) => engine.update(digits),
        }
    }

    pub fn remainders(&self) -> Result<Vec<u32>> {
        match self {
            BatchModEngine::Cpu(engine) => engine.remainders(),
            BatchModEngine::Gpu(engine) => engine.remainders(),
        }
    }
}

pub struct CpuBatchModEngine {
    remainders: Vec<u32>,
    primes: Vec<u32>,
}

impl CpuBatchModEngine {
    pub fn new(primes: &[u32]) -> Self {
        Self {
            primes: primes.to_vec(),
            remainders: vec![0; primes.len()],
        }
    }

    pub fn update(&mut self, digits: &[u32]) -> Result<()> {
        for (remainder, &prime) in self.remainders.iter_mut().zip(&self.primes) {
            for &digit in digits {
                let tmp = (*remainder as u64) * 10 + (digit as u64);
                *remainder = (tmp % prime as u64) as u32;
            }
        }
        Ok(())
    }

    pub fn remainders(&self) -> Result<Vec<u32>> {
        Ok(self.remainders.clone())
    }
}

pub struct GpuBatchModEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    primes_buffer: wgpu::Buffer,
    prime_count: u32,
}

impl GpuBatchModEngine {
    pub fn new(primes: &[u32]) -> Result<Self> {
        Self::new_with_force_fallback(primes, false)
    }

    pub fn new_with_force_fallback(primes: &[u32], force_fallback_adapter: bool) -> Result<Self> {
        let instance_desc = InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::default(),
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
            backend_options: BackendOptions::default(),
        };
        let instance = wgpu::Instance::new(&instance_desc);
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter,
        }))
        .context("request GPU adapter")?;

        let device_desc = DeviceDescriptor {
            label: Some("batch mod device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: Trace::Off,
        };
        let (device, queue) =
            block_on(adapter.request_device(&device_desc)).context("request GPU device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batch mod shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch mod bind group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<Params>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("batch mod pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch mod pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let entries: Vec<PrimeEntry> = primes
            .iter()
            .map(|&value| PrimeEntry {
                value,
                remainder: 0,
            })
            .collect();

        let primes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("primes buffer"),
            contents: bytemuck::cast_slice(&entries),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            primes_buffer,
            prime_count: primes.len() as u32,
        })
    }

    pub fn update(&mut self, digits: &[u32]) -> Result<()> {
        if digits.is_empty() {
            return Ok(());
        }

        let digits_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("digits buffer"),
                contents: bytemuck::cast_slice(digits),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params = Params::new(digits.len() as u32, self.prime_count);
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params buffer"),
                contents: bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.primes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: digits_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: Some("batch mod bind group"),
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch mod encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("batch mod compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((self.prime_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(PollType::wait_indefinitely());
        Ok(())
    }

    pub fn remainders(&self) -> Result<Vec<u32>> {
        let readback_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("primes readback buffer"),
            size: self.primes_buffer.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("primes readback encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.primes_buffer,
            0,
            &readback_buffer,
            0,
            self.primes_buffer.size(),
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let ready = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let ready_clone = ready.clone();
        let error_clone = error.clone();
        slice.map_async(MapMode::Read, move |res| {
            if let Err(err) = res {
                *error_clone.lock().unwrap() = Some(err);
            }
            ready_clone.store(true, Ordering::Release);
        });

        while !ready.load(Ordering::Acquire) {
            let _ = self.device.poll(PollType::wait_indefinitely());
        }

        if let Some(err) = error.lock().unwrap().take() {
            anyhow::bail!("GPU buffer map failed: {err:?}");
        }

        let data = slice.get_mapped_range();
        let remainders = bytemuck::cast_slice::<u8, PrimeEntry>(&data)
            .iter()
            .map(|entry| entry.remainder)
            .collect();
        drop(data);
        readback_buffer.unmap();
        Ok(remainders)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_batch_engine_updates_remainders() {
        let primes = vec![2, 3, 5];
        let mut engine = BatchModEngine::try_new(&primes, false).unwrap();
        engine.update(&[1, 2, 3]).unwrap();
        let remainders = engine.remainders().unwrap();
        assert_eq!(remainders, vec![123 % 2, 123 % 3, 123 % 5]);
    }

    #[test]
    fn gpu_batch_engine_updates_remainders() {
        let primes = vec![2, 3, 5];
        let mut engine = match GpuBatchModEngine::new_with_force_fallback(&primes, true) {
            Ok(engine) => engine,
            Err(err) => {
                eprintln!("skipping GPU batch engine test: {err}");
                return;
            }
        };
        engine.update(&[1, 2, 3]).unwrap();
        let remainders = engine.remainders().unwrap();
        assert_eq!(remainders, vec![123 % 2, 123 % 3, 123 % 5]);
    }

    #[test]
    fn batch_engine_gpu_variant_updates_remainders() {
        let primes = vec![7, 11, 13];
        let gpu_engine = match GpuBatchModEngine::new_with_force_fallback(&primes, true) {
            Ok(engine) => engine,
            Err(err) => {
                eprintln!("skipping BatchModEngine GPU variant test: {err}");
                return;
            }
        };
        let mut engine = BatchModEngine::Gpu(gpu_engine);
        engine.update(&[1, 2, 3]).unwrap();
        let remainders = engine.remainders().unwrap();
        assert_eq!(remainders, vec![123 % 7, 123 % 11, 123 % 13]);
    }
}
