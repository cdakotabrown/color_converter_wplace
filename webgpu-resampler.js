// WebGPU resampler (Chrome-only)
(() => {
  const MODES = ["nearest", "bilinear", "mitchell", "lanczos3"];
  let initialized = false;
  let initPromise = null;
  let device = null;
  let queue = null;
  let samplerNearest = null;
  let samplerLinear = null;
  let pipelines = {};

  function shaderHeader() {
    return `
struct Uniforms {
  srcSize: vec2<f32>,
  dstSize: vec2<f32>,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> uni: Uniforms;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VSOut {
  var out: VSOut;
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
  );
  let p = positions[idx];
  out.pos = vec4<f32>(p, 0.0, 1.0);
  return out;
}
`;
  }

  function shaderSampleBasic() {
    return `
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let scale = uni.srcSize / uni.dstSize;
  let srcCoord = pos.xy * scale - vec2<f32>(0.5, 0.5);
  let uv = (srcCoord + vec2<f32>(0.5, 0.5)) / uni.srcSize;
  return textureSampleLevel(srcTex, srcSampler, uv, 0.0);
}
`;
  }

  function shaderSampleMitchell() {
    return `
fn mitchell(x: f32) -> f32 {
  let B = 1.0 / 3.0;
  let C = 1.0 / 3.0;
  let ax = abs(x);
  if (ax < 1.0) {
    return ((12.0 - 9.0 * B - 6.0 * C) * ax * ax * ax +
      (-18.0 + 12.0 * B + 6.0 * C) * ax * ax +
      (6.0 - 2.0 * B)) / 6.0;
  }
  if (ax < 2.0) {
    return ((-B - 6.0 * C) * ax * ax * ax +
      (6.0 * B + 30.0 * C) * ax * ax +
      (-12.0 * B - 48.0 * C) * ax +
      (8.0 * B + 24.0 * C)) / 6.0;
  }
  return 0.0;
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let scale = uni.srcSize / uni.dstSize;
  let srcCoord = pos.xy * scale - vec2<f32>(0.5, 0.5);
  let base = floor(srcCoord);
  let frac = srcCoord - base;
  var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var wsum: f32 = 0.0;
  for (var j: i32 = -1; j <= 2; j = j + 1) {
    let wy = mitchell(f32(j) - frac.y);
    for (var i: i32 = -1; i <= 2; i = i + 1) {
      let wx = mitchell(f32(i) - frac.x);
      let w = wx * wy;
      let samplePos = base + vec2<f32>(f32(i), f32(j));
      let clamped = clamp(samplePos, vec2<f32>(0.0, 0.0), uni.srcSize - vec2<f32>(1.0, 1.0));
      let uv = (clamped + vec2<f32>(0.5, 0.5)) / uni.srcSize;
      sum = sum + textureSampleLevel(srcTex, srcSampler, uv, 0.0) * w;
      wsum = wsum + w;
    }
  }
  if (wsum > 0.0) {
    sum = sum / wsum;
  }
  return sum;
}
`;
  }

  function shaderSampleLanczos() {
    return `
fn sinc(x: f32) -> f32 {
  if (abs(x) < 0.000001) {
    return 1.0;
  }
  let pix = 3.14159265 * x;
  return sin(pix) / pix;
}

fn lanczos(x: f32, a: f32) -> f32 {
  let ax = abs(x);
  if (ax >= a) {
    return 0.0;
  }
  return sinc(x) * sinc(x / a);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let scale = uni.srcSize / uni.dstSize;
  let srcCoord = pos.xy * scale - vec2<f32>(0.5, 0.5);
  let base = floor(srcCoord);
  let frac = srcCoord - base;
  var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var wsum: f32 = 0.0;
  let a = 3.0;
  for (var j: i32 = -2; j <= 3; j = j + 1) {
    let wy = lanczos(f32(j) - frac.y, a);
    for (var i: i32 = -2; i <= 3; i = i + 1) {
      let wx = lanczos(f32(i) - frac.x, a);
      let w = wx * wy;
      let samplePos = base + vec2<f32>(f32(i), f32(j));
      let clamped = clamp(samplePos, vec2<f32>(0.0, 0.0), uni.srcSize - vec2<f32>(1.0, 1.0));
      let uv = (clamped + vec2<f32>(0.5, 0.5)) / uni.srcSize;
      sum = sum + textureSampleLevel(srcTex, srcSampler, uv, 0.0) * w;
      wsum = wsum + w;
    }
  }
  if (wsum > 0.0) {
    sum = sum / wsum;
  }
  return sum;
}
`;
  }

  function createPipeline(code) {
    const module = device.createShaderModule({ code });
    return device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format: "rgba8unorm" }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  async function init() {
    if (initialized) return true;
    if (initPromise) return initPromise;
    initPromise = (async () => {
      if (!navigator.gpu) return false;
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      device = await adapter.requestDevice();
      queue = device.queue;
      samplerNearest = device.createSampler({
        minFilter: "nearest",
        magFilter: "nearest",
      });
      samplerLinear = device.createSampler({
        minFilter: "linear",
        magFilter: "linear",
      });

      pipelines = {
        nearest: createPipeline(shaderHeader() + shaderSampleBasic()),
        bilinear: createPipeline(shaderHeader() + shaderSampleBasic()),
        mitchell: createPipeline(shaderHeader() + shaderSampleMitchell()),
        lanczos3: createPipeline(shaderHeader() + shaderSampleLanczos()),
      };

      initialized = true;
      return true;
    })();
    return initPromise;
  }

  function getSampler(mode) {
    return mode === "bilinear" ? samplerLinear : samplerNearest;
  }

  function getPipeline(mode) {
    return pipelines[mode] || pipelines.mitchell;
  }

  function compactBuffer(mapped, width, height, bytesPerRow) {
    const out = new Uint8ClampedArray(width * height * 4);
    let offset = 0;
    for (let y = 0; y < height; y += 1) {
      const rowStart = y * bytesPerRow;
      out.set(mapped.subarray(rowStart, rowStart + width * 4), offset);
      offset += width * 4;
    }
    return out;
  }

  async function scaleImageToCanvas(image, targetCanvas, width, height, mode) {
    if (!MODES.includes(mode)) return false;
    if (width <= 0 || height <= 0) return false;
    const ready = await init();
    if (!ready || !device) return false;

    const bitmap =
      image instanceof ImageBitmap ? image : await createImageBitmap(image);
    const srcWidth = bitmap.width;
    const srcHeight = bitmap.height;

    const srcTex = device.createTexture({
      size: { width: srcWidth, height: srcHeight },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: srcTex },
      { width: srcWidth, height: srcHeight },
    );
    if (!(image instanceof ImageBitmap)) bitmap.close?.();

    const dstTex = device.createTexture({
      size: { width, height },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });

    const uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(
      uniformBuffer,
      0,
      new Float32Array([srcWidth, srcHeight, width, height]).buffer,
    );

    const pipeline = getPipeline(mode);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: getSampler(mode) },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: dstTex.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();

    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const bufferSize = bytesPerRow * height;
    const readBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: dstTex },
      { buffer: readBuffer, bytesPerRow },
      { width, height },
    );

    queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(readBuffer.getMappedRange());
    const compact = compactBuffer(mapped, width, height, bytesPerRow);
    readBuffer.unmap();

    targetCanvas.width = width;
    targetCanvas.height = height;
    const ctx = targetCanvas.getContext("2d");
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(compact);
    ctx.putImageData(imgData, 0, 0);
    return true;
  }

  window.webgpuResampler = {
    isSupported() {
      return !!navigator.gpu;
    },
    init,
    scaleImageToCanvas,
  };
})();
