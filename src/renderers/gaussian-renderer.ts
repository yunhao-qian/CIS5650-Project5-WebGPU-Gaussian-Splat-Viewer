import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (multiplier: number) => void,
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const render_settings_buffer = createBuffer(
    device,
    'render settings buffer',
    8,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg]), // gaussian_multiplier, sh_degree
  );

  const splats_buffer = createBuffer(
    device,
    'splats buffer',
    pc.num_points * 5 * 4, // 1 + 2 + 2 u32's per splat
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );

  const indirect_draw_buffer = createBuffer(
    device,
    'indirect draw buffer',
    4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([6, pc.num_points, 0, 0])
  );

  // Bind group layouts
  const camera_bind_group_layout = device.createBindGroupLayout({
    label: 'camera bind group layout',
    entries: [
      { // camera
        binding: 0,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      { // render_settings
        binding: 1,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  });
  const preprocess_gaussian_bind_group_layout = device.createBindGroupLayout({
    label: 'gaussian bind group layout',
    entries: [
      { // gaussians
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      { // splats
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      { // sh_coeffs
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
    ],
  });
  const sort_bind_group_layout = device.createBindGroupLayout({
    label: 'sort bind group layout',
    entries: [
      { // sort_infos
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      { // sort_depths
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      { // sort_indices
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      { // sort_dispatch
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });
  const render_gaussian_bind_group_layout = device.createBindGroupLayout({
    label: 'render gaussian bind group layout',
    entries: [
      { // splats
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'read-only-storage' },
      },
      {
        // sort indices
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
    ],
  });

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: device.createPipelineLayout({
      label: 'preprocess pipeline layout',
      bindGroupLayouts: [
        camera_bind_group_layout,
        preprocess_gaussian_bind_group_layout,
        sort_bind_group_layout,
      ],
    }),
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'camera bind group',
    layout: camera_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const preprocess_gaussian_bind_group = device.createBindGroup({
    label: 'gaussian bind group',
    layout: preprocess_gaussian_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splats_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: sort_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_pipeline = device.createRenderPipeline({
    label: 'render gaussian pipeline',
    layout: device.createPipelineLayout({
      label: 'render gaussian pipeline layout',
      bindGroupLayouts: [
        camera_bind_group_layout,
        render_gaussian_bind_group_layout,
      ],
    }),
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'vs_main',
      buffers: [],
    },
    fragment: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const render_gaussian_bind_group = device.createBindGroup({
    label: 'render gaussian bind group',
    layout: render_gaussian_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: splats_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    let pass = encoder.beginComputePass({ label: 'preprocess gaussians' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, preprocess_gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    const workgroupCount = Math.ceil(pc.num_points / C.histogram_wg_size);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'render gaussians',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, render_gaussian_bind_group);

    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // Set sort_info.keys_size to 0
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      // Set sort_dispatch.dispatch_x to 0
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
      preprocess(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 0, 4 * 4);
      render(encoder, texture_view);
    },
    camera_buffer,
    set_gaussian_multiplier: (multiplier: number) => {
      device.queue.writeBuffer(render_settings_buffer, 0, new Float32Array([multiplier, pc.sh_deg]));
    },
  };
}
