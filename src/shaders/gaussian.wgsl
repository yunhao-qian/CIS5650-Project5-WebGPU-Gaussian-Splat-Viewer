struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // TODO: information passed from vertex shader to fragment shader
    @location(0) center_pixel: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) conic: vec3<f32>,
};

struct Splat {
    // TODO: information defined in preprocess compute shader
    center: u32,
    color: array<u32, 2>,
    conic_radius: array<u32, 2>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> sort_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // TODO: reconstruct 2D quad based on information from splat, pass
    let splat = splats[sort_indices[instance_index]];

    let center = unpack2x16float(splat.center); // in NDC space
    let color = vec4<f32>(
        unpack2x16float(splat.color[0]),
        unpack2x16float(splat.color[1])
    );
    let conic_0 = unpack2x16float(splat.conic_radius[0]);
    let conic_1_radius = unpack2x16float(splat.conic_radius[1]);
    let conic = vec3<f32>(conic_0, conic_1_radius[0]);
    let radius = conic_1_radius[1];
    let quad_size = 2.0 * radius / camera.viewport; // in NDC space

    let offsets = array<vec2<f32>, 6>(
        vec2<f32>(-1, -1),
        vec2<f32>(1, -1),
        vec2<f32>(-1, 1),
        vec2<f32>(-1, 1),
        vec2<f32>(1, -1),
        vec2<f32>(1, 1)
    );
    let offset = offsets[vertex_index] * quad_size; // in NDC space

    var out: VertexOutput;
    out.position = vec4<f32>(center + offset, 0.0, 1.0);
    out.center_pixel = (center * vec2<f32>(0.5, -0.5) + 0.5) * camera.viewport; // in pixel space
    out.color = color;
    out.conic = conic;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L330
    let d = in.position.xy - in.center_pixel; // in pixel space
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.y * d.y * d.y) - in.conic.y * d.x * d.y;
    if (power > 0.0) {
        discard;
    } 
    let alpha = min(0.99, in.color.a * exp(power));
    return vec4<f32>(in.color.rgb, alpha);
}
