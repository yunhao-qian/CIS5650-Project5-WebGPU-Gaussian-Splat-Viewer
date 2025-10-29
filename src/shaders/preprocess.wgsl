const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

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

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    // TODO: store information for 2D splat rendering
    center: u32,
    color: array<u32, 2>,
    conic_radius: array<u32, 2>,
};

// TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_coeffs: array<u32>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

const MAX_SH_COEFFS: u32 = 16u;
const SH_CHANNELS: u32 = 3u;
const SH_HALFS_PER_GAUSSIAN: u32 = MAX_SH_COEFFS * SH_CHANNELS; // 48 half floats

fn read_sh_half(splat_idx: u32, half_offset: u32) -> f32 {
    // Each u32 in the buffer stores two f16 values packed together.
    let global_half_index = splat_idx * SH_HALFS_PER_GAUSSIAN + half_offset;
    let packed_index = global_half_index >> 1u;
    let packed = sh_coeffs[packed_index];
    let decoded = unpack2x16float(packed);
    let is_high = (global_half_index & 1u) == 1u;
    return select(decoded.x, decoded.y, is_high);
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    if (c_idx >= MAX_SH_COEFFS) {
        return vec3<f32>(0.0);
    }

    let base_half = c_idx * SH_CHANNELS;
    return vec3<f32>(
        read_sh_half(splat_idx, base_half + 0u),
        read_sh_half(splat_idx, base_half + 1u),
        read_sh_half(splat_idx, base_half + 2u)
    );
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn float_to_u32_key(f: f32) -> u32 {
    let u = bitcast<u32>(f);
    let neg = (u & 0x80000000u) != 0u;
    let mask = select(0x80000000u, 0xffffffffu, neg);
    return u ^ mask;
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    // TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];
    let pos_xy = unpack2x16float(gaussian.pos_opacity[0]);
    let pos_z_opacity = unpack2x16float(gaussian.pos_opacity[1]);
    let pos_world = vec3<f32>(pos_xy, pos_z_opacity[0]);
    let opacity = sigmoid(pos_z_opacity[1]);

    let pos_view = (camera.view * vec4<f32>(pos_world, 1.0)).xyz;
    var pos_ndc = camera.proj * vec4<f32>(pos_view, 1.0);
    pos_ndc /= pos_ndc.w;
    // View-frustum culling:
    if (
        !(abs(pos_ndc.x) <= 1.1) ||
        !(abs(pos_ndc.y) <= 1.1) ||
        !(pos_ndc.z >= -0.1 && pos_ndc.z <= 1.1)
    ) {
        return;
    }

    let rot_wx = unpack2x16float(gaussian.rot[0]);
    let rot_yz = unpack2x16float(gaussian.rot[1]);
    let rot = vec4<f32>(rot_wx[1], rot_yz, rot_wx[0]); // xyzw

    let scale_xy = exp(unpack2x16float(gaussian.scale[0]));
    let scale_z_padding = exp(unpack2x16float(gaussian.scale[1]));
    let scale = vec3<f32>(scale_xy, scale_z_padding[0]);

    // https://github.com/kwea123/gaussian_splatting_notes
    let R = mat3x3<f32>(
        1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z),
        2.0 * (rot.x * rot.y - rot.w * rot.z),
        2.0 * (rot.x * rot.z + rot.w * rot.y),
        2.0 * (rot.x * rot.y + rot.w * rot.z),
        1.0 - 2.0 * (rot.x * rot.x + rot.z * rot.z),
        2.0 * (rot.y * rot.z - rot.w * rot.x),
        2.0 * (rot.x * rot.z - rot.w * rot.y),
        2.0 * (rot.y * rot.z + rot.w * rot.x),
        1.0 - 2.0 * (rot.x * rot.x + rot.y * rot.y)
    );
    let S = render_settings.gaussian_scaling * mat3x3<f32>(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    let cov3d = R * S * S * transpose(R);
    let Vrk = mat3x3<f32>( // TBD
        cov3d[0][0], cov3d[0][1], cov3d[0][2],
        cov3d[0][1], cov3d[1][1], cov3d[1][2],
        cov3d[0][2], cov3d[1][2], cov3d[2][2]
    );

    // Viewing transformation
    let W = transpose(mat3x3<f32>(
        camera.proj[0].xyz,
        camera.proj[1].xyz,
        camera.proj[2].xyz
    ));
    // Jacobian of the affine approximation of the projective transformation
    let J = mat3x3<f32>(
        camera.focal.x / pos_view.z, 0.0, -(camera.focal.x * pos_view.x) / (pos_view.z * pos_view.z),
        0.0, camera.focal.y / pos_view.z, -(camera.focal.y * pos_view.y) / (pos_view.z * pos_view.z),
        0.0, 0.0, 0.0
    );
    let T = W * J;

    var cov2d = transpose(T) * transpose(Vrk) * T;
    // A small trick to ensure the numerical stability of the inverse
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    let cov2d_flat = vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);
    let det = cov2d_flat.x * cov2d_flat.z - cov2d_flat.y * cov2d_flat.y;
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(
        cov2d_flat.z * det_inv,
        -cov2d_flat.y * det_inv,
        cov2d_flat.x * det_inv
    );

    let mid = 0.5 * (cov2d_flat.x + cov2d_flat.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2))); // Make it at least 1

    let camera_pos = camera.view_inv[3].xyz;
    let view_dir = normalize(pos_world - camera_pos);
    let color = computeColorFromSH(view_dir, idx, u32(render_settings.sh_deg));

    let splat_idx = atomicAdd(&sort_infos.keys_size, 1u);
    splats[splat_idx].center = pack2x16float(pos_ndc.xy);
    splats[splat_idx].color = array<u32,2>(
        pack2x16float(color.rg),
        pack2x16float(vec2<f32>(color.b, opacity))
    );
    splats[splat_idx].conic_radius = array<u32,2>(
        pack2x16float(conic.xy),
        pack2x16float(vec2<f32>(conic.z, radius))
    );

    sort_indices[splat_idx] = splat_idx;
    sort_depths[splat_idx] = float_to_u32_key(-pos_view.z);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (splat_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}