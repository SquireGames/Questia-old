pub mod resources;
pub mod shader;

use gfx_hal::{adapter::PhysicalDevice, device::Device};
use shaderc::ShaderKind;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PushConstants {
    pub transform: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(serde::Deserialize)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

pub unsafe fn make_buffer<B: gfx_hal::Backend>(
    device: &B::Device,
    physical_device: &B::PhysicalDevice,
    buffer_len: usize,
    usage: gfx_hal::buffer::Usage,
    properties: gfx_hal::memory::Properties,
) -> (B::Memory, B::Buffer) {
    let mut buffer = device
        .create_buffer(buffer_len as u64, usage)
        .expect("Failed to create buffer");

    let requirements = device.get_buffer_requirements(&buffer);

    let memory_types = physical_device.memory_properties().memory_types;

    let memory_type = memory_types
        .iter()
        .enumerate()
        .find(|(id, mem_type)| {
            let type_supported = requirements.type_mask & (1_u32 << id) != 0;
            type_supported && mem_type.properties.contains(properties)
        })
        .map(|(id, _type)| gfx_hal::MemoryTypeId(id))
        .expect("No compatible memory type");

    let buffer_memory = device
        .allocate_memory(memory_type, requirements.size)
        .expect("Failed to allocate memory");

    device
        .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
        .expect("tast");

    (buffer_memory, buffer)
}

pub unsafe fn make_pipeline<B: gfx_hal::Backend>(
    device: &B::Device,
    render_pass: &B::RenderPass,
    pipeline_layout: &B::PipelineLayout,
    vertex_shader: &str,
    fragment_shader: &str,
) -> B::GraphicsPipeline {
    use gfx_hal::format::Format;
    use gfx_hal::pass::Subpass;
    use gfx_hal::pso::{
        AttributeDesc, BlendState, ColorBlendDesc, ColorMask, Element, EntryPoint, Face,
        GraphicsPipelineDesc, InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc, Rasterizer,
        Specialization, VertexBufferDesc, VertexInputRate,
    };

    let vertex_shader_module = device
        .create_shader_module(&shader::compile_shader(vertex_shader, ShaderKind::Vertex))
        .expect("Failed to create vertex module");

    let fragment_shader_module = device
        .create_shader_module(&shader::compile_shader(
            fragment_shader,
            ShaderKind::Fragment,
        ))
        .expect("Failed to create fragment shader");

    let (vs_entry, fs_entry) = (
        EntryPoint {
            entry: "main",
            module: &vertex_shader_module,
            specialization: Specialization::default(),
        },
        EntryPoint {
            entry: "main",
            module: &fragment_shader_module,
            specialization: Specialization::default(),
        },
    );

    let vertex_buffers = vec![VertexBufferDesc {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        rate: VertexInputRate::Vertex,
    }];

    let attributes: Vec<AttributeDesc> = vec![
        AttributeDesc {
            location: 0,
            binding: 0,
            element: Element {
                format: Format::Rgb32Sfloat,
                offset: 0,
            },
        },
        AttributeDesc {
            location: 1,
            binding: 0,
            element: Element {
                format: Format::Rgb32Sfloat,
                offset: 12,
            },
        },
    ];
    let input_assembler = InputAssemblerDesc {
        primitive: Primitive::TriangleList,
        with_adjacency: false,
        restart_index: None,
    };

    let mut pipeline_desc = GraphicsPipelineDesc::new(
        PrimitiveAssemblerDesc::Vertex {
            buffers: &vertex_buffers,
            attributes: &attributes,
            input_assembler: input_assembler,
            vertex: vs_entry,
            tessellation: None,
            geometry: None,
        },
        Rasterizer {
            cull_face: Face::BACK,
            ..Rasterizer::FILL
        },
        Some(fs_entry),
        pipeline_layout,
        Subpass {
            index: 0,
            main_pass: render_pass,
        },
    );

    pipeline_desc.blender.targets.push(ColorBlendDesc {
        mask: ColorMask::ALL,
        blend: Some(BlendState::ALPHA),
    });

    let pipeline = device
        .create_graphics_pipeline(&pipeline_desc, None)
        .expect("Failed to create graphics pipeline");

    device.destroy_shader_module(vertex_shader_module);
    device.destroy_shader_module(fragment_shader_module);

    pipeline
}
