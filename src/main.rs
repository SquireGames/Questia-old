extern crate gfx_backend_vulkan as backend;

use gfx_hal::{format, image, Instance};

use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::command::Level;
use gfx_hal::device::Device;
use gfx_hal::queue::family::QueueFamily;

use gfx_hal::pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc};

use gfx_hal::window::{Extent2D, PresentationSurface, Surface};

use gfx_hal::pool::CommandPool;

use winit::{event_loop::EventLoop, window::WindowBuilder};

use shaderc::ShaderKind;

use std::mem::ManuallyDrop;

struct Resources<B: gfx_hal::Backend> {
    instance: B::Instance,
    surface: B::Surface,
    device: B::Device,
    render_passes: Vec<B::RenderPass>,
    pipeline_layouts: Vec<B::PipelineLayout>,
    pipelines: Vec<B::GraphicsPipeline>,
    command_pool: B::CommandPool,
    submission_complete_fence: B::Fence,
    rendering_complete_semaphore: B::Semaphore,
    vertex_buffer_memory: B::Memory,
    vertex_buffer: B::Buffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PushConstants {
    transform: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(serde::Deserialize)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

fn make_tranform(translate: [f32; 3], angle: f32, scale: f32) -> [[f32; 4]; 4] {
    let c = angle.cos() * scale;
    let s = angle.sin() * scale;
    let [dx, dy, dz] = translate;
    [
        [c, 0., s, 0.],
        [0., scale, 0., 0.],
        [-s, 0., c, 0.],
        [dx, dy, dz, 1.],
    ]
}

unsafe fn make_buffer<B: gfx_hal::Backend>(
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

struct ResourceHolder<B: gfx_hal::Backend>(ManuallyDrop<Resources<B>>);

impl<B: gfx_hal::Backend> Drop for ResourceHolder<B> {
    fn drop(&mut self) {
        unsafe {
            let Resources {
                instance,
                mut surface,
                device,
                command_pool,
                render_passes,
                pipeline_layouts,
                pipelines,
                submission_complete_fence,
                rendering_complete_semaphore,
                vertex_buffer_memory,
                vertex_buffer,
            } = ManuallyDrop::take(&mut self.0);

            device.free_memory(vertex_buffer_memory);
            device.destroy_buffer(vertex_buffer);
            device.destroy_semaphore(rendering_complete_semaphore);
            device.destroy_fence(submission_complete_fence);
            for pipeline in pipelines {
                device.destroy_graphics_pipeline(pipeline);
            }
            for pipeline_layout in pipeline_layouts {
                device.destroy_pipeline_layout(pipeline_layout);
            }
            for render_pass in render_passes {
                device.destroy_render_pass(render_pass);
            }
            device.destroy_command_pool(command_pool);
            surface.unconfigure_swapchain(&device);
            instance.destroy_surface(surface);
        }
    }
}

fn compile_shader(glsl: &str, shader_kind: ShaderKind) -> Vec<u32> {
    use std::io::Cursor;

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));
    let compiled_file = compiler
        .compile_into_spirv(glsl, shader_kind, "shader.glsl", "main", Some(&options))
        .unwrap();

    let compiled_file = compiled_file.as_binary_u8();

    let spv =
        gfx_auxil::read_spirv(Cursor::new(&compiled_file)).expect("Failed to read spirv file");

    spv
}

unsafe fn make_pipeline<B: gfx_hal::Backend>(
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
        .create_shader_module(&compile_shader(vertex_shader, ShaderKind::Vertex))
        .expect("Failed to create vertex module");

    let fragment_shader_module = device
        .create_shader_module(&compile_shader(fragment_shader, ShaderKind::Fragment))
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

fn main() {
    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("Questia Game")
        .with_inner_size(winit::dpi::LogicalSize {
            width: 500,
            height: 500,
        });
    let gfx_instance =
        backend::Instance::create("name", 1).expect("Failed to create graphics instance");

    let window = window_builder.build(&event_loop).unwrap();

    let surface = unsafe {
        gfx_instance
            .create_surface(&window)
            .expect("Failed to create surface")
    };
    let adapter = gfx_instance.enumerate_adapters().remove(0);

    let family = adapter
        .queue_families
        .iter()
        .find(|family| {
            surface.supports_queue_family(family) && family.queue_type().supports_graphics()
        })
        .unwrap();

    let mut gpu = unsafe {
        adapter
            .physical_device
            .open(&[(family, &[1.0])], gfx_hal::Features::empty())
            .unwrap()
    };

    let mut queue_group = gpu.queue_groups.pop().unwrap();
    let device = gpu.device;

    let mut command_pool = unsafe {
        device.create_command_pool(
            queue_group.family,
            gfx_hal::pool::CommandPoolCreateFlags::empty(),
        )
    }
    .expect("Can't create command pool");

    let mut command_buffer = unsafe { command_pool.allocate_one(Level::Primary) };

    let formats = surface.supported_formats(&adapter.physical_device);
    let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == format::ChannelType::Srgb)
            .map(|format| *format)
            .unwrap_or(formats[0])
    });

    let binary_mesh_data = include_bytes!("../assets/teapot_mesh.bin");
    let mesh: Vec<Vertex> =
        bincode::deserialize(binary_mesh_data).expect("Failed to deserialize mesh");

    let vertex_buffer_len = mesh.len() * std::mem::size_of::<Vertex>();
    let (vertex_buffer_memory, vertex_buffer) = unsafe {
        make_buffer::<backend::Backend>(
            &device,
            &adapter.physical_device,
            vertex_buffer_len,
            gfx_hal::buffer::Usage::VERTEX,
            gfx_hal::memory::Properties::CPU_VISIBLE,
        )
    };

    unsafe {
        let mapped_memory = device
            .map_memory(&vertex_buffer_memory, gfx_hal::memory::Segment::ALL)
            .expect("Failed to map memory");

        std::ptr::copy_nonoverlapping(mesh.as_ptr() as *const u8, mapped_memory, vertex_buffer_len);

        device
            .flush_mapped_memory_ranges(vec![(
                &vertex_buffer_memory,
                gfx_hal::memory::Segment::ALL,
            )])
            .expect("Failed to flush mapped memory");

        device.unmap_memory(&vertex_buffer_memory);
    }

    let render_pass = {
        let color_attachment = Attachment {
            format: Some(format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: image::Layout::Undefined..image::Layout::Present,
        };

        let subpass = SubpassDesc {
            colors: &[(0, image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        unsafe {
            device
                .create_render_pass(&[color_attachment], &[subpass], &[])
                .expect("Out of memory")
        }
    };

    use gfx_hal::pso::ShaderStageFlags;
    let push_constant_bytes = std::mem::size_of::<PushConstants>() as u32;
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&[], &[(ShaderStageFlags::VERTEX, 0..push_constant_bytes)])
            .expect("Out of memory")
    };

    let vertex_shader = include_str!("../shaders/simple.vert");
    let fragment_shader = include_str!("../shaders/simple.frag");

    let pipeline = unsafe {
        make_pipeline::<backend::Backend>(
            &device,
            &render_pass,
            &pipeline_layout,
            vertex_shader,
            fragment_shader,
        )
    };

    let submission_complete_fence = device.create_fence(true).expect("Out of memory");
    let rendering_complete_semaphore = device.create_semaphore().expect("Out of memory");

    let mut resource_holder: ResourceHolder<backend::Backend> =
        ResourceHolder(ManuallyDrop::new(Resources {
            instance: gfx_instance,
            surface,
            device,
            command_pool,
            render_passes: vec![render_pass],
            pipeline_layouts: vec![pipeline_layout],
            pipelines: vec![pipeline],
            submission_complete_fence,
            rendering_complete_semaphore,
            vertex_buffer_memory,
            vertex_buffer,
        }));

    let mut should_configure_swapchain = true;

    let start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit
                }
                winit::event::WindowEvent::Resized(_) => {
                    should_configure_swapchain = true;
                }
                _ => {}
            },
            winit::event::Event::MainEventsCleared => window.request_redraw(),
            winit::event::Event::RedrawRequested(_) => {
                let res: &mut Resources<_> = &mut resource_holder.0;
                let render_pass = &res.render_passes[0];
                let pipeline_layout = &res.pipeline_layouts[0];
                let pipeline = &res.pipelines[0];

                unsafe {
                    // one second timeout
                    let render_timeout_ns = 1_000_000_000;

                    res.device
                        .wait_for_fence(&res.submission_complete_fence, render_timeout_ns)
                        .expect("Out of memory or device lost");

                    res.device
                        .reset_fence(&res.submission_complete_fence)
                        .expect("Out of memory");

                    res.command_pool.reset(false);
                }

                let mut extent = Extent2D {
                    width: 500,
                    height: 500,
                };

                if should_configure_swapchain {
                    use gfx_hal::window::SwapchainConfig;

                    let caps = res.surface.capabilities(&adapter.physical_device);

                    let mut swapchain_config = SwapchainConfig::from_caps(&caps, format, extent);

                    if caps.image_count.contains(&3) {
                        swapchain_config.image_count = 3;
                    }

                    extent = swapchain_config.extent;

                    unsafe {
                        res.surface
                            .configure_swapchain(&res.device, swapchain_config)
                            .expect("Failed to configure swapchain");
                    }

                    should_configure_swapchain = false;
                }

                let surface_image = unsafe {
                    let acquire_timeout_ns = 1_000_000_000;

                    match res.surface.acquire_image(acquire_timeout_ns) {
                        Ok((image, _)) => image,
                        Err(_) => {
                            should_configure_swapchain = true;
                            return;
                        }
                    }
                };

                let framebuffer = unsafe {
                    use gfx_hal::image::Extent;
                    use std::borrow::Borrow;

                    res.device
                        .create_framebuffer(
                            render_pass,
                            vec![surface_image.borrow()],
                            Extent {
                                width: extent.width,
                                height: extent.height,
                                depth: 1,
                            },
                        )
                        .unwrap()
                };

                let viewport = {
                    use gfx_hal::pso::{Rect, Viewport};
                    Viewport {
                        rect: Rect {
                            x: 0,
                            y: 0,
                            w: extent.width as i16,
                            h: extent.height as i16,
                        },
                        depth: 0.0..1.0,
                    }
                };

                let angle = start_time.elapsed().as_secs_f32();
                let transforms = &[PushConstants {
                    transform: make_tranform([0., 0., 0.5], angle, 1.0),
                }];

                unsafe fn push_constant_bytes<T>(push_constants: &T) -> &[u32] {
                    let size_in_bytes = std::mem::size_of::<T>();
                    let size_in_u32s = size_in_bytes / std::mem::size_of::<u32>();
                    let start_ptr = push_constants as *const T as *const u32;
                    std::slice::from_raw_parts(start_ptr, size_in_u32s)
                }

                unsafe {
                    use gfx_hal::command::{
                        ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, SubpassContents,
                    };

                    command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

                    command_buffer.set_viewports(0, &[viewport.clone()]);
                    command_buffer.set_scissors(0, &[viewport.rect]);

                    command_buffer.bind_vertex_buffers(
                        0,
                        vec![(&res.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)],
                    );

                    command_buffer.begin_render_pass(
                        render_pass,
                        &framebuffer,
                        viewport.rect,
                        &[ClearValue {
                            color: ClearColor {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        }],
                        SubpassContents::Inline,
                    );
                    command_buffer.bind_graphics_pipeline(pipeline);

                    for transform in transforms {
                        command_buffer.push_graphics_constants(
                            pipeline_layout,
                            ShaderStageFlags::VERTEX,
                            0,
                            push_constant_bytes(transform),
                        );

                        let vertex_count = mesh.len() as u32;
                        command_buffer.draw(0..vertex_count, 0..1);
                    }

                    command_buffer.end_render_pass();
                    command_buffer.finish();
                }

                unsafe {
                    use gfx_hal::queue::{CommandQueue, Submission};

                    let submission = Submission {
                        command_buffers: vec![&command_buffer],
                        wait_semaphores: None,
                        signal_semaphores: vec![&res.rendering_complete_semaphore],
                    };

                    queue_group.queues[0].submit(submission, Some(&res.submission_complete_fence));

                    let result = queue_group.queues[0].present(
                        &mut res.surface,
                        surface_image,
                        Some(&res.rendering_complete_semaphore),
                    );

                    should_configure_swapchain |= result.is_err();
                    res.device.destroy_framebuffer(framebuffer);
                }
            }
            _ => {}
        }
    });
}
