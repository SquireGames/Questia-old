extern crate gfx_backend_vulkan as backend;

mod graphics;
mod math;

use gfx::PipelineState;
use gfx_hal::{
    adapter::PhysicalDevice,
    buffer, command,
    command::Level,
    device::Device,
    format as hal_format,
    format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
    image as hal_img, memory as hal_mem,
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc},
    pool::CommandPool,
    prelude::*,
    pso::{self, PipelineStage},
    queue::{family::QueueFamily, CommandQueue},
    window::{Extent2D, PresentationSurface, Surface},
    Instance,
};

use std::{borrow::Borrow, io::Cursor, iter, mem::ManuallyDrop};

use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event_loop::EventLoop,
    window::WindowBuilder,
};

fn main() {
    const DEFAULT_WINDOW_SIZE: [u32; 2] = [1000, 1000];

    // winit creation
    // winit has a concept of physical vs logical window size for different dpi screens
    let event_loop = EventLoop::new();
    let (physical_window_size, logical_window_size) = {
        let dpi = event_loop.primary_monitor().unwrap().scale_factor();
        let logical_size: LogicalSize<u32> = DEFAULT_WINDOW_SIZE.into();
        let physical_size: PhysicalSize<u32> = logical_size.to_physical(dpi);
        (physical_size, logical_size)
    };
    let window_builder = WindowBuilder::new()
        .with_title("Questia Game")
        .with_inner_size(logical_window_size);
    let window = window_builder.build(&event_loop).unwrap();
    let mut window_render_size = Extent2D {
        width: physical_window_size.width,
        height: physical_window_size.height,
    };

    // gfx instance creation
    let (gfx_instance, surface, adapter) = {
        let gfx_instance =
            backend::Instance::create("name", 1).expect("Failed to create graphics instance");

        let surface = unsafe {
            gfx_instance
                .create_surface(&window)
                .expect("Failed to create surface")
        };
        let adapter = gfx_instance.enumerate_adapters().remove(0);

        (gfx_instance, surface, adapter)
    };

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

    let surface_color_format = {
        let formats = surface.supported_formats(&adapter.physical_device);
        let format = formats.map_or(hal_format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });
        format
    };

    // create descriptor set
    let set_layout = {
        unsafe {
            device.create_descriptor_set_layout(
                vec![
                    pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: pso::DescriptorType::Image {
                            ty: pso::ImageDescriptorType::Sampled {
                                with_sampler: false,
                            },
                        },
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        binding: 1,
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                ]
                .into_iter(),
                std::iter::empty(),
            )
        }
        .expect("Failed to make the descriptor set layout")
    };
    let mut descriptor_pool = {
        unsafe {
            device.create_descriptor_pool(
                1,
                vec![
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Image {
                            ty: pso::ImageDescriptorType::Sampled {
                                with_sampler: false,
                            },
                        },
                        count: 1,
                    },
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                    },
                ]
                .into_iter(),
                pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("Failed to make descriptor pool")
    };
    let mut descriptor_set = unsafe { descriptor_pool.allocate_one(&set_layout) }.unwrap();

    // vertex buffer allocation
    let binary_mesh_data = include_bytes!("../assets/teapot_mesh.bin");
    let mesh: Vec<graphics::Vertex> =
        bincode::deserialize(binary_mesh_data).expect("Failed to deserialize mesh");

    let mut new_mesh_format = vec![];
    new_mesh_format.reserve(mesh.len());

    let mut bounds_it = 0;
    let bounds = vec![
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ];

    for v in mesh {
        new_mesh_format.push(graphics::VertexNew {
            pos: v.pos,
            xy: bounds[bounds_it],
        });
        bounds_it = (bounds_it + 1) % 6;
    }

    let vertex_buffer_len = new_mesh_format.len() * std::mem::size_of::<graphics::VertexNew>();
    let (mut vertex_buffer_memory, vertex_buffer) = unsafe {
        let mut buffer = device
            .create_buffer(vertex_buffer_len as u64, gfx_hal::buffer::Usage::VERTEX)
            .expect("Failed to create buffer");

        let requirements = device.get_buffer_requirements(&buffer);

        let memory_types = adapter.physical_device.memory_properties().memory_types;

        let memory_type = memory_types
            .iter()
            .enumerate()
            .find(|(id, mem_type)| {
                let type_supported = requirements.type_mask & (1_u32 << id) != 0;
                type_supported
                    && mem_type
                        .properties
                        .contains(gfx_hal::memory::Properties::CPU_VISIBLE)
            })
            .map(|(id, _type)| gfx_hal::MemoryTypeId(id))
            .expect("No compatible memory type");

        let buffer_memory = device
            .allocate_memory(memory_type, requirements.size)
            .expect("Failed to allocate memory");

        device
            .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
            .expect("Failed to bind buffer memory");

        (buffer_memory, buffer)
    };

    unsafe {
        let mapped_memory = device
            .map_memory(&mut vertex_buffer_memory, gfx_hal::memory::Segment::ALL)
            .expect("Failed to map memory");

        std::ptr::copy_nonoverlapping(
            new_mesh_format.as_ptr() as *const u8,
            mapped_memory,
            vertex_buffer_len,
        );

        device
            .flush_mapped_memory_ranges(iter::once((
                &vertex_buffer_memory,
                gfx_hal::memory::Segment::ALL,
            )))
            .expect("Failed to flush mapped memory");

        device.unmap_memory(&mut vertex_buffer_memory);
    }

    // load texture
    let image_bytes = include_bytes!("../data/Chicken_SW.png");
    let image = image::load(Cursor::new(&image_bytes[..]), image::ImageFormat::Png)
        .unwrap()
        .to_rgba8();

    let sampler = {
        unsafe {
            device.create_sampler(&hal_img::SamplerDesc::new(
                hal_img::Filter::Linear,
                hal_img::WrapMode::Clamp,
            ))
        }
        .expect("Failed to create sampler")
    };

    let (width, height) = image.dimensions();
    let kind = hal_img::Kind::D2(width as hal_img::Size, height as hal_img::Size, 1, 1);

    // for buffer upload
    let limits = adapter.physical_device.limits();
    let non_coherent_alignment = limits.non_coherent_atom_size as u64;
    // multiple of 2, subtract 1 to get lower bits
    let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
    // TODO presumably 4 bytes per pixel
    let image_stride = 4usize;
    // calculate row pitch length, but use mask to ensure valid low bit alignment
    // inner mask addition is carryover from mask-removed lower bits
    let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
    let upload_size = (height * row_pitch) as u64;
    // round up to match alignment
    let padded_upload_size = ((upload_size + non_coherent_alignment - 1) / non_coherent_alignment)
        * non_coherent_alignment;

    let mut image_upload_buffer = unsafe {
        device
            .create_buffer(padded_upload_size, buffer::Usage::TRANSFER_SRC)
            .unwrap()
    };
    let image_mem_reqs = unsafe { device.get_buffer_requirements(&image_upload_buffer) };

    let memory_types = adapter.physical_device.memory_properties().memory_types;

    let upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            image_mem_reqs.type_mask & (1 << id) != 0
                && mem_type
                    .properties
                    .contains(hal_mem::Properties::CPU_VISIBLE)
        })
        .unwrap()
        .into();

    // copy image onto staging buffer
    // TODO add to resources and free later
    let image_upload_memory = unsafe {
        let mut memory = device
            .allocate_memory(upload_type, image_mem_reqs.size)
            .unwrap();
        device
            .bind_buffer_memory(&memory, 0, &mut image_upload_buffer)
            .unwrap();
        let mapping = device
            .map_memory(&mut memory, hal_mem::Segment::ALL)
            .unwrap();
        for y in 0..height as usize {
            let row = &(*image)
                [y * (width as usize) * image_stride..(y + 1) * (width as usize) * image_stride];
            std::ptr::copy_nonoverlapping(
                row.as_ptr(),
                mapping.offset(y as isize * row_pitch as isize),
                width as usize * image_stride,
            );
        }
        device
            .flush_mapped_memory_ranges(iter::once((&memory, hal_mem::Segment::ALL)))
            .unwrap();
        device.unmap_memory(&mut memory);
        memory
    };

    let mut image_material = ManuallyDrop::new(
        unsafe {
            device.create_image(
                kind,
                1,
                ColorFormat::SELF,
                hal_img::Tiling::Optimal,
                hal_img::Usage::TRANSFER_DST | hal_img::Usage::SAMPLED,
                // TODO look into sparsely bound
                hal_img::ViewCapabilities::empty(),
            )
        }
        .unwrap(),
    );

    let image_requirements = unsafe { device.get_image_requirements(&image_material) };
    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            image_requirements.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(hal_mem::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();
    let image_memory = ManuallyDrop::new(
        unsafe { device.allocate_memory(device_type, image_requirements.size) }.unwrap(),
    );
    unsafe { device.bind_image_memory(&image_memory, 0, &mut image_material) }.unwrap();

    let image_view = ManuallyDrop::new(
        unsafe {
            device.create_image_view(
                &image_material,
                hal_img::ViewKind::D2,
                ColorFormat::SELF,
                Swizzle::NO,
                hal_img::SubresourceRange {
                    aspects: hal_format::Aspects::COLOR,
                    ..Default::default()
                },
            )
        }
        .unwrap(),
    );

    unsafe {
        device.write_descriptor_set(pso::DescriptorSetWrite {
            set: &mut descriptor_set,
            binding: 0,
            array_offset: 0,
            descriptors: vec![
                pso::Descriptor::Image(&*image_view, hal_img::Layout::ShaderReadOnlyOptimal),
                pso::Descriptor::Sampler(&sampler),
            ]
            .into_iter(),
        });
    }

    // copy buffer to texture
    let mut copy_fence = device.create_fence(false).unwrap();
    unsafe {
        let mut command_buffer = command_pool.allocate_one(command::Level::Primary);
        command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

        let image_barrier = hal_mem::Barrier::Image {
            states: (hal_img::Access::empty(), hal_img::Layout::Undefined)
                ..(
                    hal_img::Access::TRANSFER_WRITE,
                    hal_img::Layout::TransferDstOptimal,
                ),
            target: &*image_material,
            families: None,
            range: hal_img::SubresourceRange {
                aspects: hal_format::Aspects::COLOR,
                ..Default::default()
            },
        };
        command_buffer.pipeline_barrier(
            pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::TRANSFER,
            hal_mem::Dependencies::empty(),
            iter::once(image_barrier),
        );
        command_buffer.copy_buffer_to_image(
            &image_upload_buffer,
            &image_material,
            hal_img::Layout::TransferDstOptimal,
            iter::once(command::BufferImageCopy {
                buffer_offset: 0,
                buffer_width: row_pitch / (image_stride as u32),
                buffer_height: height as u32,
                image_layers: hal_img::SubresourceLayers {
                    aspects: hal_format::Aspects::COLOR,
                    level: 0,
                    layers: 0..1,
                },
                image_offset: hal_img::Offset { x: 0, y: 0, z: 0 },
                image_extent: hal_img::Extent {
                    width,
                    height,
                    depth: 1,
                },
            }),
        );

        let image_barrier = hal_mem::Barrier::Image {
            states: (
                hal_img::Access::TRANSFER_WRITE,
                hal_img::Layout::TransferDstOptimal,
            )
                ..(
                    hal_img::Access::SHADER_READ,
                    hal_img::Layout::ShaderReadOnlyOptimal,
                ),
            target: &*image_material,
            families: None,
            range: hal_img::SubresourceRange {
                aspects: hal_format::Aspects::COLOR,
                ..Default::default()
            },
        };
        command_buffer.pipeline_barrier(
            pso::PipelineStage::TRANSFER..pso::PipelineStage::FRAGMENT_SHADER,
            hal_mem::Dependencies::empty(),
            iter::once(image_barrier),
        );
        command_buffer.finish();

        queue_group.queues[0].submit(
            iter::once(&command_buffer),
            iter::empty(),
            iter::empty(),
            Some(&mut copy_fence),
        );

        device.wait_for_fence(&copy_fence, !0).unwrap();
    };

    unsafe {
        device.destroy_fence(copy_fence);
    }

    let render_pass = {
        let color_attachment = Attachment {
            format: Some(surface_color_format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: hal_img::Layout::Undefined..hal_img::Layout::Present,
        };

        let subpass = SubpassDesc {
            colors: &[(0, hal_img::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        unsafe {
            device
                .create_render_pass(
                    iter::once(color_attachment),
                    iter::once(subpass),
                    iter::empty(),
                )
                .expect("Out of memory")
        }
    };

    use gfx_hal::pso::ShaderStageFlags;
    let push_constant_bytes = std::mem::size_of::<graphics::PushConstants>() as u32;
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(
                iter::once(&*&set_layout),
                iter::once((ShaderStageFlags::VERTEX, 0..push_constant_bytes)),
            )
            .expect("Out of memory")
    };

    let vertex_shader = include_str!("../shaders/texture.vert");
    let fragment_shader = include_str!("../shaders/texture.frag");

    let pipeline = unsafe {
        use gfx_hal::format::Format;
        use gfx_hal::pass::Subpass;
        use gfx_hal::pso::{
            AttributeDesc, BlendState, ColorBlendDesc, ColorMask, Element, EntryPoint, Face,
            GraphicsPipelineDesc, InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc,
            Rasterizer, Specialization, VertexBufferDesc, VertexInputRate,
        };
        use graphics::VertexNew;
        use shaderc::ShaderKind;

        let vertex_shader_module = device
            .create_shader_module(&graphics::compile_shader(vertex_shader, ShaderKind::Vertex))
            .expect("Failed to create vertex module");

        let fragment_shader_module = device
            .create_shader_module(&graphics::compile_shader(
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
            stride: std::mem::size_of::<VertexNew>() as u32,
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
            &pipeline_layout,
            Subpass {
                index: 0,
                main_pass: &render_pass,
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
    };

    let submission_complete_fence = device.create_fence(true).expect("Out of memory");
    let rendering_complete_semaphore = device.create_semaphore().expect("Out of memory");

    let mut resource_holder: graphics::resources::ResourceHolder<backend::Backend> =
        graphics::resources::ResourceHolder(ManuallyDrop::new(graphics::resources::Resources {
            instance: gfx_instance,
            surface,
            device,
            command_pool,
            render_passes: vec![render_pass],
            pipeline_layouts: vec![pipeline_layout],

            descriptor_pool: descriptor_pool,
            descriptor_set: Some(descriptor_set),
            descriptor_set_layout: set_layout,

            pipelines: vec![pipeline],
            submission_complete_fence,
            rendering_complete_semaphore,
            vertex_buffer_memory,
            vertex_buffer,

            sampler,
            image_upload_memory,
        }));

    let mut should_configure_swapchain = true;

    let start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit
                }
                winit::event::WindowEvent::Resized(new_render_size) => {
                    window_render_size = Extent2D {
                        width: new_render_size.width,
                        height: new_render_size.height,
                    };

                    should_configure_swapchain = true;
                }
                _ => {}
            },
            winit::event::Event::MainEventsCleared => window.request_redraw(),
            winit::event::Event::RedrawRequested(_) => {
                let res: &mut graphics::resources::Resources<_> = &mut resource_holder.0;
                let render_pass = &res.render_passes[0];
                let pipeline_layout = &res.pipeline_layouts[0];
                let pipeline = &res.pipelines[0];

                // TODO see if this copied code is necessary
                use gfx_hal::window::SwapchainConfig;
                let caps = res.surface.capabilities(&adapter.physical_device);
                let swapchain_config =
                    SwapchainConfig::from_caps(&caps, surface_color_format, window_render_size);

                unsafe {
                    // one second timeout
                    let render_timeout_ns = 1_000_000_000;

                    res.device
                        .wait_for_fence(&res.submission_complete_fence, render_timeout_ns)
                        .expect("Out of memory or device lost");

                    res.device
                        .reset_fence(&mut res.submission_complete_fence)
                        .expect("Out of memory");

                    res.command_pool.reset(false);
                }

                if should_configure_swapchain {
                    let caps = res.surface.capabilities(&adapter.physical_device);

                    let mut swapchain_config =
                        SwapchainConfig::from_caps(&caps, surface_color_format, window_render_size);

                    if caps.image_count.contains(&3) {
                        swapchain_config.image_count = 3;
                    }

                    window_render_size = swapchain_config.extent;

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

                    res.device
                        .create_framebuffer(
                            &render_pass,
                            iter::once(swapchain_config.framebuffer_attachment()),
                            Extent {
                                width: window_render_size.width,
                                height: window_render_size.height,
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
                            w: window_render_size.width as i16,
                            h: window_render_size.height as i16,
                        },
                        depth: 0.0..1.0,
                    }
                };

                let angle = start_time.elapsed().as_secs_f32();
                let transforms = &[graphics::PushConstants {
                    transform: math::graphics::make_tranform([0., 0., 0.5], angle, 1.0),
                }];

                unsafe fn push_constant_bytes<T>(push_constants: &T) -> &[u32] {
                    let size_in_bytes = std::mem::size_of::<T>();
                    let size_in_u32s = size_in_bytes / std::mem::size_of::<u32>();
                    let start_ptr = push_constants as *const T as *const u32;
                    std::slice::from_raw_parts(start_ptr, size_in_u32s)
                }

                unsafe {
                    use gfx_hal::command::{
                        ClearColor, ClearValue, CommandBufferFlags, RenderAttachmentInfo,
                        SubpassContents,
                    };

                    command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

                    command_buffer.set_viewports(0, iter::once(viewport.clone()));
                    command_buffer.set_scissors(0, iter::once(viewport.rect));
                    command_buffer.bind_graphics_pipeline(pipeline);

                    command_buffer.bind_vertex_buffers(
                        0,
                        iter::once((&res.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)),
                    );

                    command_buffer.bind_graphics_descriptor_sets(
                        pipeline_layout,
                        0,
                        res.descriptor_set.as_ref().into_iter(),
                        iter::empty(),
                    );

                    command_buffer.begin_render_pass(
                        &render_pass,
                        &framebuffer,
                        viewport.rect,
                        iter::once(RenderAttachmentInfo {
                            image_view: surface_image.borrow(),
                            clear_value: ClearValue {
                                color: ClearColor {
                                    float32: [0.0, 0.0, 0.0, 0.0],
                                },
                            },
                        }),
                        SubpassContents::Inline,
                    );

                    for transform in transforms {
                        command_buffer.push_graphics_constants(
                            pipeline_layout,
                            ShaderStageFlags::VERTEX,
                            0,
                            push_constant_bytes(transform),
                        );

                        let vertex_count = new_mesh_format.len() as u32;
                        command_buffer.draw(0..vertex_count, 0..1);
                    }

                    command_buffer.end_render_pass();
                    command_buffer.finish();
                }

                unsafe {
                    queue_group.queues[0].submit(
                        iter::once(&command_buffer),
                        iter::empty(),
                        iter::once(&res.rendering_complete_semaphore),
                        Some(&mut res.submission_complete_fence),
                    );

                    let result = queue_group.queues[0].present(
                        &mut res.surface,
                        surface_image,
                        Some(&mut res.rendering_complete_semaphore),
                    );

                    should_configure_swapchain |= result.is_err();
                    res.device.destroy_framebuffer(framebuffer);
                }
            }
            _ => {}
        }
    });
}
