extern crate gfx_backend_vulkan as backend;

mod graphics;
mod math;

use gfx_hal::{
    adapter::PhysicalDevice,
    command::Level,
    device::Device,
    format, image,
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc},
    pool::CommandPool,
    queue::{family::QueueFamily, CommandQueue, Submission},
    window::{Extent2D, PresentationSurface, Surface},
    Instance,
};
use std::mem::ManuallyDrop;
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
        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == format::ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });
        format
    };

    let binary_mesh_data = include_bytes!("../assets/teapot_mesh.bin");
    let mesh: Vec<graphics::Vertex> =
        bincode::deserialize(binary_mesh_data).expect("Failed to deserialize mesh");

    let vertex_buffer_len = mesh.len() * std::mem::size_of::<graphics::Vertex>();
    let (vertex_buffer_memory, vertex_buffer) = unsafe {
        graphics::make_buffer::<backend::Backend>(
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
            format: Some(surface_color_format),
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
    let push_constant_bytes = std::mem::size_of::<graphics::PushConstants>() as u32;
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&[], &[(ShaderStageFlags::VERTEX, 0..push_constant_bytes)])
            .expect("Out of memory")
    };

    let vertex_shader = include_str!("../shaders/simple.vert");
    let fragment_shader = include_str!("../shaders/simple.frag");

    let pipeline = unsafe {
        graphics::make_pipeline::<backend::Backend>(
            &device,
            &render_pass,
            &pipeline_layout,
            vertex_shader,
            fragment_shader,
        )
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

                if should_configure_swapchain {
                    use gfx_hal::window::SwapchainConfig;

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
                    use std::borrow::Borrow;

                    res.device
                        .create_framebuffer(
                            render_pass,
                            vec![surface_image.borrow()],
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
