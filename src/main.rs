extern crate gfx_backend_vulkan as backend;

use gfx_hal::queue::family::QueueFamily;
use gfx_hal::window::Surface;
use gfx_hal::window::PresentationSurface;

use gfx_hal::Instance;
use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::device::Device;
use gfx_hal::format;
use gfx_hal::image;
use gfx_hal::command::Level;
use gfx_hal::pass::{
    Attachment,
    AttachmentOps,
    AttachmentLoadOp,
    AttachmentStoreOp,
    SubpassDesc,
};
use gfx_hal::window::Extent2D;

use winit::{
    event_loop::{EventLoop},
    window::WindowBuilder,
};

use gfx_hal::pool::{
    CommandPool,
    CommandPoolCreateFlags,
};

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
            } = ManuallyDrop::take(&mut self.0);

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
    use std::io::{Cursor, Read};

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));
    let mut compiled_file = compiler.compile_into_spirv(
        glsl, shader_kind, "shader.glsl", "main", Some(&options)).unwrap();

    let mut compiled_file = compiled_file.as_binary_u8();

    let spiv = gfx_hal::pso::read_spirv(Cursor::new(&compiled_file)).expect("Invalid spirv read");
    spiv
}

unsafe fn make_pipeline<B: gfx_hal::Backend>(
    device: &B::Device,
    render_pass: &B::RenderPass,
    pipeline_layout: &B::PipelineLayout,
    vertex_shader: &str,
    fragment_shader: &str,
) -> B::GraphicsPipeline {
    use gfx_hal::pso::{
        BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face, GraphicsPipelineDesc,
        GraphicsShaderSet, Primitive, Rasterizer, Specialization
    };
    use gfx_hal::pass::Subpass;

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

    let shader_entries = GraphicsShaderSet {
        vertex: vs_entry,
        hull: None,
        domain: None,
        geometry: None,
        fragment: Some(fs_entry),   
    };

    let mut pipeline_desc = GraphicsPipelineDesc::new(
        shader_entries,
        Primitive::TriangleList,
        Rasterizer {
            cull_face: Face::BACK,
            ..Rasterizer::FILL
        },
        pipeline_layout,
        Subpass {
            index: 0,
            main_pass: render_pass,
        },
    );

    pipeline_desc.blender.targets.push(ColorBlendDesc {
        mask: ColorMask::ALL,
        blend: Some(BlendState::ALPHA)
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
    let window_builder = WindowBuilder::new().with_title("Questia Game");
    let gfx_instance =
        backend::Instance::create("name", 1).expect("Failed to create graphics instance");

    let window = window_builder.build(&event_loop).unwrap();

    let mut surface = unsafe {
        gfx_instance
            .create_surface(&window)
            .expect("Failed to create surface")
    };
    let mut adapter = gfx_instance.enumerate_adapters().remove(0);

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
        device.create_command_pool(queue_group.family, gfx_hal::pool::CommandPoolCreateFlags::empty())
    }
    .expect("Can't create command pool");

    let mut command_buffer = unsafe {
        command_pool.allocate_one(Level::Primary)
    };

    let formats = surface.supported_formats(&adapter.physical_device);
    let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == format::ChannelType::Srgb)
            .map(|format| *format)
            .unwrap_or(formats[0])
    });

    let render_pass = {
        let color_attachment = Attachment {
          format: Some(format),
          samples: 1,
          ops: AttachmentOps::new(
              AttachmentLoadOp::Clear, 
              AttachmentStoreOp::Store,
            ),
          stencil_ops: AttachmentOps::DONT_CARE,
          layouts: image::Layout::Undefined .. image::Layout::Present,
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

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&[], &[])
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
    }));

    let mut should_configure_swapchain = true;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Wait;

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
            winit::event::Event::RedrawRequested(_) => {
                let res: &mut Resources<_> = &mut resource_holder.0;
                let render_pass = &res.render_passes[0];
                let pipeline = &res.pipelines[0];
                {
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
                    let mut extent = Extent2D{width: 100, height: 100};

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
                        use std::borrow::Borrow;
                        use gfx_hal::image::Extent;

                        res
                            .device
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

                    unsafe {
                        use gfx_hal::command::{
                            ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, SubpassContents,
                        };

                        command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

                        command_buffer.set_viewports(0, &[viewport.clone()]);
                        command_buffer.set_scissors(0, &[viewport.rect]);
                        command_buffer.begin_render_pass(
                            render_pass,
                            &framebuffer,
                            viewport.rect,
                            &[ClearValue {
                                color: ClearColor {
                                    float32: [0.0, 0.0, 0.0, 0.0],
                                }
                            }],
                            SubpassContents::Inline,
                        );
                        command_buffer.bind_graphics_pipeline(pipeline);
                        command_buffer.draw(0..3, 0..1);
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
                        let result = queue_group.queues[0].present_surface(
                            &mut res.surface,
                            surface_image,
                            Some(&res.rendering_complete_semaphore),
                        );

                        should_configure_swapchain |=  result.is_err();
                        res.device.destroy_framebuffer(framebuffer);
                    }
                }
            }
            _ => {}
        }
    })
}
