extern crate gfx_backend_vulkan as back;

use gfx_hal::queue::family::QueueFamily;
use gfx_hal::window::Surface;

use gfx_hal::Instance;
use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::device::Device;

use winit::{
    event_loop::{EventLoop},
    window::WindowBuilder,
};


fn main() {
    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new().with_title("Questia Game");
    let gfx_instance =
        back::Instance::create("name", 1).expect("Failed to create graphics instance");

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

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Wait;

        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit
                }
                _ => {}
            },
            _ => {}
        }
    })
}
