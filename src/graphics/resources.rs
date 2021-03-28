use gfx_hal;

use std::mem::ManuallyDrop;

use gfx_hal::{device::Device, window::PresentationSurface, Instance};

pub struct Resources<B: gfx_hal::Backend> {
    pub instance: B::Instance,
    pub surface: B::Surface,
    pub device: B::Device,
    pub render_passes: Vec<B::RenderPass>,
    pub descriptor_pool: B::DescriptorPool,
    pub descriptor_set: Option<B::DescriptorSet>,
    pub descriptor_set_layout: B::DescriptorSetLayout,
    pub pipeline_layouts: Vec<B::PipelineLayout>,
    pub pipelines: Vec<B::GraphicsPipeline>,
    pub command_pool: B::CommandPool,
    pub submission_complete_fence: B::Fence,
    pub rendering_complete_semaphore: B::Semaphore,
    pub vertex_buffer_memory: B::Memory,
    pub vertex_buffer: B::Buffer,
    pub sampler: B::Sampler,
    pub image_upload_memory: B::Memory,
}

pub struct ResourceHolder<B: gfx_hal::Backend>(pub ManuallyDrop<Resources<B>>);

impl<B: gfx_hal::Backend> Drop for ResourceHolder<B> {
    fn drop(&mut self) {
        unsafe {
            let Resources {
                instance,
                mut surface,
                device,
                command_pool,
                render_passes,
                descriptor_pool,
                descriptor_set,
                descriptor_set_layout,
                pipeline_layouts,
                pipelines,
                submission_complete_fence,
                rendering_complete_semaphore,
                vertex_buffer_memory,
                vertex_buffer,
                sampler,
                image_upload_memory,
            } = ManuallyDrop::take(&mut self.0);

            // TODO look into
            device.wait_idle().unwrap();

            // no need to free descriptor set, as it belongs to descriptor_pool
            let _ = descriptor_set;

            device.destroy_descriptor_pool(descriptor_pool);
            device.destroy_descriptor_set_layout(descriptor_set_layout);

            device.destroy_sampler(sampler);

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
