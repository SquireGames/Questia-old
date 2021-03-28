#version 450

layout(push_constant) uniform PushConstants {
    mat4 transform;
} push_constants;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_xy;
layout(location = 0) out vec2 texture_uv;

void main() {
    texture_uv = texture_xy;
    gl_Position = push_constants.transform * vec4(position, 1.0);
}
