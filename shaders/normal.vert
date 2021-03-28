#version 450

layout(push_constant) uniform PushConstants {
    mat4 transform;
} push_constants;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec4 vertex_color;

void main() {
    vertex_color = vec4(abs(normal), 1.0);
    gl_Position = push_constants.transform * vec4(position, 1.0);
}
