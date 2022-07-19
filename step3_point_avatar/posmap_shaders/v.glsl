#version 460 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 outColor;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(position, 1.0f);
    outColor = color;
}