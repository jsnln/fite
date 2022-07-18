#version 460 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 outColor;
// out vec3 fragment_normal;
// out vec3 fragment_pos;

uniform mat4 projection;

void main()
{
    // gl_Position = projection * view * model_loc * vec4(position, 1.0f);
    gl_Position = projection * vec4(position, 1.0f);// * vec4(1.0f, 1.0f, -1.0f, 1.0f);

    // fragment_normal = normal_mat * fnormal;
    // fragment_pos = vec3(model_loc * vec4(position, 1.0f));
    outColor = color;
    // outColor = vec3(1.0f, 1.0f, 1.0f);
}