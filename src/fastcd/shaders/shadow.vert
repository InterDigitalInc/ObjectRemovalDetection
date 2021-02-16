#version 330 core

layout (location = 0) in vec4 position_in;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * position_in;
}
