#version 330 core
in vec2 coord2d;
in vec2 vertexUv;
out vec2 fragmentUv;

void main() {
    gl_Position = vec4(coord2d, 0, 1);
    fragmentUv = vertexUv;
};
