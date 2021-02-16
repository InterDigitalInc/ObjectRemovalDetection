#version 330 core

uniform float bias;

void main() {
    gl_FragDepth = gl_FragCoord.z + bias;
}
