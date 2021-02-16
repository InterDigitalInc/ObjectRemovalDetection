#version 330

in vec3 v_position;
in vec3 v_normal;
out vec4 pos_out;

void main(void) {
  vec3 n = normalize(v_normal);
  vec3 viewDir = -normalize(v_position.xyz);
  pos_out = vec4(vec3(dot(n, viewDir)),1.0);
}
