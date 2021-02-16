#version 330 core

layout (location = 0) in vec4 position_in;
layout (location = 1) in vec4 normal_in;

uniform mat4 mvp;          // model-view-projection matrix. (applied p*v*m)
uniform mat4 view_mat;
uniform mat4 normal_mat;

out vec3 v_position;
out vec3 v_normal;

void main(void) {
  gl_Position = mvp * position_in;  // this must be in NDC.
  v_position = (  view_mat * position_in).xyz;
  v_normal   = (normal_mat *   normal_in).xyz;
}
