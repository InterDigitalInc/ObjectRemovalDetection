/*
 * Copyright 2021 InterDigital
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * License is avaiable on the root under license.txt and you may obtain a copy
 * of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef NOLIVIER
#include "fastcd/depth_projector.h"
#include "fastcd/mesh.h"

namespace fastcd {

DepthProjector::DepthProjector(uint32_t width, uint32_t height,
                               const Eigen::Matrix4f& projection, int mode)
    : framebuffer_(width, height),
      output_(width, height, glow::TextureFormat::RGBA_FLOAT),
      projection_(projection) {
  framebuffer_.attach(glow::FramebufferAttachment::COLOR0, output_);
  glow::GlRenderbuffer rbo(width, height,
                           glow::RenderbufferFormat::DEPTH_STENCIL);
  framebuffer_.attach(glow::FramebufferAttachment::DEPTH_STENCIL,
                      rbo);  // framebuffers needs a depth/stencil buffer.
  switch (mode) {
    case 0:
    default:
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/project_mesh.vert"));
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/project_mesh.frag"));
    break;
    case 1:
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/normal_mesh.vert"));
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/normal_mesh.frag"));
    break;
    case 2:
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/warp.vert"));
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/warp.frag"));
    break;
    case 3:
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/delta.vert"));
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/delta.frag"));
    break;
    case 4:
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/delta.vert"));
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/min_delta.frag"));
    program_median_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/test.vert"));
    program_median_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/test.frag"));
    program_median_.link();
    program_median_.setUniform(glow::GlUniform<int32_t>("myTextureSampler", 0));
    break;
    case 5:
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/delta.vert"));
    program_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/min_delta_blur.frag"));
    program_median_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/test.vert"));
    program_median_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/test.frag"));
    program_median_.link();
    program_median_.setUniform(glow::GlUniform<int32_t>("myTextureSampler", 0));
    break;
  }
  program_.link();
  if (mode >= 2 ) {
    program_shadow_.attach(glow::GlShader::fromCache(
        glow::ShaderType::VERTEX_SHADER, "fastcd/shaders/shadow.vert"));
    program_shadow_.attach(glow::GlShader::fromCache(
        glow::ShaderType::FRAGMENT_SHADER, "fastcd/shaders/shadow.frag"));
    program_shadow_.link();
    program_.setUniform(glow::GlUniform<int32_t>("texColor1", 0));
    program_.setUniform(glow::GlUniform<int32_t>("shadowMap1", 1));
    sampler_.setMagnifyingOperation(glow::TexMagOp::NEAREST);
    sampler_.setMinifyingOperation(glow::TexMinOp::NEAREST);
    sampler_.setWrapOperation(glow::TexWrapOp::CLAMP_TO_BORDER, glow::TexWrapOp::CLAMP_TO_BORDER);
  }
  if (mode >= 3 ) {
    program_.setUniform(glow::GlUniform<int32_t>("texColor2", 2));
    program_.setUniform(glow::GlUniform<int32_t>("shadowMap2", 3));
    program_.setUniform(glow::GlUniform<glow::vec2>("pixel",
      glow::vec2(1.0/framebuffer_.width(),
                 1.0/framebuffer_.height())));
  }
  if (mode >= 4) {
    program_.setUniform(glow::GlUniform<int32_t>("texOutput", 4));
    program_.setUniform(glow::GlUniform<GLint>("initial", 1));
  }
  mode_ = mode;
}

void DepthProjector::renderNormal(const Mesh& mesh, const Eigen::Matrix4f& view) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);

  framebuffer_.bind();
  glEnable(GL_DEPTH_TEST);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  //glDepthMask(GL_TRUE);
  //glDepthFunc(GL_LEQUAL);
  //glDepthRange(1.0f, 0.0f);
  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());

  program_.bind();
  program_.setUniform(
      glow::GlUniform<Eigen::Matrix4f>("mvp", projection_ * view));
  if (mode_ == 1) {
    program_.setUniform(glow::GlUniform<Eigen::Matrix4f>("view_mat", view));
    program_.setUniform(glow::GlUniform<Eigen::Matrix4f>("normal_mat",
      view.inverse().transpose()));
  }

  mesh.draw(program_);
  program_.release();

  framebuffer_.release();
  //glDepthMask(GL_TRUE);
  //glDepthFunc(GL_LEQUAL);
  //glDepthRange(0.0f, 1.0f);
  glViewport(ov[0], ov[1], ov[2], ov[3]);
}

unsigned int DepthProjector::shadows(const Mesh& mesh,
                                     const Eigen::Matrix4f& view) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);
  glEnable(GL_DEPTH_TEST);

  unsigned int depthMapFBO;
  glGenFramebuffers(1, &depthMapFBO);

  unsigned int depthMap;
  glGenTextures(1, &depthMap);
  glBindTexture(GL_TEXTURE_2D, depthMap);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
               framebuffer_.width(), framebuffer_.height(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  program_shadow_.bind();
  program_shadow_.setUniform(
      glow::GlUniform<Eigen::Matrix4f>("mvp", projection_ * view));
  program_shadow_.setUniform(glow::GlUniform<GLfloat>("bias", 1.0/(double)framebuffer_.width()));

  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());
  glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
  glClear(GL_DEPTH_BUFFER_BIT);
  glCullFace(GL_FRONT);
  mesh.draw(program_shadow_);
  glCullFace(GL_BACK);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glDeleteFramebuffers(1, &depthMapFBO);

  glViewport(ov[0], ov[1], ov[2], ov[3]);

  return depthMap;
}

void DepthProjector::render(const Mesh& mesh, const Eigen::Matrix4f& view,
                            const Eigen::Matrix4f& view1, glow::GlTexture& texture1) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);
  glEnable(GL_DEPTH_TEST);

  // RENDER SHADOW
  unsigned int depthMapFBO;
  glGenFramebuffers(1, &depthMapFBO);
  const unsigned int SHADOW_WIDTH  = 2*framebuffer_.width(),
                     SHADOW_HEIGHT = 2*framebuffer_.height();//1024

  unsigned int depthMap;
  glGenTextures(1, &depthMap);
  glBindTexture(GL_TEXTURE_2D, depthMap);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
               SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  program_shadow_.bind();
  program_shadow_.setUniform(
      glow::GlUniform<Eigen::Matrix4f>("mvp", projection_ * view1));
  program_shadow_.setUniform(glow::GlUniform<GLfloat>("bias", 1.0/(double)SHADOW_WIDTH));

  glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
  glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
  glClear(GL_DEPTH_BUFFER_BIT);
  glCullFace(GL_FRONT);
  mesh.draw(program_shadow_);
  glCullFace(GL_BACK);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // RENDER IMAGE
  framebuffer_.bind();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());
  program_.bind();
  program_.setUniform(
      glow::GlUniform<Eigen::Matrix4f>("mvp", projection_ * view));
  program_.setUniform(
      glow::GlUniform<Eigen::Matrix4f>("mvp1", projection_ * view1));

  glActiveTexture(GL_TEXTURE0);
  texture1.bind();
  sampler_.bind(0);  // ensure nearest neighbor interp.
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depthMap);
  sampler_.bind(1);  // ensure nearest neighbor interp.
  mesh.draw(program_);
  glBindTexture(GL_TEXTURE_2D, 0);
  sampler_.release(0);
  sampler_.release(1);
  program_.release();

  framebuffer_.release();

  glViewport(ov[0], ov[1], ov[2], ov[3]);
}

void DepthProjector::render(const Mesh& mesh, int kernel_size,
                            const Eigen::Matrix4f& view1, glow::GlTexture& texture1,
                            const Eigen::Matrix4f& view2, glow::GlTexture& texture2) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);
  glEnable(GL_DEPTH_TEST);

  unsigned int depthMap1 = shadows(mesh, view1);
  unsigned int depthMap2 = shadows(mesh, view2);

  // RENDER IMAGE
  framebuffer_.bind();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());
  program_.bind();
  program_.setUniform(
    glow::GlUniform<Eigen::Matrix4f>("mvp1", projection_ * view1));
  program_.setUniform(
    glow::GlUniform<Eigen::Matrix4f>("mvp2", projection_ * view2));
  program_.setUniform(glow::GlUniform<GLfloat>("iwidth", 1.0/framebuffer_.width()));
  program_.setUniform(glow::GlUniform<GLfloat>("iheight", 1.0/framebuffer_.height()));
  program_.setUniform(glow::GlUniform<GLint>("kernel_upper", kernel_size - kernel_size/2));
  program_.setUniform(glow::GlUniform<GLint>("kernel_lower", -kernel_size/2));

  glActiveTexture(GL_TEXTURE0);
  texture1.bind();
  sampler_.bind(0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depthMap1);
  sampler_.bind(1);

  glActiveTexture(GL_TEXTURE2);
  texture2.bind();
  sampler_.bind(2);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, depthMap2);
  sampler_.bind(3);

  mesh.draw(program_);
  glBindTexture(GL_TEXTURE_2D, 0);
  sampler_.release(0);
  sampler_.release(1);
  sampler_.release(2);
  sampler_.release(3);
  program_.release();

  framebuffer_.release();

  glDeleteTextures(1, &depthMap1);
  glDeleteTextures(1, &depthMap2);

  glViewport(ov[0], ov[1], ov[2], ov[3]);
}

void DepthProjector::render(const Mesh& mesh, int kernel_size,
  const Eigen::Matrix4f& view1, glow::GlTexture& texture1, unsigned int depthMap1,
  const Eigen::Matrix4f& view2, glow::GlTexture& texture2, unsigned int depthMap2) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);
  glEnable(GL_DEPTH_TEST);

  framebuffer_.bind();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());
  program_.bind();
  program_.setUniform(
    glow::GlUniform<Eigen::Matrix4f>("mvp1", projection_ * view1));
  program_.setUniform(
    glow::GlUniform<Eigen::Matrix4f>("mvp2", projection_ * view2));
  program_.setUniform(glow::GlUniform<GLint>("kernel_upper", kernel_size - kernel_size/2));
  program_.setUniform(glow::GlUniform<GLint>("kernel_lower", -kernel_size/2));
  program_.setUniform(glow::GlUniform<GLint>("initial", 1));

  glActiveTexture(GL_TEXTURE0);
  texture1.bind();
  sampler_.bind(0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depthMap1);
  sampler_.bind(1);

  glActiveTexture(GL_TEXTURE2);
  texture2.bind();
  sampler_.bind(2);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, depthMap2);
  sampler_.bind(3);

  mesh.draw(program_);
  glBindTexture(GL_TEXTURE_2D, 0);
  sampler_.release(0);
  sampler_.release(1);
  sampler_.release(2);
  sampler_.release(3);
  program_.release();

  framebuffer_.release();

  glViewport(ov[0], ov[1], ov[2], ov[3]);
}

void DepthProjector::render(const Mesh& mesh, int kernel_size, glow::GlTexture& texture,
  const Eigen::Matrix4f& view1, glow::GlTexture& texture1, unsigned int depthMap1,
  const Eigen::Matrix4f& view2, glow::GlTexture& texture2, unsigned int depthMap2) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);
  glEnable(GL_DEPTH_TEST);

  framebuffer_.bind();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());
  program_.bind();
  program_.setUniform(
    glow::GlUniform<Eigen::Matrix4f>("mvp1", projection_ * view1));
  program_.setUniform(
    glow::GlUniform<Eigen::Matrix4f>("mvp2", projection_ * view2));
  program_.setUniform(glow::GlUniform<GLint>("kernel_upper", kernel_size - kernel_size/2));
  program_.setUniform(glow::GlUniform<GLint>("kernel_lower", -kernel_size/2));

  glActiveTexture(GL_TEXTURE0);
  texture1.bind();
  sampler_.bind(0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depthMap1);
  sampler_.bind(1);

  glActiveTexture(GL_TEXTURE2);
  texture2.bind();
  sampler_.bind(2);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, depthMap2);
  sampler_.bind(3);

  glActiveTexture(GL_TEXTURE4);
  texture.bind();
  sampler_.bind(4);

  mesh.draw(program_);
  glBindTexture(GL_TEXTURE_2D, 0);
  sampler_.release(0);
  sampler_.release(1);
  sampler_.release(2);
  sampler_.release(3);
  sampler_.release(4);
  program_.setUniform(glow::GlUniform<GLint>("initial", 0));
  program_.release();

  framebuffer_.release();

  glViewport(ov[0], ov[1], ov[2], ov[3]);
}

glow::GlTexture DepthProjector::median(glow::GlTexture& input) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);
  glEnable(GL_DEPTH_TEST);

  unsigned int depthMapFBO;
  glGenFramebuffers(1, &depthMapFBO);

  GLint coord2d_location, vertexUv_location, pixD_location;
  GLuint ebo, program, texture, vbo, vao;

  glow::GlTexture output(framebuffer_.width(), framebuffer_.height(), glow::TextureFormat::RGB_FLOAT);

  // Shader setup.
  program = program_median_.id();//common_get_shader_program(vertex_shader_source, fragment_shader_source);
  coord2d_location = glGetAttribLocation(program, "coord2d");
  vertexUv_location = glGetAttribLocation(program, "vertexUv");
  pixD_location = glGetUniformLocation(program, "pixD");

  const GLfloat vertices[] = {
  //  xy            uv
      -1.0,  1.0,   1.0, 0.0,
       1.0,  1.0,   1.0, 1.0,
       1.0, -1.0,   0.0, 1.0,
      -1.0, -1.0,   0.0, 0.0,
  };
  const GLuint indices[] = {
      0, 1, 2,
      0, 2, 3,
  };

  // Create vbo.
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Create ebo.
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // vao.
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(coord2d_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(vertices[0]), (GLvoid*)0);
  glEnableVertexAttribArray(coord2d_location);
  glVertexAttribPointer(vertexUv_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(vertices[0])));
  glEnableVertexAttribArray(vertexUv_location);
  glBindVertexArray(0);

  // Texture buffer.
  texture = output.id();
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Constant state.
  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());
  glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
  glActiveTexture(GL_TEXTURE0);

  // Apply.
  glClear(GL_COLOR_BUFFER_BIT);
  input.bind();
  glUseProgram(program);
  sampler_.bind(0);
  glUniform2f(pixD_location, 1.0 / framebuffer_.width(), 1.0 / framebuffer_.height());
  glBindVertexArray(vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Cleanup.
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteFramebuffers(1, &depthMapFBO);
  glViewport(ov[0], ov[1], ov[2], ov[3]);

  return output;
}

void DepthProjector::medianBlur(glow::GlTexture& input) {
  GLint ov[4];
  glGetIntegerv(GL_VIEWPORT, ov);
  glEnable(GL_DEPTH_TEST);

  GLint coord2d_location, vertexUv_location, pixD_location;
  GLuint ebo, program, texture, vbo, vao;

  // Shader setup.
  program = program_median_.id();//common_get_shader_program(vertex_shader_source, fragment_shader_source);
  coord2d_location = glGetAttribLocation(program, "coord2d");
  vertexUv_location = glGetAttribLocation(program, "vertexUv");
  pixD_location = glGetUniformLocation(program, "pixD");

  const GLfloat vertices[] = {
  //  xy            uv
      -1.0,  1.0,   1.0, 0.0,
       1.0,  1.0,   1.0, 1.0,
       1.0, -1.0,   0.0, 1.0,
      -1.0, -1.0,   0.0, 0.0,
  };
  const GLuint indices[] = {
      0, 1, 2,
      0, 2, 3,
  };

  // Create vbo.
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Create ebo.
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // vao.
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(coord2d_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(vertices[0]), (GLvoid*)0);
  glEnableVertexAttribArray(coord2d_location);
  glVertexAttribPointer(vertexUv_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(vertices[0])));
  glEnableVertexAttribArray(vertexUv_location);
  glBindVertexArray(0);

  // Texture buffer.
  framebuffer_.bind();

  // Constant state.
  glViewport(0, 0, framebuffer_.width(), framebuffer_.height());
  glActiveTexture(GL_TEXTURE0);

  // Apply.
  glClear(GL_COLOR_BUFFER_BIT);
  input.bind();
  glUseProgram(program);
  sampler_.bind(0);
  glUniform2f(pixD_location, 1.0 / framebuffer_.width(), 1.0 / framebuffer_.height());
  glBindVertexArray(vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
  framebuffer_.release();

  // Cleanup.
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glDeleteVertexArrays(1, &vao);
  glViewport(ov[0], ov[1], ov[2], ov[3]);

  return;
}

}  // namespace fastcd
#endif //NOLIVIER
