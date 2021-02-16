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
// Copyright 2017 Jens Behley (jens.behley@igg.uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <stdint.h>
#if defined(NOLIVIER)
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif //NOLIVIER

#include <glow/GlFramebuffer.h>
#include <glow/GlProgram.h>
#include <glow/GlTexture.h>
#ifndef NOLIVIER
#include <glow/GlSampler.h>
#endif //NOLIVIER
#include "fastcd/mesh.h"

namespace fastcd {

/**
 * @brief      Basic projection into a framebuffer texture.
 */
class DepthProjector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief      Constructor.
   *
   * @param[in]  width       The image width
   * @param[in]  height      The image height
   * @param[in]  projection  The camera projection matrix
   */
  DepthProjector(uint32_t width, uint32_t height,
                 const Eigen::Matrix4f& projection);
#ifndef NOLIVIER
  DepthProjector(uint32_t width, uint32_t height,
                 const Eigen::Matrix4f& projection, int mode);
#endif //NOLIVIER

  /**
   * @brief      Renders the given mesh from the given view relative to the
   *             mesh.
   *
   * @param[in]  mesh  The mesh
   * @param[in]  view  The view
   */
  void render(const Mesh& mesh, const Eigen::Matrix4f& view);
#ifndef NOLIVIER
  void renderNormal(const Mesh& mesh, const Eigen::Matrix4f& view);
  unsigned int shadows(const Mesh& mesh, const Eigen::Matrix4f& view);
  void render(const Mesh& mesh, const Eigen::Matrix4f& view,
              const Eigen::Matrix4f& view1, glow::GlTexture& texture1);
  void render(const Mesh& mesh, int kernel_size,
              const Eigen::Matrix4f& view1, glow::GlTexture& texture1,
              const Eigen::Matrix4f& view2, glow::GlTexture& texture2);
  void render(const Mesh& mesh, int kernel_size,
    const Eigen::Matrix4f& view1, glow::GlTexture& texture1, unsigned int depthMap1,
    const Eigen::Matrix4f& view2, glow::GlTexture& texture2, unsigned int depthMap2);
  void render(const Mesh& mesh, int kernel_size, glow::GlTexture& texture,
    const Eigen::Matrix4f& view1, glow::GlTexture& texture1, unsigned int depthMap1,
    const Eigen::Matrix4f& view2, glow::GlTexture& texture2, unsigned int depthMap2);
  glow::GlTexture median(glow::GlTexture& input);
  void medianBlur(glow::GlTexture& input);
#endif //NOLIVIER

  /**
   * @brief      Gets the depth projection.
   *
   * @return     The depth image.
   */
  glow::GlTexture& texture();

 protected:
  /** The framebuffer used to render the mesh */
  glow::GlFramebuffer framebuffer_;

  /** The output texture */
  glow::GlTexture output_;

  /** The program used to render the mesh */
  glow::GlProgram program_;
#ifndef NOLIVIER
  /** The program used to render shadows */
  glow::GlProgram program_shadow_;
  glow::GlProgram program_median_;
  glow::GlSampler sampler_;
  int mode_;
#endif //NOLIVIER

  /** The projection matrix of the cameras*/
  Eigen::Matrix4f projection_;
};

}  // namespace fastcd
