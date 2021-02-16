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
// Copyright 2017 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <memory>
#include <vector>
#include "fastcd/point_covariance3d.h"
#include "fastcd/image.h"
#include "fastcd/image_sequence.h"
#include "fastcd/depth_projector.h"
#include "fastcd/mesh.h"

namespace fastcd {

/**
 * @brief      Main class of the change detector algorithm.
 */
class ChangeDetector {
 public:
  /**
   * @brief      Options for the change detector algorithm (see ChangeDetector).
   */
  struct ChangeDetectorOptions {
    /** The maximum amount of images stored in the queue. */
    int cache_size;

    /** The maximum amount of comparisons per image. */
    int max_comparisons;

    /** Chi square value for 2D confidence ellipses (95% of the variance
     * retained by default).
     */
    double chi_square2d = 5.991;

    /** Chi square value for 3D confidence ellipsoids (95% of the variance
     * retained by default).
     */
    double chi_square3d = 7.815;

    /** Width to which rescale the images */
    int rescale_width;

    /** The threshold area under which a 2D change is discarded */
    int threshold_change_area;
#ifndef NOLIVIER
    /** The threshold pixel value under which a 2D change is discarded */
    int threshold_change_value;
#endif //NOLIVIER
  };

  /**
   * @brief      Constructor.
   *
   * @param[in]  mesh     The 3D model of the environment
   * @param[in]  options  The options of the algorithm (see ChangeDetectorOptions)
   */
  explicit ChangeDetector(const Mesh &mesh,
                          const ChangeDetectorOptions &options);

#ifndef NOLIVIER
  void DepthFilter(Image &image);
#endif //NOLIVIER
  /**
   * @brief      Adds an image to the sequence and process it.
   *
   * @param[in]  image  The image
   * @param[in]  kernel_size The size of the window over which the image is
   * compared with the others. It represents the uncertainty of the camera pose
   */
  void AddImage(Image &image, int kernel_size);
#ifndef NOLIVIER
  void AddImage(Image &image2, int kernel_size, bool normal);
#endif //NOLIVIER

  /**
   * @brief      Gets the detected changes.
   *
   * @return     The changes.
   */
  std::vector<PointCovariance3d> GetChanges();
#ifndef NOLIVIER
  void GetChanges2(std::vector<PointCovariance3d> regions3d_in_model[2]);
  cv::Mat ImageChange(Image &image, const std::vector<PointCovariance3d> &regions3d, int width = 0);
  cv::Mat ImageChange(Image &image, const std::vector<PointCovariance3d> &regions3d, Image &mask, int width = 0);
  std::shared_ptr<ImageSequence> GetImageSequence() const { return image_sequence_; }
#endif //NOLIVIER

 protected:
  /** The 3D model of the environment */
  Mesh mesh_;

  /** The struct containing the options of the algorithm */
  ChangeDetectorOptions options_;

  /** It stores whether the depth projector has been initialized or not */
  bool projector_init_ = false;

  /** The sequence of images */
  std::shared_ptr<ImageSequence> image_sequence_;

  /** The depth projector */
  std::unique_ptr<DepthProjector> depth_projector_;
#ifndef NOLIVIER
  /** The normal projector */
  std::unique_ptr<DepthProjector> normal_projector_;
#endif //NOLIVIER
};

}  // namespace fastcd
