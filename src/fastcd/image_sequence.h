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
#include <deque>
#include <memory>
#include <vector>
#include "fastcd/processed_image.h"

#ifndef NOLIVIER
#include <opencv2/opencv.hpp>
#endif

namespace fastcd {

/**
 * @brief      Class representing a sequence of images.
 */
class ImageSequence {
 public:
  /**
   * @brief      Constructor.
   *
   * @param[in]  cache_size      The maximum number of images stored in the
   *                             sequence
   * @param[in]  max_comparisons  The maximum number of comparisons of a single
   *                              image with other images
   */
  explicit ImageSequence(int cache_size, int max_comparisons);

  /**
   * @brief      Adds an image to the sequence and update all the relevant
   *             inconsistency images.
   *
   * @param[in]  image  The image to add
   *
   * @param[in]  kernel_size The size of the window over which the image is
   * compared with the others. It represents the uncertainty of the camera pose
   */
  void AddImage(std::shared_ptr<ProcessedImage> image, int kernel_size);
#ifndef NOLIVIER
  void AddImage2(std::shared_ptr<ProcessedImage> image, int kernel_size);
#endif //NOLIVIER

  /**
   * @brief      Segments the regions of inconsistencies in each image and
   *             matches them.
   *
   * @param[in]  threshold_change_area Threshold area under which regions are
   * not considered
   */
  void ComputeAndMatchRegions(int threshold_change_area);
#ifndef NOLIVIER
  void ComputeAndMatchRegions(int threshold_change_area, int threshold_change_value);// = 240
#endif //NOLIVIER

  /**
   * @brief      Access the i-th element of the sequence.
   *
   * @param[in]  i     Element of the sequence to access
   *
   * @return     The selected element.
   */
  ProcessedImage operator[](int i) const { return sequence_[i]; }

  /**
   * @brief      Access the i-th element of the sequence.
   *
   * @param[in]  i     Element of the sequence to access
   *
   * @return     The selected element.
   */
  ProcessedImage &operator[](int i) { return sequence_[i]; }

  /**
   * @brief      Gets the size of the sequence.
   *
   * @return     The size of the sequence.
   */
  std::size_t size() { return sequence_.size(); }

 protected:
  /** The maximum number of images stored in the sequence */
  int cache_size_;

  /** The maximum number of comparisons of a single image with other images */
  int max_comparisons_;

  /** The sequence of images */
  std::deque<ProcessedImage> sequence_;
};

}  // namespace fastcd
