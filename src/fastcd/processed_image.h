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
#include <string>
#include <vector>
#include <unordered_map>
#include <glow/glutil.h>
#include <opencv2/core/core.hpp>
#include "fastcd/camera.h"
#include "fastcd/point_covariance2d.h"
#include "fastcd/image.h"
#include "fastcd/image_region.h"

#ifndef NOLIVIER
#include "example.h"
#endif  // NOLIVIER

namespace fastcd {

/**
 * @brief      Class that stores an image, its camera calibration, and all the
 *             necessary information for the change detection algorithm. It also
 *             includes the algorithms to generate such information.
 */
class ProcessedImage : public Image {
 public:
  /**
   * @brief      Constructor.
   *
   * @param[in]  img    The image
   * @param[in]  depth  The depth image w.r.t. the 3D model
   */
  explicit ProcessedImage(const Image &img,
                          const std::vector<glow::vec4> &depth);
#ifndef NOLIVIER
  explicit ProcessedImage(const Image &img,
                          const std::vector<glow::vec4> &depth,
                          const std::vector<glow::vec4> &background);
#endif //NOLIVIER
  /**
   * @brief      Warps the image using camera calibration of the provided image
   *             and the depth.
   *
   * @param[in]  target_image  The target image
   *
   * @return     The warped image
   */
  Image Warp(const ProcessedImage &target_image) const;

#ifndef NOLIVIER
  void Interpolate(cv::Mat &result, const int dilation_size = 1) const;
  // For testing
  Image Warp(const ProcessedImage &target_image, bool interpolate, int factor = 2) const;
  /**
   * @brief      Warps the image using camera calibration of the provided image
   *             and the depth.
   *
   * @param[in]  target_image  The target image
   * @param[in]  shadows  Wether or not occlusions should be filled or redered
   *                      exclusively.
   *                      -2: fill with target image
   *                      -1: no filling (black)
   *                      0: fill with projected image
   *                      1: show occlusions only (projected image)
   *
   * @return     The warped image
   */
  Image Warp2(const ProcessedImage &target_image, int shadows = -1) const;
  cv::Mat Warp2(const ProcessedImage &target_image, cv::Mat &visible, cv::Mat &occluded) const;
  // Gives warp coordinates for one pixel
  Eigen::Vector2i WarpPixel(int i, int j, const ProcessedImage &target_image) const;
  Eigen::Vector2i WarpPixel(Eigen::Vector2i coord, const ProcessedImage &target_image) const;
  // Replaces raw_image_ by delta
  Image Warp2(const ProcessedImage &target_image, const cv::Mat &delta, int shadows = -1) const;
  // Replaces raw_image_ by delta of type double (used for depth)
  Image Warp2f(const ProcessedImage &target_image, const cv::Mat &delta, int shadows = -1) const;
  // Computes for shadows == -1
  Image WarpFast(const ProcessedImage &target_image) const;
  // Computes for shadows == 1
  Image WarpShadows(const ProcessedImage &target_image) const;
  // Computes for shadows == -1 with half the N/V resolution
  Image WarpHalf(const ProcessedImage &target_image, bool interpolate = true) const;
  // Computes with no regard to shadows
  Image WarpFastest(const ProcessedImage &target_image) const;

  cv::Mat WarpEpiline(const ProcessedImage &target, const Eigen::Vector2i& coord) const;
  cv::Mat WarpLine(const ProcessedImage &target_image, const Eigen::Vector2i& coord) const;

  void WarpDepthSparse(const ProcessedImage &target_image);
  void WarpDepth(const ProcessedImage &target_image);

  //cv::Mat WarpMap(const ProcessedImage &target_image) const;

  cv::Mat UnWarp(const ProcessedImage &target_image, const cv::Mat &delta) const;
  cv::Mat UnWarp(const ProcessedImage &target_image, const cv::Mat &delta, int incertitudes) const;

  /**
   * @brief      Check the inconsistency image by comparing this image with the
   *             one passed as input (it computes the difference over a
   *             window of size kernel_size).
   *
   * @param[in]  image  The new image to be compared
   * @param[in]  kernel_size The size of the window over which the
   *                        difference is computed
   */
  cv::Mat CheckInconsistencies(const Image &image, int kernel_size);

  /**
   * @brief      Compound the inconsistency image by comparing this image with the
   *             one passed as input (it computes the maximum difference over a
   *             window of size kernel_size).
   *
   * @param[in]  image  The new image to be compared
   * @param[in]  kernel_size The size of the window over which the maximum
   *                         difference is computed
   */
  void CompoundInconsistencies(const Image &image, int kernel_size);
  //void UpdateInconsistencies2(const ProcessedImage &image, int kernel_size);
  void UpdateInconsistencies(const ProcessedImage &image,
                             cv::Mat &visible, cv::Mat &occluded, int kernel_size);
  void UpdateInconsistencies2(const Image &image, int kernel_size);
#endif
  /**
   * @brief      Update the inconsistency image by comparing this image with the
   *             one passed as input (it computes the minimum difference over a
   *             window of size kernel_size).
   *
   * @param[in]  image  The new image to be compared
   * @param[in]  kernel_size The size of the window over which the minimum
   *                         difference is computed
   */
  void UpdateInconsistencies(const Image &image, int kernel_size);

  /**
   * @brief      Gets the number of images compared with this one.
   *
   * @return     The number of images compared with this one.
   */
  int ImageCompared();
#ifndef NOLIVIER
  //int ImageCompared2();
#endif //NOLIVIER

  /**
   * @brief      Calculates the mean position and covariance of the points of
   *             each region with the same label.
   *
   * @param[in]  chi_square  The chi square value (95% of the variance retained
   *                          by default)
   *
   * @return     The labels of the processed regions.
   */
  std::vector<int> ComputeMeansCovariances(double chi_square = 5.991);
#ifndef NOLIVIER
  std::vector<int> ComputeMeansCovariances2(double chi_square = 5.991);
#endif //NOLIVIER

  /**
   * @brief      Extract the regions where changes occur from the inconsistency
   *             image.
   *
   * @param[in]  threshold_change_area Threshold area under which regions are
   * not considered
   */
  void ComputeRegions(int threshold_change_area);
#ifndef NOLIVIER
  void ComputeRegions(int threshold_change_area, int threshold_change_value);// = 240);
#endif //NOLIVIER

  /**
   * @brief      Gets the mean position and covariance of the points of each
   *             region with the same label.
   *
   * @return     The map containing the pairs [label, XXX].
   */
  std::unordered_map<int, PointCovariance2d> GetMeansCovariances() const;
#ifndef NOLIVIER
  std::unordered_map<int, PointCovariance2d> GetMeansCovariances2() const;
#endif

  /**
   * @brief      Gets the inconsistency image.
   *
   * @return     The inconsistency image.
   */
  cv::Mat GetInconsistencies();
#ifndef NOLIVIER
  cv::Mat GetInconsistencies2();
#endif

  /**
   * @brief      Gets the regions where changes occur.
   *
   * @return     The regions.
   */
  const std::vector<ImageRegion>& GetRegions() const;
#ifndef NOLIVIER
  const std::vector<ImageRegion>& GetRegions2() const;
#endif

  /**
   * @brief      Gets the depth image.
   *
   * @return     The depth image.
   */
  const std::vector<glow::vec4>& GetDepth() const;
#ifndef NOLIVIER
  Image DepthMap(const ProcessedImage &target_image) const;
  Image DepthDiff(const ProcessedImage &target_image) const;
  cv::Mat ComputeDepthImage();
  cv::Mat ComputeNormalImage();
  cv::Mat ComputeCoordImage(const ProcessedImage &target_image);
  void DigDepthImage(const ProcessedImage &target_image);
  void ComputePresenceImage();
  void ComputePresenceImage(cv::Mat gt);
  void ComputePresenceImage(cv::Mat gt, cv::Mat mask);
  cv::Mat GetDepthImage() const;

  void EraseBackground();
  void ComputeBackgroundImage();
  cv::Mat GetBackgroundImage() const;

  cv::Mat Occlusion(const ProcessedImage &image) const;
  void UpdatePlanes(const cv::Mat& plane);
  cv::Mat GetPlanes();
  cv::Mat GetMask();
  int GetMaskSize();
  void SaveDepthNPY(std::string fname) const;
#endif

  /**
   * @brief      Update the label of the regions with the provided values.
   *
   * @param[in]  labels  The new labels
   */
  void UpdateLabels(const std::vector<int> &labels);
#ifndef NOLIVIER
  void UpdateLabels2(const std::vector<int> &labels);
#endif //NOLIVIER

 protected:
  /**
   * @brief      For every pixel of img1 subtracts the color of img2 in a window
   *             of kernel_size pixel and stores the smallest L2 norm of the
   *             result in a OpenCV matrix of doubles.
   *
   * @param[in]  img1  The first image
   * @param[in]  img2  The second image
   * @param[in]  kernel_size The size in pixel of the window in img2 over which
   *                         the difference is computed
   *
   * @return     The resulting OpenCV matrix.
   */
  cv::Mat Subtract(const cv::Mat &img1, const cv::Mat &img2, int kernel_size);
#ifndef NOLIVIER
  //cv::Mat Subtract(const cv::Mat &rgb1, const cv::Mat &rgb2, int kernel_size);
#endif // NOLIVIER

  /**
   * @brief      Computes the minimum element-wise of two OpenCV matrices of
   * doubles.
   *
   * @param[in]  img1  The first matrix
   * @param[in]  img2  The second matrix
   *
   * @return     The resulting matrix
   */
  cv::Mat Minimum(const cv::Mat &img1, const cv::Mat &img2);

#ifndef NOLIVIER
  cv::Mat MinimumNZ(const cv::Mat &img1, const cv::Mat &img2);
  cv::Mat MinimumMC(const cv::Mat &img1, const cv::Mat &img2);
  /**
   *@brief      Computes the maximum element-wise of two OpenCV matrices of
   * doubles.
   *
   * @param[in]  img1  The first matrix
   * @param[in]  img2  The second matrix
   *
   * @return     The resulting matrix
   */
  cv::Mat Maximum(const cv::Mat &img1, const cv::Mat &img2);

  /**
   *@brief      Computes the maximum element-wise of two OpenCV matrices of
   * doubles, with the exeption of zeros.
   *
   * @param[in]  img1  The first matrix
   * @param[in]  img2  The second matrix
   *
   * @return     The resulting matrix
   */
  cv::Mat MaximumNZ(const cv::Mat &img1, const cv::Mat &img2);
#endif

  /**
   * @brief      Find non-zero elements in the matrix
   *
   * @param[in]  img   The matrix
   *
   * @return     The (rows*cols) by 2 matrix containing all the non-zero
   *             elements.
   */
  cv::Mat FindNonZero(const cv::Mat &img);

  /**
   * @brief      Segments the inconsistency image into region and fills those
   *             regions with their average intensity.
   *
   * @param[in]  img   The inconsistency image
   * @param[in]  threshold_change_area Threshold area under which regions are
   * not considered
   *
   * @return     The inconsistency image with the segmented regions.
   */
  cv::Mat AverageSegmentedRegions(const cv::Mat &img, int threshold_change_area);

  /**
   * The depth image w.r.t. the 3D model, in the form of a vector of 4D points
   * ordered row-wise from the bottom left. The 4th coordinate is used as a
   * boolean: it has value 0 if the back-projected ray does not hit the model,
   * 1 otherwise
   */
  std::vector<glow::vec4> depth_;
#ifndef NOLIVIER
  std::vector<glow::vec4> background_;
#endif

  /** The inconsistency image (doubles from 0 to 1) */
  cv::Mat inconsistencies_;
#ifndef NOLIVIER
  cv::Mat inconsistencies2_;
  cv::Mat depth_image_;
  cv::Mat background_image_;
  cv::Mat planes_;
  int mask_size_ = -1;
#endif

  /** The segmented regions */
  std::vector<ImageRegion> regions_;
#ifndef NOLIVIER
  std::vector<ImageRegion> regions2_;
#endif

  /**
   * The mean position and covariance of the points of each region with the same
   * label
   */
  std::unordered_map<int, PointCovariance2d> regions_mean_cov_;
#ifndef NOLIVIER
  std::unordered_map<int, PointCovariance2d> regions2_mean_cov_;
#endif

  /** The highest label assigned to a region */
  int max_region_label_ = 0;
#ifndef NOLIVIER
  int max_region_label2_ = 0;
#endif

  /** The number of images this one has been compared to*/
  int num_img_compared_ = 0;
#ifndef NOLIVIER
  //int num_img_compared2_ = 0;
  int num_img_compounded_ = 0;
  int num_planes_compared_ = 0;
#endif
};

}  // namespace fastcd
