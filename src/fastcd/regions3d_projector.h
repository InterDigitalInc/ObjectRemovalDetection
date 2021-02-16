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
#include <vector>
#include <set>
#include <utility>
#include <Eigen/Core>
#include "fastcd/image_sequence.h"
#include "fastcd/processed_image.h"
#include "fastcd/point_covariance3d.h"

namespace fastcd {

/**
 * @brief      Class that backprojects in 3D the changes identified in a
 *             sequence of images.
 */
class Regions3dProjector {
 public:
  /**
   * @brief      Constructor
   *
   * @param[in]  img_sequence  The image sequence
   * @param[in]  chi_square2d  The chi square for the ellipses in the images
   *                           (95% of the variance retained by default)
   * @param[in]  chi_square3d  The chi square for the 3D ellipsoids
   *                           (95% of the variance retained by default)
   * @param[in]  threshold_change_area Threshold area under which regions of
   *                                   change in 2D are not considered
   */
  Regions3dProjector(std::shared_ptr<ImageSequence> img_sequence,
                     int threshold_change_area, double chi_square2d = 5.991,
                     double chi_square3d = 7.815);
#ifndef NOLIVIER
  Regions3dProjector(std::shared_ptr<ImageSequence> img_sequence,
                     int threshold_change_area, int threshold_change_value,// = 240
                     double chi_square2d = 5.991, double chi_square3d = 7.815);
#endif //NOLIVIER

  /**
   * @brief      Gets the mean points of the changed 3D regions with their
   *             covariance.
   *
   * @return     The mean points of the changed 3D regions with their
   *             covariance.
   */
  std::vector<PointCovariance3d> GetProjectedRegions();
#ifndef NOLIVIER
  std::vector<PointCovariance3d> GetProjectedRegions2();
#endif //NOLIVIER

 protected:
  /**
   * @brief      Triangulates the mean points of the 2D regions in the image
   *             sequence with a certain label
   *
   * @param[in]  img_deque  The deque of images
   * @param[in]  label      The label of the regions to triangulate
   *
   * @return     The triangulated 3D point.
   */
  Eigen::Vector3d TriangulateMeanRay(
      const std::shared_ptr<ImageSequence> &img_deque,
      const std::pair<int, std::set<int>> &label);
#ifndef NOLIVIER
  Eigen::Vector3d TriangulateMeanRay2(
      const std::shared_ptr<ImageSequence> &img_deque,
      const std::pair<int, std::set<int>> &label);
#endif //NOLIVIER

  /**
   * @brief      Projects the vertices of the ellipses corresponding to a
   *             certain label onto a plane passing through the triangulated
   *             mean and parallel to the image plane.
   *
   * @param[in]  img_deque          The deque of images
   * @param[in]  triangulated_mean  The triangulated mean
   * @param[in]  label              The label of the regions to consider
   *
   * @return     The 3D positions of the projected vertices.
   */
  std::vector<Eigen::Vector3d> ProjectVertices(
      const std::shared_ptr<ImageSequence> &img_deque,
      const Eigen::Vector3d &triangulated_mean,
      const std::pair<int, std::set<int>> &label);
#ifndef NOLIVIER
  std::vector<Eigen::Vector3d> ProjectVertices2(
      const std::shared_ptr<ImageSequence> &img_deque,
      const Eigen::Vector3d &triangulated_mean,
      const std::pair<int, std::set<int>> &label);
#endif //NOLIVIER

  /**
   * @brief      Compute the skew-symmetric matrix corresponding to the input
   *             vector.
   *
   * @param[in]  vec   The vector
   *
   * @return     The skew-symmetric matrix
   */
  Eigen::Matrix3d SkewSymmetric3d(const Eigen::Vector3d &vec);

  /**
   * @brief      Calculates the mean position of the given 3D points.
   *
   * @param[in]  points  The vector of 3D points
   *
   * @return     The mean position.
   */
  Eigen::Vector3d ComputeMean(const std::vector<Eigen::Vector3d> &points);

  /**
   * @brief      Calculates the covariance corresponding to the mean position of
   *             the given 3D points.
   *
   * @param[in]  points  The vector of 3D points
   * @param[in]  mean    The mean position
   *
   * @return     The covariance.
   */
  Eigen::Matrix3d ComputeCovariance(const std::vector<Eigen::Vector3d> &points,
                                    const Eigen::Vector3d &mean);

  /**
   * The mean of the points of the 3D regions with the corresponding covariance
   */
  std::vector<PointCovariance3d> projected_regions_;
#ifndef NOLIVIER
  std::vector<PointCovariance3d> projected_regions2_;
#endif //NOLIVIER
};

}  // namespace fastcd
