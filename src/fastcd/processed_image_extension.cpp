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
#include "fastcd/processed_image.h"
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include "fastcd/camera.h"

#include "example.h"
#include "utils/otsu_miki.h"

namespace fastcd {

ProcessedImage::ProcessedImage(const Image &img,
                               const std::vector<glow::vec4> &depth,
                               const std::vector<glow::vec4> &background)
    : Image(img), depth_(depth), background_(background) {
  inconsistencies_ =
        cv::Mat(raw_image_.rows, raw_image_.cols, CV_32FC1, cv::Scalar(0));
  inconsistencies2_ =
        cv::Mat(raw_image_.rows, raw_image_.cols, CV_32FC1, cv::Scalar(0));
  planes_ =
        cv::Mat(raw_image_.rows, raw_image_.cols, raw_image_.type(), cv::Scalar(0, 0, 0));
}

#include "fastcd/processed_image_warping.cpp"

cv::Mat ProcessedImage::CheckInconsistencies(const Image &image,
                                             int kernel_size) {
  cv::Mat inconsistency =
      Subtract(raw_image_, image.GetRawImage(), kernel_size);

  cv::normalize(inconsistency, inconsistency, 0.0f, 1.0f, cv::NORM_MINMAX);
  return inconsistency;
}

void ProcessedImage::CompoundInconsistencies(const Image &image,
                                             int kernel_size) {
  cv::Mat inconsistency =
      Subtract(raw_image_, image.GetRawImage(), kernel_size);

  cv::normalize(inconsistency, inconsistency, 0.0f, 1.0f, cv::NORM_MINMAX);
  if (num_img_compounded_ > 0)
    inconsistencies2_ = MaximumNZ(inconsistencies2_, inconsistency);
  else
    inconsistencies2_ = inconsistency;
  cv::medianBlur(inconsistencies2_, inconsistencies2_, 3);
  cv::normalize(inconsistencies2_, inconsistencies2_, 0.0f, 1.0f,
                cv::NORM_MINMAX);

  num_img_compounded_++;
}

/*void ProcessedImage::UpdateInconsistencies2(const ProcessedImage &image,
                                            int kernel_size) {
  cv::Mat inconsistency = image.UnWarp(*this,
      Subtract(raw_image_, image.WarpShadows(*this).GetRawImage(), kernel_size));

  cv::normalize(inconsistency, inconsistency, 0.0f, 1.0f, cv::NORM_MINMAX);
  if (num_img_compared2_ > 0)
    inconsistencies_ = Maximum(inconsistencies_, inconsistency);
  else
    inconsistencies_ = inconsistency;
  cv::medianBlur(inconsistencies_, inconsistencies_, 3);
  cv::normalize(inconsistencies_, inconsistencies_, 0.0f, 1.0f,
                cv::NORM_MINMAX);

  num_img_compared2_++;
}*/

void ProcessedImage::UpdateInconsistencies(const ProcessedImage &image,
                                           cv::Mat &visible, cv::Mat &occluded,
                                           int kernel_size) {
  cv::Mat inconsistency =
      Subtract(raw_image_, visible, kernel_size);
  cv::Mat inconsistency2 = image.UnWarp(*this,
      Subtract(raw_image_, occluded, kernel_size));

  cv::normalize(inconsistency,  inconsistency,  0.0f, 1.0f, cv::NORM_MINMAX);
  cv::normalize(inconsistency2, inconsistency2, 0.0f, 1.0f, cv::NORM_MINMAX);
#ifdef INSERTIONS
  inconsistency = inconsistency & ~inconsistency2;
#endif //INSERTIONS
  if (num_img_compared_ > 0) {
#if defined(NON_ZERO)
    inconsistencies_ = MinimumNZ(inconsistencies_, inconsistency);
#else
    inconsistencies_  = Minimum(inconsistencies_,  inconsistency);
#endif //NON_ZERO
    inconsistencies2_ = Maximum(inconsistencies2_, inconsistency2);
  } else {
    inconsistencies_  = inconsistency;
    inconsistencies2_ = inconsistency2;
  }
  cv::medianBlur(inconsistencies_,  inconsistencies_,  3);
  cv::normalize(inconsistencies_,  inconsistencies_,  0.0f, 1.0f,
                cv::NORM_MINMAX);
  cv::medianBlur(inconsistencies2_, inconsistencies2_, 3);
  cv::normalize(inconsistencies2_, inconsistencies2_, 0.0f, 1.0f,
                cv::NORM_MINMAX);
  num_img_compared_++;
}

void ProcessedImage::UpdateInconsistencies2(const Image &image,
                                           int kernel_size) {
  cv::Mat inconsistency =
      Subtract(raw_image_, image.GetRawImage(), kernel_size);

  cv::normalize(inconsistency, inconsistency, 0.0f, 1.0f, cv::NORM_MINMAX);
  if (num_img_compared_ > 0)
#if defined(NON_ZERO)
    inconsistencies_ = MinimumNZ(inconsistencies_, inconsistency);
#else
    inconsistencies_ = Minimum(inconsistencies_, inconsistency);
#endif //NON_ZERO
  else
    inconsistencies_ = inconsistency;
  cv::medianBlur(inconsistencies_, inconsistencies_, 3);
  cv::normalize(inconsistencies_, inconsistencies_, 0.0f, 1.0f,
                cv::NORM_MINMAX);

  num_img_compared_++;
}

cv::Mat ProcessedImage::GetInconsistencies2() { return inconsistencies2_; }

std::vector<int> ProcessedImage::ComputeMeansCovariances2(double chi_square) {
  std::vector<int> found_labels;
  for (int i = 0; i <= max_region_label2_; i++) {
    std::vector<int> idx;
    for (size_t j = 0; j < regions2_.size(); j++) {
      if (regions2_[j].label_ == i) {
        idx.push_back(j);
      }
    }
    if (!idx.empty()) {
      found_labels.push_back(i);
      cv::Mat region(Height(), Width(), CV_8UC1, cv::Scalar(0));
      std::vector<std::vector<cv::Point>> contours;
      for (int j : idx) {
        contours.push_back(regions2_[j].contour_);
      }
      cv::drawContours(region, contours, -1, cv::Scalar(255), CV_FILLED);
      cv::Mat non_zero = FindNonZero(region);
      cv::Mat mean;
      cv::Mat covariance(2, 2, CV_64F);
      cv::calcCovarMatrix(non_zero, covariance, mean,
                        CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
      Eigen::Vector2d emean;
      Eigen::Matrix2d ecovariance;
      emean << mean.at<double>(0, 0), mean.at<double>(0, 1);
      ecovariance << covariance.at<double>(0, 0), covariance.at<double>(0, 1),
      covariance.at<double>(1, 0), covariance.at<double>(1, 1);
      PointCovariance2d mean_covariance(emean, ecovariance, chi_square);
      regions2_mean_cov_[i] = mean_covariance;
    }
  }
  return found_labels;
}

void ProcessedImage::ComputeRegions(int threshold_change_area, int threshold_change_value) {
  cv::Mat image(Height(), Width(), CV_8UC1);
  inconsistencies_.convertTo(image, CV_8UC1, 255);

  cv::Mat image2(Height(), Width(), CV_8UC1);
  inconsistencies2_.convertTo(image2, CV_8UC1, 255);
  //if (threshold_change_area > 0) { // For insertions
    cv::threshold(image, image, 30, 255.0f, cv::THRESH_TOZERO);
    image2 = 255-image2;
    cv::threshold(image2, image2, 254, 255.0f, cv::THRESH_TOZERO_INV);
    if (threshold_change_value == -1)
      threshold_change_value = triangle_8u_with_mask(image2);
    //std::cout << threshold_change_value << std::endl;
    //if (threshold_change_value == -2)
    //  threshold_change_value = otsu_8u_with_mask(image2);
    cv::threshold(image2, image2, threshold_change_value, 255.0f, cv::THRESH_TOZERO);
  // } else { // For removals
  //   threshold_change_area = -threshold_change_area;
  //   image = 255-image;
  //   cv::threshold(image, image, 254, 255.0f, cv::THRESH_TOZERO_INV);
  //   cv::Mat mask = cv::Mat(image != 0);
  //   cv::threshold(image, image, /*triangle_8u_with_mask(image)*//*otsu_8u_with_mask(image, mask)*/240, 255.0f, cv::THRESH_TOZERO);
  // }

  cv::Mat dilation_kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
  cv::Mat erosion_kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::erode(image, image, erosion_kernel);
  cv::dilate(image, image, dilation_kernel);

  //if (threshold_change_area > 0) {
    cv::erode(image2, image2, erosion_kernel);
    cv::dilate(image2, image2, dilation_kernel);
    image2 = AverageSegmentedRegions(image2, threshold_change_area);

    cv::equalizeHist(image2, image2);
    cv::normalize(image2, image2, 0, 255, cv::NORM_MINMAX);
    cv::threshold(image2, image2, 50, 255.0f, cv::THRESH_TOZERO);
    std::vector<std::vector<cv::Point>> contours2;
    cv::findContours(image2, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours2.size(); i++) {
      ImageRegion region(contours2[i]);
      regions2_.push_back(region);
    }
  //} else
  //  threshold_change_area = -threshold_change_area;
  image = AverageSegmentedRegions(image, threshold_change_area);

  cv::equalizeHist(image, image);
  cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
  cv::threshold(image, image, 50, 255.0f, cv::THRESH_TOZERO);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); i++) {
    ImageRegion region(contours[i]);
    regions_.push_back(region);
  }
}

const std::vector<ImageRegion>& ProcessedImage::GetRegions2() const {
  return regions2_;
}

#include "fastcd/processed_image_depth.cpp"

void ProcessedImage::UpdateLabels2(const std::vector<int> &labels) {
  for (size_t i = 0; i < regions2_.size(); i++) {
    if (labels[i] > max_region_label2_) max_region_label2_ = labels[i];
    regions2_[i].label_ = labels[i];
  }
}

std::unordered_map<int, PointCovariance2d>
ProcessedImage::GetMeansCovariances2() const {
  return regions2_mean_cov_;
}

cv::Mat ProcessedImage::MinimumNZ(const cv::Mat &img1, const cv::Mat &img2) {
  cv::Mat result(img1.rows, img1.cols, CV_32FC1, 0.0f);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < img1.rows; ++i) {
    for (int j = 0; j < img1.cols; ++j) {
      if (img1.at<float>(i, j) > 0 && img2.at<float>(i, j) > 0)
        result.at<float>(i, j) =
            std::min(img1.at<float>(i, j), img2.at<float>(i, j));
      else if (img1.at<float>(i, j) > 0)
        result.at<float>(i, j) = img1.at<float>(i, j);
      else
        result.at<float>(i, j) = img2.at<float>(i, j);
    }
  }
  return result;
}

cv::Mat ProcessedImage::MinimumMC(const cv::Mat &img1, const cv::Mat &img2) {
  cv::Mat result(img1.rows, img1.cols, img1.type(), cv::Scalar(0, 0, 0));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < img1.rows; ++i) {
    for (int j = 0; j < img1.cols; ++j) {
      result.at<cv::Vec3b>(i, j) =
        cv::Vec3b(std::min(img1.at<cv::Vec3b>(i, j)[0], img2.at<cv::Vec3b>(i, j)[0]),
                  std::max(img1.at<cv::Vec3b>(i, j)[1], img2.at<cv::Vec3b>(i, j)[1]),
                  std::min(img1.at<cv::Vec3b>(i, j)[2], img2.at<cv::Vec3b>(i, j)[2]));
    }
  }
  return result;
}

cv::Mat ProcessedImage::Maximum(const cv::Mat &img1, const cv::Mat &img2) {
  cv::Mat result(img1.rows, img1.cols, CV_32FC1, 0.0f);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < img1.rows; ++i) {
    for (int j = 0; j < img1.cols; ++j) {
      result.at<float>(i, j) =
        std::max(img1.at<float>(i, j), img2.at<float>(i, j));
    }
  }
  return result;
}

cv::Mat ProcessedImage::MaximumNZ(const cv::Mat &img1, const cv::Mat &img2) {
  cv::Mat result(img1.rows, img1.cols, CV_32FC1, 0.0f);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < img1.rows; ++i) {
    for (int j = 0; j < img1.cols; ++j) {
      if(img1.at<float>(i, j) != 0 && img2.at<float>(i, j) != 0)
        result.at<float>(i, j) =
          std::max(img1.at<float>(i, j), img2.at<float>(i, j));
    }
  }
  return result;
}

//int ProcessedImage::ImageCompared2() { return num_img_compared2_; }

}  // namespace fastcd
#endif //NOLIVIER
