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
#include "fastcd/regions_matcher_hist.h"
#include <algorithm>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

namespace fastcd {

void RegionsMatcherHist::Match2(const ProcessedImage &img1,
                           const ProcessedImage &img2, int min_label) {
  max_label_ = min_label;
  std::vector<ImageRegion> regions_img1 = img1.GetRegions2();
  std::vector<ImageRegion> regions_img2 = img2.GetRegions2();
  labels1_.resize(regions_img1.size());
  labels2_.resize(regions_img2.size());
  std::vector<std::vector<cv::Point>> contours(1);
  std::unordered_map<int, cv::Mat> epimasks1;
  std::vector<cv::MatND> img1_hists(regions_img1.size());
  std::vector<cv::MatND> img2_hists(regions_img2.size());
  for (size_t i = 0; i < regions_img1.size(); i++) {
    labels1_[i] = regions_img1[i].label_;
    cv::Mat epimask = EpilineMask2(img1, img2, i);
    epimasks1[i] = epimask;
    contours[0] = regions_img1[i].contour_;
    cv::Mat mask(img1.Height(), img1.Width(), CV_8UC1, cv::Scalar(0));
    cv::drawContours(mask, contours, -1, cv::Scalar(255), CV_FILLED);
  }
  std::unordered_map<int, cv::Mat> epimasks2;
  for (size_t i = 0; i < regions_img2.size(); i++) {
    labels2_[i] = regions_img2[i].label_;
    cv::Mat epimask = EpilineMask2(img2, img1, i);
    epimasks2[i] = epimask;
    contours[0] = regions_img2[i].contour_;
    cv::Mat mask(img2.Height(), img2.Width(), CV_8UC1, cv::Scalar(0));
    cv::drawContours(mask, contours, -1, cv::Scalar(255), CV_FILLED);
  }
  std::vector<int> idxs;
  std::vector<double> corrs;
  #ifndef NOLIVIER
    int pic(rand());
  #endif
  for (size_t i = 0; i < regions_img1.size(); i++) {
    double max = 0;
    int maxj;
    bool geomeric_check1, geomeric_check2;
    for (size_t j = 0; j < regions_img2.size(); j++) {
      /*geomeric_check1 = false;
      geomeric_check2 = false;
      std::vector<cv::Point> contour = regions_img2[j].contour_;
      for (auto &pt : contour) {
        if (epimasks1[i].at<uchar>(pt.y, pt.x) == 0) {
          geomeric_check1 = true;
          break;
        }
      }
      contour = regions_img1[i].contour_;
      for (auto &pt : contour) {
        if (epimasks2[j].at<uchar>(pt.y, pt.x) == 0) {
          geomeric_check2 = true;
          break;
        }
      }*/

      double corr;
      /*if (0) {//(geomeric_check1 || geomeric_check2) {
        corr = 0;
      } else {*/
        contours[0] = regions_img1[i].contour_;
        cv::Mat mask1(img1.Height(), img1.Width(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(mask1, contours, -1, cv::Scalar(255), CV_FILLED);

        cv::Moments M1(cv::moments(contours[0]));
        int cx1(M1.m10/M1.m00), cy1(M1.m01/M1.m00);
        Eigen::Vector2i coord1(img2.WarpPixel(mask1.rows - cy1 - 1, cx1, img1));

        contours[0] = regions_img2[j].contour_;
        cv::Mat mask2(img2.Height(), img2.Width(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(mask2, contours, -1, cv::Scalar(255), CV_FILLED);

        cv::Moments M2(cv::moments(contours[0]));
        int cx2(M2.m10/M2.m00), cy2(M2.m01/M2.m00);
        Eigen::Vector2i coord2(img1.WarpPixel(mask2.rows - cy2 - 1, cx2, img2));

        corr = 0;
        if (coord1(1) >= 0 && coord1(1) < mask2.rows &&
            coord1(0) >= 0 && coord1(0) < mask2.cols)
          if (mask2.at<unsigned char>(coord1(1), coord1(0)) > 0)
            corr += 0.5;
        if (coord2(1) >= 0 && coord2(1) < mask1.rows &&
            coord2(0) >= 0 && coord2(0) < mask1.cols)
          if (mask1.at<unsigned char>(coord2(1), coord2(0)) > 0)
            corr += 0.5;

        /*if (corr) {
          cv::Mat mask1b(mask1.clone());
          mask1b.at<unsigned char>(cy1, cx1) = 63;//depart1
          mask1b.at<unsigned char>(coord2(1), coord2(0)) = 191;//destination2
          cv::imwrite("/home/olivier/bmask2/"+std::to_string(pic)+"_"+std::to_string(i)+"_1_"+std::to_string(i)+".png", mask1b);
          cv::Mat mask2b(mask2.clone());
          mask2b.at<unsigned char>(cy2, cx2) = 63;//depart2
          mask2b.at<unsigned char>(coord1(1), coord1(0)) = 191;//destination1
          cv::imwrite("/home/olivier/bmask2/"+std::to_string(pic)+"_"+std::to_string(i)+"_2_"+std::to_string(i)+".png", mask2b);
        }*/
      /*}*/
      if (corr > max) {
        max = corr;
        maxj = j;
      }
    }
    idxs.push_back(maxj);
    corrs.push_back(max);
  }

  double threshold, max = 0, min = 1;
  for (size_t i = 0; i < corrs.size(); i++) {
    if (corrs[i] > max) max = corrs[i];
    if (corrs[i] < min) min = corrs[i];
  }
  threshold = std::max((max - min) * 0.7 + min, 0.5);
  for (size_t i = 0; i < regions_img1.size(); i++) {
    if (corrs[i] >= threshold) {
      if (labels2_[idxs[i]] < 0) {
        labels2_[idxs[i]] = labels1_[i] = max_label_++;
      } else {
        labels1_[i] = labels2_[idxs[i]];
      }
    }
  }
}

cv::Mat RegionsMatcherHist::EpilineMask2(const ProcessedImage &img1,
                                    const ProcessedImage &img2, int idx) {
  //Eigen::Matrix3d F = ComputeF(img1.GetCamera().GetP(),
   //                            img2.GetCamera().GetP());
  Eigen::Matrix3d F = img1.GetCamera().ComputeF(img2.GetCamera());
  cv::Mat epimask(img1.Height(), img1.Width(), CV_8UC1,
                  cv::Scalar(0));
  std::vector<ImageRegion> regions_img1 = img1.GetRegions2();
  for (size_t i = 0; i < regions_img1[idx].contour_.size(); i++) {
    Eigen::Vector3d pt;
    pt << regions_img1[idx].contour_[i].x,
        regions_img1[idx].contour_[i].y, 1;
    Eigen::Vector3d line = F * pt;
    cv::line(
        epimask, cv::Point(0, -line(2) / line(1)),
        cv::Point(epimask.cols, -(line(2) + line(0) * epimask.cols) / line(1)),
        cv::Scalar(255));
  }
  char name[256];
  sprintf(name, "C:/Users/olivier.roupin/out/%d_1.png", idx);
  cv::imwrite(name, epimask);
  cv::Mat dilation_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(51, 51));
  cv::dilate(epimask, epimask, dilation_kernel);
  sprintf(name, "C:/Users/olivier.roupin/out/%d_2.png", idx);
  cv::imwrite(name, epimask);
  return epimask;
}

}  // namespace fastcd
#endif //NOLIVIER
