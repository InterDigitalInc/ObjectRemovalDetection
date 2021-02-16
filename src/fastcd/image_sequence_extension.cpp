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
#include "fastcd/image_sequence.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include "fastcd/regions_matcher_hist.h"

#include "example.h"

namespace fastcd {

void ImageSequence::AddImage2(std::shared_ptr<ProcessedImage> image,
                             int kernel_size) {
  if (sequence_.size() == cache_size_) {
    sequence_.pop_front();
  }
  if (!sequence_.empty()) {
    for (int i = sequence_.size() - 1; i >= 0; i--) {
      if (sequence_[i].ImageCompared()  < max_comparisons_ /*&&
          sequence_[i].ImageCompared2() < max_comparisons_*/) {
#if defined(PALAZZOLO)
        sequence_[i].UpdateInconsistencies(image->Warp(sequence_[i]),kernel_size);
#else
        cv::Mat visible(image->Height(), image->Width(),
                        image->GetRawImage().type(), cv::Scalar(0, 0, 0));
        cv::Mat occluded(image->Height(), image->Width(),
                         image->GetRawImage().type(), cv::Scalar(0, 0, 0));
        /*cv::Mat mask = */image->Warp2(sequence_[i], visible, occluded);
        sequence_[i].UpdateInconsistencies(*image, visible, occluded, kernel_size);
#endif //PALAZZOLO
      }
      if (image->ImageCompared()  < max_comparisons_ /*&&
          image->ImageCompared2() < max_comparisons_*/) {
#if defined(PALAZZOLO)
        image->UpdateInconsistencies(sequence_[i].Warp(*image),kernel_size);
#else
        cv::Mat visible(sequence_[i].Height(), sequence_[i].Width(),
                        sequence_[i].GetRawImage().type(), cv::Scalar(0, 0, 0));
        cv::Mat occluded(sequence_[i].Height(), sequence_[i].Width(),
                         sequence_[i].GetRawImage().type(), cv::Scalar(0, 0, 0));
        /*cv::Mat mask = */sequence_[i].Warp2(*image, visible, occluded);
        image->UpdateInconsistencies(sequence_[i], visible, occluded, kernel_size);
#endif //PALAZZOLO
      }
    }
  }
  sequence_.push_back(*image);
}

void ImageSequence::ComputeAndMatchRegions(int threshold_change_area, int threshold_change_value) {
  for (size_t i = 0; i < sequence_.size(); i++) {
      sequence_[i].ComputeRegions(threshold_change_area, threshold_change_value);
  }

  int label  = 0;
  int label2 = 0;
    RegionsMatcherHist regions_matcher;
    RegionsMatcherHist regions_matcher2;
    for (size_t i = 0; i < sequence_.size(); i++) {
      for (size_t j = i + 1; j < sequence_.size(); j++) {
        regions_matcher.Match(sequence_[i], sequence_[j], label);
        sequence_[i].UpdateLabels(regions_matcher.GetLabelsImg1());
        sequence_[j].UpdateLabels(regions_matcher.GetLabelsImg2());
        label = regions_matcher.GetMaxLabel();

        regions_matcher2.Match2(sequence_[i], sequence_[j], label2);
        sequence_[i].UpdateLabels2(regions_matcher2.GetLabelsImg1());
        sequence_[j].UpdateLabels2(regions_matcher2.GetLabelsImg2());
        label2 = regions_matcher2.GetMaxLabel();
      }
    }
  //}
}

}  // namespace fastcd
#endif //NOLIVIER
