/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <thread>

#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>

#include <basalt/image/image_pyr.h>
#include <basalt/utils/keypoints.h>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

namespace basalt {

/// Unlike PatchOpticalFlow, FrameToFrameOpticalFlow always tracks patches
/// against the previous frame, not the initial frame where a track was created.
/// While it might cause more drift of the patch location, it leads to longer
/// tracks in practice.
template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow : public OpticalFlowBase {
 public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;

  FrameToFrameOpticalFlow(const VioConfig& config,
                          const basalt::Calibration<double>& calib)
      : t_ns(-1), frame_counter(0), last_keypoint_id(0), config(config) {
    input_queue.set_capacity(10);

    this->calib = calib.cast<Scalar>();

    patch_coord = PatchT::pattern2.template cast<float>();

    if (calib.intrinsics.size() > 1) {
      
      for (size_t k = 0; k < calib.intrinsics.size(); k+=2)
      {
        std::cout << "Optical flow initialise camera " << k << " and " << k+1;
        Eigen::Matrix4d Ed;
        Sophus::SE3d T_i_j = calib.T_i_c[k].inverse() * calib.T_i_c[k+1];
        computeEssential(T_i_j, Ed);
        E.push_back(Ed.cast<Scalar>());
      }
      
    }

    processing_thread.reset(
        new std::thread(&FrameToFrameOpticalFlow::processingLoop, this));

    std::cout << "Initialised FrameToFrameOpticalFlow" << std::endl;
  }

  ~FrameToFrameOpticalFlow() { processing_thread->join(); }

  void processingLoop() {
    OpticalFlowInput::Ptr input_ptr;

    while (true) {
      input_queue.pop(input_ptr);

      if (!input_ptr.get()) {
        if (output_queue) output_queue->push(nullptr);
        break;
      }

      processFrame(input_ptr->t_ns, input_ptr);
    }
  }

  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    if (t_ns < 0) {
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      transforms->observations.resize(calib.intrinsics.size());
      transforms->t_ns = t_ns;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(calib.intrinsics.size());

      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();

      BASALT_ASSERT(transforms->pre_last_keypoint_id == 0);

    } else {
      t_ns = curr_t_ns;

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(calib.intrinsics.size());
      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(calib.intrinsics.size());
      new_transforms->t_ns = t_ns;

      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        trackPoints(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i], i, i);
      }

      transforms = new_transforms;
      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();

      //draw matching points
      if(config.feature_match_show){

        for (size_t k=0; k < calib.intrinsics.size(); k+=2)
        {
          std::vector<cv::KeyPoint> kp1, kp2, kp0;
          std::vector<cv::DMatch> match;
          int match_id = 0;
          basalt::Image<const uint16_t> img_raw_1(pyramid->at(k).lvl(1)), img_raw_2(pyramid->at(k+1).lvl(1));
          int w = img_raw_1.w; 
          int h = img_raw_1.h;
          cv::Mat img1(h, w, CV_8U);
          cv::Mat img2(h, w, CV_8U);
          for(int y = 0; y < h; y++){
            uchar* sub_ptr_1 = img1.ptr(y);
            uchar* sub_ptr_2 = img2.ptr(y);

            for(int x = 0; x < w; x++){
              sub_ptr_1[x] = (img_raw_1(x,y) >> 8);
              sub_ptr_2[x] = (img_raw_2(x,y) >> 8);

            }
          }

          for(const auto& kv: transforms->observations[k]){
            
            // hm: skip keypoints that are too new
            if (kv.first > pre_last_keypoint_id)
              continue;
            
            auto it = transforms->observations[k+1].find(kv.first);
            if(it != transforms->observations[k+1].end()){
              
              kp1.push_back(cv::KeyPoint(cv::Point2f(kv.second.translation()[0]/2, kv.second.translation()[1]/2), 1));
              kp2.push_back(cv::KeyPoint(cv::Point2f(it->second.translation()[0]/2, it->second.translation()[1]/2), 1));
              match.push_back(cv::DMatch(match_id,match_id,1));
              match_id++;
            }
            else{
              kp0.push_back(cv::KeyPoint(cv::Point2f(kv.second.translation()[0]/2, kv.second.translation()[1]/2), 1));
            }
          }
          cv::Mat image_show(h, w*2, CV_8U);
          cv::drawKeypoints(img1, kp0,img1);
          cv::drawMatches(img1,kp1,img2,kp2,match, image_show);
          std::string title = "matching result" + std::to_string(k/2);

          cv::imshow(title, image_show);
          cv::waitKey(1);
        }
        
      }
    }

    // hm: addtional metadata regarding the ids that are newly added
    transforms->last_keypoint_id = last_keypoint_id;
    transforms->pre_last_keypoint_id = pre_last_keypoint_id;

    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      output_queue->push(transforms);
    }

    frame_counter++;
  }

  void trackPoints(const basalt::ManagedImagePyr<uint16_t>& pyr_1,
                   const basalt::ManagedImagePyr<uint16_t>& pyr_2,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_1,
                   Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_2, int cam_id_1, int cam_id_2) const {
    size_t num_points = transform_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto& kv : transform_map_1) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);
    }

    tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f,
                                  std::hash<KeypointId>>
        result;

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {

        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;

        if (cam_id_1 != cam_id_2){
          // hm: we want to modify the tranlation part of the transform, as we have two diff cameras (assume identical time)
          Eigen::Vector2f t1 = transform_1.translation();
          Eigen::Vector4f p1_3d;
          calib.intrinsics[cam_id_1].unproject(t1, p1_3d);
          Eigen::Vector4f p2_3d = calib.T_i_c[cam_id_2].so3().inverse() * calib.T_i_c[cam_id_1].so3() * p1_3d;
          Eigen::Vector2f t2;
          calib.intrinsics[cam_id_2].project(p2_3d, t2);

          transform_2.translation() = t2;

        }

        bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2, cam_id_2);

        if (valid) {
          Eigen::AffineCompact2f transform_1_recovered = transform_2;

          if (cam_id_2 != cam_id_1){
            // hm: we want to modify the tranlation part of the transform, as we have two diff cameras (assume identical time)
            Eigen::Vector2f t2 = transform_2.translation();
            Eigen::Vector4f p2_3d;
            calib.intrinsics[cam_id_2].unproject(t2, p2_3d);
            Eigen::Vector4f p1_3d = calib.T_i_c[cam_id_1].so3().inverse() * calib.T_i_c[cam_id_2].so3() * p2_3d;
            Eigen::Vector2f t1;
            calib.intrinsics[cam_id_1].project(p1_3d, t1);

            transform_1_recovered.translation() = t1;
          }

          valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered, cam_id_1);

          if (valid) {
            Scalar dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                               .squaredNorm();

            if (dist2 < config.optical_flow_max_recovered_dist2) {
              // result[ids[r]] = transform_2;
              // hm: using initialisation list get rid of the maybe un-initialised warning within use of concurrent map
              result.emplace(ids[r], transform_2);
            }
          }
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);

    tbb::parallel_for(range, compute_func);
    // compute_func(range);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }

  inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t>& old_pyr,
                         const basalt::ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform, int cam_id_2) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = config.optical_flow_levels; level >= 0 && patch_valid;
         level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
      (void)cam_id_2;
      // patch_valid &= calib.intrinsics[cam_id_2].inBound(transform.translation());
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2,
                                const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int iteration = 0;
         patch_valid && iteration < config.optical_flow_max_iterations;
         iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res;

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix();

          const int filter_margin = 2;

          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    return patch_valid;
  }

  void addPoints() {
    pre_last_keypoint_id = last_keypoint_id;
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (size_t k=0; k < calib.intrinsics.size(); k+=2)
    {
      Eigen::aligned_vector<Eigen::Vector2d> pts0;

      for (const auto& kv : transforms->observations.at(k)) {

        // hm: for the detection, if it is a stereo detection, then we do not detect for new points
        if (seq % ADD_STEREO_ONLY_INTERVAL != 0 || k + 1 >= calib.intrinsics.size())
          pts0.emplace_back(kv.second.translation().cast<double>());
        else if (transforms->observations.at(k+1).count(kv.first))
          pts0.emplace_back(kv.second.translation().cast<double>());
      }

      KeypointsData kd;

      detectKeypoints(pyramid->at(k).lvl(0), kd,
                      config.optical_flow_detection_grid_size, 1, pts0);

      Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
          new_poses1;

      
      if (seq % ADD_STEREO_ONLY_INTERVAL != 0 || k + 1 >= calib.intrinsics.size()) {
        for (size_t i = 0; i < kd.corners.size(); i++) {
          Eigen::AffineCompact2f transform;
          transform.setIdentity();
          transform.translation() = kd.corners[i].cast<Scalar>();

          transforms->observations.at(k)[last_keypoint_id] = transform;
          new_poses0[last_keypoint_id] = transform;

          last_keypoint_id++;
        }

        if (k + 1 < calib.intrinsics.size()) {
          trackPoints(pyramid->at(k), pyramid->at(k+1), new_poses0, new_poses1, k, k+1);

          for (const auto& kv : new_poses1) {
            transforms->observations.at(k+1).emplace(kv);
          }
        }
      }else {
        // we only add stereo matches here

        for (size_t i = 0; i < kd.corners.size(); i++) {
          Eigen::AffineCompact2f transform;
          transform.setIdentity();
          transform.translation() = kd.corners[i].cast<Scalar>();
          new_poses0[last_keypoint_id] = transform;
          last_keypoint_id++;
        }

        BASALT_ASSERT(k + 1 < calib.intrinsics.size());

        trackPoints(pyramid->at(k), pyramid->at(k+1), new_poses0, new_poses1, k, k+1);

        for (const auto& kv : new_poses1) {
          transforms->observations.at(k)[kv.first] = new_poses0[kv.first];
          transforms->observations.at(k+1).emplace(kv);
        }

      }

      
    }
    
    seq++;
  }

  void filterPoints() {

    // hm: filter points based on input queue
    {
      KeypointId id_to_remove;
      while(input_filter_ids.try_pop(id_to_remove))
      {
        std::cout << "removing kp " << id_to_remove << "in optical flow" << std::endl;
        for (size_t k=0; k < calib.intrinsics.size(); k++) {
          if (transforms->observations.at(k).count(id_to_remove))
            transforms->observations.at(k).erase(id_to_remove);
        }
      }
    }

    // filter points for stereo setup

    transforms->num_stereo_matches = 0;

    if (calib.intrinsics.size() < 2) return;

    for (size_t k=0; k < calib.intrinsics.size(); k+=2)
    {
      std::set<KeypointId> lm_to_remove;

      std::vector<KeypointId> kpid;
      Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

      for (const auto& kv : transforms->observations.at(k+1)) {
        auto it = transforms->observations.at(k).find(kv.first);

        if (it != transforms->observations.at(k).end()) {
          proj0.emplace_back(it->second.translation());
          proj1.emplace_back(kv.second.translation());
          kpid.emplace_back(kv.first);
        }
      }

      Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
      std::vector<bool> p3d0_success, p3d1_success;

      calib.intrinsics[k].unproject(proj0, p3d0, p3d0_success);
      calib.intrinsics[k+1].unproject(proj1, p3d1, p3d1_success);

      for (size_t i = 0; i < p3d0_success.size(); i++) {
        if (p3d0_success[i] && p3d1_success[i]) {
          const double epipolar_error =
              std::abs(p3d0[i].transpose() * E.at(k/2) * p3d1[i]);

          if (epipolar_error > config.optical_flow_epipolar_error) {
            lm_to_remove.emplace(kpid[i]);
          }
        } else {
          lm_to_remove.emplace(kpid[i]);
        }
      }

      for (int id : lm_to_remove) {
        transforms->observations.at(k+1).erase(id);
      }

      transforms->num_stereo_matches += transforms->observations.at(k+1).size();
    }
    
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  int64_t t_ns;

  size_t frame_counter;

  KeypointId last_keypoint_id;

  VioConfig config;
  basalt::Calibration<Scalar> calib;

  OpticalFlowResult::Ptr transforms;
  std::shared_ptr<std::vector<basalt::ManagedImagePyr<uint16_t>>> old_pyramid,
      pyramid;

  Eigen::aligned_vector<Matrix4> E;

  std::shared_ptr<std::thread> processing_thread;
};

}  // namespace basalt
