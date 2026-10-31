#pragma once
/***********************************************************
 * byte_tracker.hpp
 * 2D ByteTrack-style multi-object tracker (per camera).
 *
 * Ported from the ROS node main_cam_opt_track.cpp. The algorithm is unchanged:
 *   - 8-D constant-velocity Kalman filter (cx, cy, aspect, h + velocities)
 *   - detections split into HIGH / LOW score pools
 *   - Stage 1: tracks <-> HIGH dets, cost = 1 - GIoU, Hungarian
 *   - Stage 2: leftover tracks <-> LOW dets, cost = 1 - IoU, Hungarian
 *   - new tracks spawn only from unmatched HIGH dets
 *
 * Header-only and ROS-free. Needs OpenCV core + video (cv::KalmanFilter),
 * so link opencv_video if you list OpenCV libs by hand.
 *
 * Detections must be in ORIGINAL image pixel coordinates (i.e. after the
 * letterbox has been undone), because the Kalman noise model scales with
 * the box height in pixels.
 ***********************************************************/

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace tracker2d {

// --- Configuration ----------------------------------------------------------
struct Tracker2DConfig {
    float high_thresh        = 0.50f;   // score >= this      -> HIGH pool
    float low_thresh         = 0.15f;   // score in [low,high)-> LOW pool, below -> dropped
    float stage1_cost_thresh = 0.80f;   // max (1 - GIoU) accepted in stage 1
    float stage2_iou_thresh  = 0.30f;   // min IoU accepted in stage 2
    int   confirm_hits       = 2;       // matches needed: Tentative -> Confirmed
    int   max_age            = 15;      // frames without an update before deletion
    float std_weight_pos     = 1.0f / 20.0f;
    float std_weight_vel     = 1.0f / 160.0f;
    int   label_max          = std::numeric_limits<int>::max(); // keep dets with label <= label_max
};

enum class TrackState {
    Tentative = 1,
    Confirmed = 2,
    Lost      = 3,
    Deleted   = 4
};

// A single detection in original-image pixel coordinates.
// (Kept a plain aggregate, no default member initializers, so `{rect, score, label}`
//  brace-init also compiles under C++11.)
struct Detection {
    cv::Rect2f bbox;
    float      score;
    int        label;
};

// --- Low-footprint Kalman filter on (cx, cy, aspect, h) ---------------------
class ByteKalmanFilter {
private:
    cv::KalmanFilter kf_;
    float std_weight_position_;
    float std_weight_velocity_;

public:
    ByteKalmanFilter(float w_pos = 1.0f / 20.0f, float w_vel = 1.0f / 160.0f)
        : std_weight_position_(w_pos), std_weight_velocity_(w_vel) {

        kf_.init(8, 4, 0, CV_32F);
        cv::setIdentity(kf_.transitionMatrix);
        for (int i = 0; i < 4; ++i) {
            kf_.transitionMatrix.at<float>(i, i + 4) = 1.0f;
        }
        cv::setIdentity(kf_.measurementMatrix);
    }

    void init(const cv::Rect2f& box) {
        float cx = box.x + box.width * 0.5f;
        float cy = box.y + box.height * 0.5f;
        float a  = box.width / std::max(1.0f, box.height);
        float h  = box.height;

        kf_.statePost.setTo(0);
        kf_.statePost.at<float>(0) = cx;
        kf_.statePost.at<float>(1) = cy;
        kf_.statePost.at<float>(2) = a;
        kf_.statePost.at<float>(3) = h;

        float pos_std = 2.0f * std_weight_position_ * h;
        float vel_std = 10.0f * std_weight_velocity_ * h;

        kf_.errorCovPost.setTo(0);
        kf_.errorCovPost.at<float>(0, 0) = pos_std * pos_std;
        kf_.errorCovPost.at<float>(1, 1) = pos_std * pos_std;
        kf_.errorCovPost.at<float>(2, 2) = 1e-4f;
        kf_.errorCovPost.at<float>(3, 3) = pos_std * pos_std;
        kf_.errorCovPost.at<float>(4, 4) = vel_std * vel_std;
        kf_.errorCovPost.at<float>(5, 5) = vel_std * vel_std;
        kf_.errorCovPost.at<float>(6, 6) = 1e-6f;
        kf_.errorCovPost.at<float>(7, 7) = vel_std * vel_std;
    }

    cv::Rect2f predict() {
        float h = kf_.statePost.at<float>(3);
        float pos_std = std_weight_position_ * h;
        float vel_std = std_weight_velocity_ * h;

        kf_.processNoiseCov.setTo(0);
        kf_.processNoiseCov.at<float>(0, 0) = pos_std * pos_std;
        kf_.processNoiseCov.at<float>(1, 1) = pos_std * pos_std;
        kf_.processNoiseCov.at<float>(2, 2) = 1e-4f;
        kf_.processNoiseCov.at<float>(3, 3) = pos_std * pos_std;
        kf_.processNoiseCov.at<float>(4, 4) = vel_std * vel_std;
        kf_.processNoiseCov.at<float>(5, 5) = vel_std * vel_std;
        kf_.processNoiseCov.at<float>(6, 6) = 1e-6f;
        kf_.processNoiseCov.at<float>(7, 7) = vel_std * vel_std;

        const cv::Mat& pred = kf_.predict();
        return stateToBbox(pred);
    }

    cv::Rect2f update(const cv::Rect2f& box) {
        float h = box.height;
        float pos_std = std_weight_position_ * h;

        kf_.measurementNoiseCov.setTo(0);
        kf_.measurementNoiseCov.at<float>(0, 0) = pos_std * pos_std;
        kf_.measurementNoiseCov.at<float>(1, 1) = pos_std * pos_std;
        kf_.measurementNoiseCov.at<float>(2, 2) = 1e-2f;
        kf_.measurementNoiseCov.at<float>(3, 3) = pos_std * pos_std;

        cv::Mat meas(4, 1, CV_32F);
        meas.at<float>(0) = box.x + box.width * 0.5f;
        meas.at<float>(1) = box.y + box.height * 0.5f;
        meas.at<float>(2) = box.width / std::max(1.0f, box.height);
        meas.at<float>(3) = box.height;

        const cv::Mat& corrected = kf_.correct(meas);
        return stateToBbox(corrected);
    }

    cv::Rect2f stateToBbox(const cv::Mat& state) const {
        float cx = state.at<float>(0);
        float cy = state.at<float>(1);
        float a  = std::max(0.01f, state.at<float>(2));
        float h  = std::max(1.0f, state.at<float>(3));
        float w  = a * h;
        return cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
    }
};

// --- Track ------------------------------------------------------------------
struct Track2D {
    int id;
    TrackState state = TrackState::Tentative;
    ByteKalmanFilter kf;
    cv::Rect2f bbox;
    int hits = 1;
    int age = 0;
    int time_since_update = 0;

    // Added vs. the ROS version: class + score of the last matched detection,
    // so the overlay can show them next to the ID. Not used by the tracking logic.
    int   label = 0;
    float score = 0.0f;

    Track2D(int track_id, const Detection& det, float w_pos, float w_vel)
        : id(track_id), kf(w_pos, w_vel), bbox(det.bbox), label(det.label), score(det.score) {
        kf.init(det.bbox);
    }

    void predict() {
        bbox = kf.predict();
        age++;
        time_since_update++;
    }

    void update(const Detection& det, int confirm_hits) {
        bbox = kf.update(det.bbox);
        label = det.label;
        score = det.score;
        time_since_update = 0;
        hits++;
        if (state == TrackState::Tentative && hits >= confirm_hits) {
            state = TrackState::Confirmed;
        } else if (state == TrackState::Lost) {
            state = TrackState::Confirmed;
        }
    }

    void markMissed(int max_age) {
        if (time_since_update > max_age) {
            state = TrackState::Deleted;
        } else if (state == TrackState::Confirmed && time_since_update > 0) {
            state = TrackState::Lost;
        }
    }

    // True for the tracks that should be published/drawn this frame.
    bool isReportable() const {
        return state == TrackState::Confirmed && time_since_update == 0;
    }
};

// --- Kuhn-Munkres solver on flat vectors (no allocations once warmed up) -----
class FlatHungarianAlgorithm {
private:
    std::vector<float> a_;
    std::vector<float> u_, v_, minv_;
    std::vector<int> p_, way_;
    std::vector<bool> flagged_;

public:
    // costMatrix is row-major rows x cols. assignment[row] = col, or -1.
    void solve(const std::vector<float>& costMatrix, int rows, int cols, std::vector<int>& assignment) {
        assignment.assign(rows, -1);
        if (rows == 0 || cols == 0) return;

        int n = std::max(rows, cols);
        int stride = n + 1;

        if ((int)u_.size() < stride) {
            a_.assign(stride * stride, 0.0f);
            u_.resize(stride);
            v_.resize(stride);
            minv_.resize(stride);
            p_.resize(stride);
            way_.resize(stride);
            flagged_.resize(stride);
        }

        std::fill(a_.begin(), a_.begin() + (stride * stride), 0.0f);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                a_[(i + 1) * stride + (j + 1)] = costMatrix[i * cols + j];
            }
        }

        std::fill(u_.begin(), u_.begin() + stride, 0.0f);
        std::fill(v_.begin(), v_.begin() + stride, 0.0f);
        std::fill(p_.begin(), p_.begin() + stride, 0);
        std::fill(way_.begin(), way_.begin() + stride, 0);

        for (int i = 1; i <= n; ++i) {
            p_[0] = i;
            int j0 = 0;
            std::fill(minv_.begin(), minv_.begin() + stride, std::numeric_limits<float>::max());
            std::fill(flagged_.begin(), flagged_.begin() + stride, false);

            do {
                flagged_[j0] = true;
                int i0 = p_[j0], j1 = 0;
                float delta = std::numeric_limits<float>::max();

                for (int j = 1; j <= n; ++j) {
                    if (!flagged_[j]) {
                        float cur = a_[i0 * stride + j] - u_[i0] - v_[j];
                        if (cur < minv_[j]) { minv_[j] = cur; way_[j] = j0; }
                        if (minv_[j] < delta) { delta = minv_[j]; j1 = j; }
                    }
                }
                for (int j = 0; j <= n; ++j) {
                    if (flagged_[j]) { u_[p_[j]] += delta; v_[j] -= delta; }
                    else { minv_[j] -= delta; }
                }
                j0 = j1;
            } while (p_[j0] != 0);

            do {
                int j1 = way_[j0];
                p_[j0] = p_[j1];
                j0 = j1;
            } while (j0);
        }

        for (int j = 1; j <= n; ++j) {
            if (p_[j] != 0 && p_[j] <= rows && j <= cols) {
                assignment[p_[j] - 1] = j - 1;
            }
        }
    }
};

// --- Overlap helpers --------------------------------------------------------
inline float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    float inter_area = inter.area();
    float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

inline float computeGIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1_a = a.x, y1_a = a.y, x2_a = a.x + a.width, y2_a = a.y + a.height;
    float x1_b = b.x, y1_b = b.y, x2_b = b.x + b.width, y2_b = b.y + b.height;

    float x1_i = std::max(x1_a, x1_b);
    float y1_i = std::max(y1_a, y1_b);
    float x2_i = std::min(x2_a, x2_b);
    float y2_i = std::min(y2_a, y2_b);

    float inter_w = std::max(0.0f, x2_i - x1_i);
    float inter_h = std::max(0.0f, y2_i - y1_i);
    float inter_area = inter_w * inter_h;

    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    float iou = inter_area / union_area;

    float x1_c = std::min(x1_a, x1_b);
    float y1_c = std::min(y1_a, y1_b);
    float x2_c = std::max(x2_a, x2_b);
    float y2_c = std::max(y2_a, y2_b);

    float c_w = std::max(0.0f, x2_c - x1_c);
    float c_h = std::max(0.0f, y2_c - y1_c);
    float enclosing_area = c_w * c_h;

    if (enclosing_area <= 0.0f) return iou;

    return iou - ((enclosing_area - union_area) / enclosing_area);   // [-1, 1]
}

// --- Multi-camera tracker ---------------------------------------------------
// One independent track list per camera; track IDs are unique across cameras
// (shared counter), same as the ROS node. There is no cross-camera association.
// Not thread-safe: call update() from a single thread.
class MultiCamByteTracker {
public:
    MultiCamByteTracker(int numCams, const Tracker2DConfig& cfg = Tracker2DConfig())
        : cfg_(cfg), camera_tracks_(numCams) {
        for (auto& v : camera_tracks_) v.reserve(128);
        dets_high_.reserve(128);
        dets_low_.reserve(128);
        cost_matrix_.reserve(128 * 128);
        assignment_.reserve(128);
        unmatched_tracks_.reserve(128);
        unmatched_high_dets_.reserve(128);
        still_unmatched_tracks_.reserve(128);
        det_matched_.reserve(128);
    }

    const Tracker2DConfig& config() const { return cfg_; }
    int numCams() const { return (int)camera_tracks_.size(); }

    // Tracks of one camera after the latest update() (includes Tentative/Lost ones;
    // filter with Track2D::isReportable() for the ones to publish/draw).
    const std::vector<Track2D>& tracks(int cam) const { return camera_tracks_[cam]; }

    // Advance camera `cam` by one frame. `dets` = every detection of that frame
    // in original-image pixel coordinates (anything under low_thresh is ignored).
    void update(int cam, const std::vector<Detection>& dets) {
        if (cam < 0 || cam >= (int)camera_tracks_.size()) return;
        auto& tracks = camera_tracks_[cam];

        // 1. Predict state for all tracks
        for (auto& t : tracks) t.predict();

        // 2. Split detections into High & Low score pools
        dets_high_.clear();
        dets_low_.clear();
        for (const auto& det : dets) {
            if (det.label > cfg_.label_max) continue;
            if (det.score >= cfg_.high_thresh)      dets_high_.push_back(det);
            else if (det.score >= cfg_.low_thresh)  dets_low_.push_back(det);
        }

        unmatched_tracks_.clear();
        unmatched_high_dets_.clear();

        // =====================================================================
        // STAGE 1: tracks vs HIGH-confidence detections (cost = 1 - GIoU)
        // =====================================================================
        const int n_tracks = (int)tracks.size();
        const int n_high   = (int)dets_high_.size();

        if (n_tracks > 0 && n_high > 0) {
            cost_matrix_.resize((size_t)n_tracks * n_high);
            for (int ti = 0; ti < n_tracks; ++ti) {
                for (int di = 0; di < n_high; ++di) {
                    cost_matrix_[ti * n_high + di] = 1.0f - computeGIoU(tracks[ti].bbox, dets_high_[di].bbox);
                }
            }

            hungarian_solver_.solve(cost_matrix_, n_tracks, n_high, assignment_);

            det_matched_.assign(n_high, 0);
            for (int ti = 0; ti < n_tracks; ++ti) {
                int di = assignment_[ti];
                if (di >= 0 && di < n_high && cost_matrix_[ti * n_high + di] <= cfg_.stage1_cost_thresh) {
                    tracks[ti].update(dets_high_[di], cfg_.confirm_hits);
                    det_matched_[di] = 1;
                } else {
                    unmatched_tracks_.push_back(ti);
                }
            }
            for (int di = 0; di < n_high; ++di) {
                if (!det_matched_[di]) unmatched_high_dets_.push_back(di);
            }
        } else {
            for (int ti = 0; ti < n_tracks; ++ti) unmatched_tracks_.push_back(ti);
            for (int di = 0; di < n_high; ++di)   unmatched_high_dets_.push_back(di);
        }

        // =====================================================================
        // STAGE 2: leftover tracks vs LOW-confidence detections (cost = 1 - IoU)
        // =====================================================================
        still_unmatched_tracks_.clear();
        const int n_unmatched = (int)unmatched_tracks_.size();
        const int n_low       = (int)dets_low_.size();

        if (n_unmatched > 0 && n_low > 0) {
            cost_matrix_.resize((size_t)n_unmatched * n_low);
            for (int ui = 0; ui < n_unmatched; ++ui) {
                int ti = unmatched_tracks_[ui];
                for (int di = 0; di < n_low; ++di) {
                    cost_matrix_[ui * n_low + di] = 1.0f - computeIoU(tracks[ti].bbox, dets_low_[di].bbox);
                }
            }

            hungarian_solver_.solve(cost_matrix_, n_unmatched, n_low, assignment_);

            const float stage2_max_cost = 1.0f - cfg_.stage2_iou_thresh;
            for (int ui = 0; ui < n_unmatched; ++ui) {
                int di = assignment_[ui];
                int ti = unmatched_tracks_[ui];
                if (di >= 0 && di < n_low && cost_matrix_[ui * n_low + di] <= stage2_max_cost) {
                    tracks[ti].update(dets_low_[di], cfg_.confirm_hits);
                } else {
                    still_unmatched_tracks_.push_back(ti);
                }
            }
        } else {
            still_unmatched_tracks_ = unmatched_tracks_;
        }

        // 3. Age / lose the tracks nobody claimed
        for (int ti : still_unmatched_tracks_) tracks[ti].markMissed(cfg_.max_age);

        // 4. Spawn new tracks ONLY from unmatched HIGH-confidence detections
        for (int di : unmatched_high_dets_) {
            tracks.emplace_back(next_track_id_++, dets_high_[di], cfg_.std_weight_pos, cfg_.std_weight_vel);
        }

        // 5. Purge dead tracks
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [](const Track2D& t) { return t.state == TrackState::Deleted; }),
                     tracks.end());
    }

    // Drop every track of every camera (e.g. after a long stall). IDs keep counting up.
    void reset() {
        for (auto& v : camera_tracks_) v.clear();
    }

private:
    Tracker2DConfig cfg_;
    FlatHungarianAlgorithm hungarian_solver_;
    std::vector<std::vector<Track2D>> camera_tracks_;
    int next_track_id_ = 0;

    // Persistent scratch buffers (allocated once, cleared every frame)
    std::vector<Detection> dets_high_;
    std::vector<Detection> dets_low_;
    std::vector<float> cost_matrix_;
    std::vector<int> assignment_;
    std::vector<int> unmatched_tracks_;
    std::vector<int> unmatched_high_dets_;
    std::vector<int> still_unmatched_tracks_;
    std::vector<char> det_matched_;
};

} // namespace tracker2d