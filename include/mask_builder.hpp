/****************************************************************************
 * mask_builder.hpp
 *
 * Post-inference mask construction for main_m_cam.cpp. Does NOT touch the
 * inference flow -- it only replaces the per-detection
 *     cv::rectangle(masks_[i], det.bbox, cv::Scalar(id, id, label), -1)
 * call with options that are cheaper to publish on AGX Orin and/or carry a
 * tighter obstacle shape than a raw bbox.
 *
 * Consumes ONLY det.bbox (cv::Rect) and det.label (int) from your Detection
 * type -- no other fields assumed.
 *
 * Encoding:
 *   3-channel (CV_8UC3): channel G (pix[1]) = instance id (1..N),
 *                        channel R (pix[2]) = class label.  (B unused.)
 *   mono (CV_8UC1):      one byte = maskcodec::encode(instance, class)
 *                        bits[7:3] instance (1..31), bits[2:0] class (0..7),
 *                        0 = background. The fusion node MUST decode the same
 *                        way (maskcodec::decode) -- both share mask_codec.hpp.
 *
 * Everything is flag-driven with a hard bbox fallback. Compile-tested logic
 * lives in mask_builder_selfcheck.cpp (geometry only; no OpenCV needed there).
 ****************************************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "mask_codec.hpp"   // shared mono8 (instance,class) packing -- MUST match the fusion decoder

// ----- mask mode ------------------------------------------------------------
enum class MaskMode {
    BBOX,        // original: fill the full bounding rectangle (safe fallback)
    SHAPE_HULL,  // convex hull of FAST keypoints (needs texture)
    SHAPE_RROT,  // rotated min-area rect of FAST keypoints (needs texture)
    SHAPE_EDGE   // Canny silhouette -> convex hull (texture-INDEPENDENT; best for
                 // low-distortion wide lenses where edges stay straight)
};

struct MaskOptions {
    MaskMode mode          = MaskMode::BBOX;   // default = original behaviour
    bool     single_channel = false;          // CV_8UC1 instead of CV_8UC3 (3x smaller)
    int      min_box_px    = 8;                // boxes smaller than this -> bbox fill
    float    min_shape_fill = 0.20f;           // if fitted shape area < this*bbox area,
                                               // distrust it and fall back to bbox
    int      max_keypoints  = 300;             // cap FAST keypoints per box (Orin budget)
    // SHAPE_EDGE params:
    int      canny_lo      = 50;               // Canny low threshold
    int      canny_hi      = 150;              // Canny high threshold
    int      close_ksize   = 5;                // dilation to close edge gaps (odd)
};

// ---------------------------------------------------------------------------
// Internal: collect cheap shape evidence inside a box (FAST corners on the
// box ROI). Returns image-space points. Bounded by max_keypoints for cost.
// ---------------------------------------------------------------------------
inline std::vector<cv::Point> shapePointsInBox(const cv::Mat& gray, const cv::Rect& box,
                                               int max_keypoints) {
    std::vector<cv::Point> pts;
    cv::Mat roi = gray(box);
    std::vector<cv::KeyPoint> kps;
    cv::FAST(roi, kps, 20 /*threshold*/, true /*nonmax*/);
    if ((int)kps.size() > max_keypoints) {
        std::nth_element(kps.begin(), kps.begin() + max_keypoints, kps.end(),
                         [](const cv::KeyPoint& a, const cv::KeyPoint& b){ return a.response > b.response; });
        kps.resize(max_keypoints);
    }
    pts.reserve(kps.size());
    for (const auto& k : kps) pts.emplace_back(cvRound(k.pt.x) + box.x, cvRound(k.pt.y) + box.y);
    return pts;
}

// ---------------------------------------------------------------------------
// Internal: Canny silhouette inside a box -> image-space boundary points of the
// largest edge component. Texture-independent; relies on the object boundary.
// ---------------------------------------------------------------------------
inline std::vector<cv::Point> edgePointsInBox(const cv::Mat& gray, const cv::Rect& box,
                                              int lo, int hi, int close_k) {
    cv::Mat roi = gray(box), edges;
    cv::Canny(roi, edges, lo, hi);
    if (close_k > 1) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close_k, close_k));
        cv::dilate(edges, edges, k);   // close gaps so the silhouette is one component
    }
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(edges, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cs.empty()) return {};
    size_t best = 0; double ba = -1;
    for (size_t i = 0; i < cs.size(); ++i) { double a = cv::contourArea(cs[i]); if (a > ba) { ba = a; best = i; } }
    std::vector<cv::Point> out = cs[best];
    for (auto& p : out) { p.x += box.x; p.y += box.y; }
    return out;
}

// ---------------------------------------------------------------------------
// Paint ONE detection into the mask. `gray` is the grayscale source frame
// (only needed for SHAPE_* modes; pass an empty Mat for BBOX). `inst_id` is the
// per-image instance index (>=1), `label` the class.
//
// Falls back to a filled bbox whenever shape evidence is weak/empty or the
// fitted region is implausibly small -- so a bad fit never drops the obstacle.
// ---------------------------------------------------------------------------
inline void paintDetection(cv::Mat& mask, const cv::Mat& gray, const cv::Rect& bbox_in,
                           int inst_id, int label, const MaskOptions& opt) {
    // clip bbox to image
    cv::Rect bbox = bbox_in & cv::Rect(0, 0, mask.cols, mask.rows);
    if (bbox.width <= 0 || bbox.height <= 0) return;

    const bool mono = opt.single_channel;
    // MONO: pack (instance,class) into one byte via the shared codec so the
    // fusion node can recover BOTH. bits[7:3]=instance(1..31), bits[2:0]=class.
    // (Was cv::Scalar(inst_id) -- instance only -- which the fusion decoder,
    // expecting a packed byte, read as inst=inst_id>>3, dropping ids 1..7.)
    cv::Scalar color = mono ? cv::Scalar(maskcodec::encode(inst_id, label))
                            : cv::Scalar(0, inst_id, label);   // B,G,R = 0,id,label

    auto fillBBox = [&]() { cv::rectangle(mask, bbox, color, -1); };

    if (opt.mode == MaskMode::BBOX || bbox.width < opt.min_box_px ||
        bbox.height < opt.min_box_px || gray.empty()) {
        fillBBox();
        return;
    }

    const double bboxArea = (double)bbox.width * bbox.height;
    std::vector<std::vector<cv::Point>> poly(1);

    if (opt.mode == MaskMode::SHAPE_EDGE) {
        std::vector<cv::Point> ep = edgePointsInBox(gray, bbox, opt.canny_lo, opt.canny_hi, opt.close_ksize);
        if (ep.size() < 3) { fillBBox(); return; }
        std::vector<int> idx; cv::convexHull(ep, idx);   // hull = robust filled silhouette
        for (int id : idx) poly[0].push_back(ep[id]);
    } else {
        std::vector<cv::Point> pts = shapePointsInBox(gray, bbox, opt.max_keypoints);
        if (pts.size() < 3) { fillBBox(); return; }
        if (opt.mode == MaskMode::SHAPE_RROT) {
            cv::RotatedRect rr = cv::minAreaRect(pts);
            cv::Point2f v[4]; rr.points(v);
            for (int k = 0; k < 4; ++k) poly[0].emplace_back(cvRound(v[k].x), cvRound(v[k].y));
        } else { // SHAPE_HULL
            std::vector<int> idx; cv::convexHull(pts, idx);
            for (int id : idx) poly[0].push_back(pts[id]);
        }
    }

    double area = std::fabs(cv::contourArea(poly[0]));
    if (area < opt.min_shape_fill * bboxArea) { fillBBox(); return; }  // distrust -> bbox

    cv::fillPoly(mask, poly, color);
}

// ---------------------------------------------------------------------------
// Allocate/clear a mask matching the frame, in the requested channel layout.
// Call once per frame per camera before painting detections into it.
// ---------------------------------------------------------------------------
inline void prepareMask(cv::Mat& mask, const cv::Size& sz, bool single_channel) {
    int type = single_channel ? CV_8UC1 : CV_8UC3;
    if (mask.size() != sz || mask.type() != type) mask = cv::Mat::zeros(sz, type);
    else mask.setTo(0);
}

// /****************************************************************************
//  * mask_builder.hpp
//  *
//  * Post-inference mask construction for main_m_cam.cpp. Does NOT touch the
//  * inference flow -- it only replaces the per-detection
//  *     cv::rectangle(masks_[i], det.bbox, cv::Scalar(id, id, label), -1)
//  * call with options that are cheaper to publish on AGX Orin and/or carry a
//  * tighter obstacle shape than a raw bbox.
//  *
//  * Consumes ONLY det.bbox (cv::Rect) and det.label (int) from your Detection
//  * type -- no other fields assumed.
//  *
//  * Encoding kept identical to the original so the fusion node still reads it:
//  *   channel G (pix[1]) = instance id (1..N)
//  *   channel R (pix[2]) = class label
//  * (B unused.) In MONO mode only the instance id is written (see notes).
//  *
//  * Everything is flag-driven with a hard bbox fallback. Compile-tested logic
//  * lives in mask_builder_selfcheck.cpp (geometry only; no OpenCV needed there).
//  ****************************************************************************/
// #pragma once
// #include <opencv2/opencv.hpp>
// #include <vector>

// // ----- mask mode ------------------------------------------------------------
// enum class MaskMode {
//     BBOX,        // original: fill the full bounding rectangle (safe fallback)
//     SHAPE_HULL,  // convex hull of FAST keypoints (needs texture)
//     SHAPE_RROT,  // rotated min-area rect of FAST keypoints (needs texture)
//     SHAPE_EDGE   // Canny silhouette -> convex hull (texture-INDEPENDENT; best for
//                  // low-distortion wide lenses where edges stay straight)
// };

// struct MaskOptions {
//     MaskMode mode          = MaskMode::BBOX;   // default = original behaviour
//     bool     single_channel = false;          // CV_8UC1 instead of CV_8UC3 (3x smaller)
//     int      min_box_px    = 8;                // boxes smaller than this -> bbox fill
//     float    min_shape_fill = 0.20f;           // if fitted shape area < this*bbox area,
//                                                // distrust it and fall back to bbox
//     int      max_keypoints  = 300;             // cap FAST keypoints per box (Orin budget)
//     // SHAPE_EDGE params:
//     int      canny_lo      = 50;               // Canny low threshold
//     int      canny_hi      = 150;              // Canny high threshold
//     int      close_ksize   = 5;                // dilation to close edge gaps (odd)
// };

// // Optional diagnostics: how often shape-fit fired vs fell back to bbox. Read
// // these in the node to confirm SHAPE_* is actually working instead of guessing
// // from the visualization. Reset per frame if you like.
// struct MaskStats { int shape = 0, bbox_fallback = 0;
//     void reset() { shape = bbox_fallback = 0; }
//     float shape_ratio() const { int t = shape + bbox_fallback; return t ? (float)shape / t : 0.f; } };

// // ---------------------------------------------------------------------------
// // Internal: collect cheap shape evidence inside a box (FAST corners on the
// // box ROI). Returns image-space points. Bounded by max_keypoints for cost.
// // ---------------------------------------------------------------------------
// inline std::vector<cv::Point> shapePointsInBox(const cv::Mat& gray, const cv::Rect& box,
//                                                int max_keypoints) {
//     std::vector<cv::Point> pts;
//     cv::Mat roi = gray(box);
//     std::vector<cv::KeyPoint> kps;
//     cv::FAST(roi, kps, 20 /*threshold*/, true /*nonmax*/);
//     if ((int)kps.size() > max_keypoints) {
//         std::nth_element(kps.begin(), kps.begin() + max_keypoints, kps.end(),
//                          [](const cv::KeyPoint& a, const cv::KeyPoint& b){ return a.response > b.response; });
//         kps.resize(max_keypoints);
//     }
//     pts.reserve(kps.size());
//     for (const auto& k : kps) pts.emplace_back(cvRound(k.pt.x) + box.x, cvRound(k.pt.y) + box.y);
//     return pts;
// }

// // ---------------------------------------------------------------------------
// // Internal: Canny silhouette inside a box -> image-space boundary points of the
// // largest edge component. Texture-independent; relies on the object boundary.
// // ---------------------------------------------------------------------------
// inline std::vector<cv::Point> edgePointsInBox(const cv::Mat& gray, const cv::Rect& box,
//                                               int lo, int hi, int close_k) {
//     cv::Mat roi = gray(box), edges;
//     cv::Canny(roi, edges, lo, hi);
//     if (close_k > 1) {
//         cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close_k, close_k));
//         cv::dilate(edges, edges, k);   // close gaps so the silhouette is one component
//     }
//     std::vector<std::vector<cv::Point>> cs;
//     cv::findContours(edges, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//     if (cs.empty()) return {};
//     size_t best = 0; double ba = -1;
//     for (size_t i = 0; i < cs.size(); ++i) { double a = cv::contourArea(cs[i]); if (a > ba) { ba = a; best = i; } }
//     std::vector<cv::Point> out = cs[best];
//     for (auto& p : out) { p.x += box.x; p.y += box.y; }
//     return out;
// }

// // ---------------------------------------------------------------------------
// // Paint ONE detection into the mask. `gray` is the grayscale source frame
// // (only needed for SHAPE_* modes; pass an empty Mat for BBOX). `inst_id` is the
// // per-image instance index (>=1), `label` the class.
// //
// // Falls back to a filled bbox whenever shape evidence is weak/empty or the
// // fitted region is implausibly small -- so a bad fit never drops the obstacle.
// // ---------------------------------------------------------------------------
// inline void paintDetection(cv::Mat& mask, const cv::Mat& gray, const cv::Rect& bbox_in,
//                            int inst_id, int label, const MaskOptions& opt,
//                            MaskStats* stats = nullptr) {
//     // clip bbox to image
//     cv::Rect bbox = bbox_in & cv::Rect(0, 0, mask.cols, mask.rows);
//     if (bbox.width <= 0 || bbox.height <= 0) return;

//     const bool mono = opt.single_channel;
//     cv::Scalar color = mono ? cv::Scalar(inst_id)
//                             : cv::Scalar(0, inst_id, label);   // B,G,R = 0,id,label

//     auto fillBBox = [&]() { cv::rectangle(mask, bbox, color, -1); if (stats) stats->bbox_fallback++; };

//     if (opt.mode == MaskMode::BBOX || bbox.width < opt.min_box_px ||
//         bbox.height < opt.min_box_px || gray.empty()) {
//         fillBBox();
//         return;
//     }

//     const double bboxArea = (double)bbox.width * bbox.height;
//     std::vector<std::vector<cv::Point>> poly(1);

//     if (opt.mode == MaskMode::SHAPE_EDGE) {
//         std::vector<cv::Point> ep = edgePointsInBox(gray, bbox, opt.canny_lo, opt.canny_hi, opt.close_ksize);
//         if (ep.size() < 3) { fillBBox(); return; }
//         std::vector<int> idx; cv::convexHull(ep, idx);   // hull = robust filled silhouette
//         for (int id : idx) poly[0].push_back(ep[id]);
//     } else {
//         std::vector<cv::Point> pts = shapePointsInBox(gray, bbox, opt.max_keypoints);
//         if (pts.size() < 3) { fillBBox(); return; }
//         if (opt.mode == MaskMode::SHAPE_RROT) {
//             cv::RotatedRect rr = cv::minAreaRect(pts);
//             cv::Point2f v[4]; rr.points(v);
//             for (int k = 0; k < 4; ++k) poly[0].emplace_back(cvRound(v[k].x), cvRound(v[k].y));
//         } else { // SHAPE_HULL
//             std::vector<int> idx; cv::convexHull(pts, idx);
//             for (int id : idx) poly[0].push_back(pts[id]);
//         }
//     }

//     double area = std::fabs(cv::contourArea(poly[0]));
//     if (area < opt.min_shape_fill * bboxArea) { fillBBox(); return; }  // distrust -> bbox

//     cv::fillPoly(mask, poly, color);
//     if (stats) stats->shape++;
// }

// // ---------------------------------------------------------------------------
// // Allocate/clear a mask matching the frame, in the requested channel layout.
// // Call once per frame per camera before painting detections into it.
// // ---------------------------------------------------------------------------
// inline void prepareMask(cv::Mat& mask, const cv::Size& sz, bool single_channel) {
//     int type = single_channel ? CV_8UC1 : CV_8UC3;
//     if (mask.size() != sz || mask.type() != type) mask = cv::Mat::zeros(sz, type);
//     else mask.setTo(0);
// }