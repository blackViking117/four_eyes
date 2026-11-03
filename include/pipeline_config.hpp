#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "mask_builder.hpp"

// Forward declaration of ROS types to avoid requiring ROS headers if included elsewhere
namespace ros {
    class NodeHandle;
}

struct RosConfig {
    bool enabled = true;
    std::string node_name = "four_eyes_perception";
    std::string mask_topic = "/obstacle_mask";
    std::string frame_id = "camera_array";
    int queue_size = 1;
};

struct WebServerConfig {
    bool enabled = true;
    int port = 8080;
};

struct CameraSetting {
    std::string exposure_auto = "Off";  // "Off", "Once", "Continuous"
    double exposure_time = 2000.0;      // microseconds
    std::string gain_auto = "Off";      // "Off", "Once", "Continuous"
    double gain = 0.0;                  // dB
};

struct TrackingConfig {
    bool enabled = true;
};

struct SaveFramesConfig {
    bool enabled = false;
    std::string directory = "saved_frames";
    bool save_pre_inference = true;
    bool save_detections = true;
    bool save_masks = true;
    bool save_combined_mask = true;
    std::string format_images = "jpg"; // "jpg" or "png"
    int jpeg_quality = 95;
    size_t max_queue_size = 500;
};

struct PipelineConfig {
    std::string engine_file = "/media/swdep/500GB_SSD_PERCEP/black_magic/yolov10x_dynamic_fp32.engine";
    std::vector<std::string> camera_serials = {
        "251102011",
        "251102012",
        "251102016",
        "251400510"
    };
    std::set<std::string> rotate_180_serials = {
        "251400510",
        "251102016"
    };
    TrackingConfig tracking;
    SaveFramesConfig save_frames;
    CameraSetting camera_defaults;
    std::map<std::string, CameraSetting> camera_settings;
    RosConfig ros;
    MaskOptions mask_options;
    float confidence_threshold = 0.45f;
    WebServerConfig web_server;

    PipelineConfig() {
        // Default to single_channel = true for maskcodec mono8 encoding
        mask_options.single_channel = true;
        mask_options.mode = MaskMode::BBOX;
    }

    CameraSetting getCameraSetting(const std::string& serial) const {
        auto it = camera_settings.find(serial);
        if (it != camera_settings.end()) {
            return it->second;
        }
        return camera_defaults;
    }

    static PipelineConfig loadFromFile(const std::string& path) {
        PipelineConfig cfg;
        try {
            YAML::Node doc = YAML::LoadFile(path);

            if (doc["engine_file"]) {
                cfg.engine_file = doc["engine_file"].as<std::string>();
            } else if (doc["engine_path"]) {
                cfg.engine_file = doc["engine_path"].as<std::string>();
            }

            if (doc["camera_serials"]) {
                cfg.camera_serials.clear();
                for (const auto& s : doc["camera_serials"]) {
                    cfg.camera_serials.push_back(s.as<std::string>());
                }
            }

            if (doc["rotate_180_serials"]) {
                cfg.rotate_180_serials.clear();
                for (const auto& s : doc["rotate_180_serials"]) {
                    cfg.rotate_180_serials.insert(s.as<std::string>());
                }
            }

            // Tracking configuration
            if (doc["tracking"]) {
                if (doc["tracking"].IsMap()) {
                    if (doc["tracking"]["enabled"]) {
                        cfg.tracking.enabled = doc["tracking"]["enabled"].as<bool>();
                    }
                } else if (doc["tracking"].IsScalar()) {
                    cfg.tracking.enabled = doc["tracking"].as<bool>();
                }
            } else if (doc["enable_tracking"]) {
                cfg.tracking.enabled = doc["enable_tracking"].as<bool>();
            }

            // Helper to parse CameraSetting from a YAML node
            auto parseCamSetting = [](const YAML::Node& node, CameraSetting& s) {
                if (node["exposure_auto"]) s.exposure_auto = node["exposure_auto"].as<std::string>();
                else if (node["ExposureAuto"]) s.exposure_auto = node["ExposureAuto"].as<std::string>();

                if (node["exposure_time"]) s.exposure_time = node["exposure_time"].as<double>();
                else if (node["exposure_time_us"]) s.exposure_time = node["exposure_time_us"].as<double>();
                else if (node["ExposureTime"]) s.exposure_time = node["ExposureTime"].as<double>();

                if (node["gain_auto"]) s.gain_auto = node["gain_auto"].as<std::string>();
                else if (node["GainAuto"]) s.gain_auto = node["GainAuto"].as<std::string>();

                if (node["gain"]) s.gain = node["gain"].as<double>();
                else if (node["gain_db"]) s.gain = node["gain_db"].as<double>();
                else if (node["Gain"]) s.gain = node["Gain"].as<double>();
            };

            // Global camera default settings
            if (doc["camera_defaults"]) {
                parseCamSetting(doc["camera_defaults"], cfg.camera_defaults);
            }

            // Per-camera settings map: camera_settings: { "serial": { ... } }
            if (doc["camera_settings"] && doc["camera_settings"].IsMap()) {
                for (const auto& it : doc["camera_settings"]) {
                    std::string serial = it.first.as<std::string>();
                    CameraSetting s = cfg.camera_defaults; // initialize with defaults
                    parseCamSetting(it.second, s);
                    cfg.camera_settings[serial] = s;
                }
            }

            // Support camera list with per-camera settings: cameras: [ { serial: "...", ... } ]
            if (doc["cameras"] && doc["cameras"].IsSequence()) {
                for (const auto& item : doc["cameras"]) {
                    if (item["serial"]) {
                        std::string serial = item["serial"].as<std::string>();
                        CameraSetting s = cfg.camera_defaults;
                        parseCamSetting(item, s);
                        cfg.camera_settings[serial] = s;
                    }
                }
            }

            if (doc["ros"]) {
                auto r = doc["ros"];
                if (r["enabled"])    cfg.ros.enabled = r["enabled"].as<bool>();
                if (r["node_name"])  cfg.ros.node_name = r["node_name"].as<std::string>();
                if (r["mask_topic"]) cfg.ros.mask_topic = r["mask_topic"].as<std::string>();
                if (r["frame_id"])   cfg.ros.frame_id = r["frame_id"].as<std::string>();
                if (r["queue_size"]) cfg.ros.queue_size = r["queue_size"].as<int>();
            }

            if (doc["mask"]) {
                auto m = doc["mask"];
                if (m["single_channel"]) cfg.mask_options.single_channel = m["single_channel"].as<bool>();
                if (m["mode"]) {
                    std::string modeStr = m["mode"].as<std::string>();
                    if (modeStr == "SHAPE_HULL")      cfg.mask_options.mode = MaskMode::SHAPE_HULL;
                    else if (modeStr == "SHAPE_RROT") cfg.mask_options.mode = MaskMode::SHAPE_RROT;
                    else if (modeStr == "SHAPE_EDGE") cfg.mask_options.mode = MaskMode::SHAPE_EDGE;
                    else                             cfg.mask_options.mode = MaskMode::BBOX;
                }
                if (m["confidence_threshold"]) cfg.confidence_threshold = m["confidence_threshold"].as<float>();
                if (m["min_box_px"])         cfg.mask_options.min_box_px = m["min_box_px"].as<int>();
                if (m["min_shape_fill"])     cfg.mask_options.min_shape_fill = m["min_shape_fill"].as<float>();
                if (m["max_keypoints"])      cfg.mask_options.max_keypoints = m["max_keypoints"].as<int>();
                if (m["canny_lo"])           cfg.mask_options.canny_lo = m["canny_lo"].as<int>();
                if (m["canny_hi"])           cfg.mask_options.canny_hi = m["canny_hi"].as<int>();
                if (m["close_ksize"])        cfg.mask_options.close_ksize = m["close_ksize"].as<int>();
            }

            if (doc["web_server"]) {
                auto w = doc["web_server"];
                if (w["enabled"]) cfg.web_server.enabled = w["enabled"].as<bool>();
                if (w["port"])    cfg.web_server.port = w["port"].as<int>();
            }

            // Frame saver configuration
            if (doc["save_frames"]) {
                if (doc["save_frames"].IsMap()) {
                    auto sf = doc["save_frames"];
                    if (sf["enabled"]) cfg.save_frames.enabled = sf["enabled"].as<bool>();
                    if (sf["directory"]) cfg.save_frames.directory = sf["directory"].as<std::string>();
                    else if (sf["dir"]) cfg.save_frames.directory = sf["dir"].as<std::string>();
                    else if (sf["path"]) cfg.save_frames.directory = sf["path"].as<std::string>();
                    else if (sf["location"]) cfg.save_frames.directory = sf["location"].as<std::string>();

                    if (sf["save_pre_inference"]) cfg.save_frames.save_pre_inference = sf["save_pre_inference"].as<bool>();
                    if (sf["save_detections"])    cfg.save_frames.save_detections = sf["save_detections"].as<bool>();
                    if (sf["save_masks"])         cfg.save_frames.save_masks = sf["save_masks"].as<bool>();
                    if (sf["save_combined_mask"]) cfg.save_frames.save_combined_mask = sf["save_combined_mask"].as<bool>();
                    if (sf["format_images"])      cfg.save_frames.format_images = sf["format_images"].as<std::string>();
                    if (sf["jpeg_quality"])       cfg.save_frames.jpeg_quality = sf["jpeg_quality"].as<int>();
                    if (sf["max_queue_size"])     cfg.save_frames.max_queue_size = sf["max_queue_size"].as<size_t>();
                } else if (doc["save_frames"].IsScalar()) {
                    cfg.save_frames.enabled = doc["save_frames"].as<bool>();
                }
            } else if (doc["enable_save_frames"]) {
                cfg.save_frames.enabled = doc["enable_save_frames"].as<bool>();
            }
            if (doc["save_directory"]) {
                cfg.save_frames.directory = doc["save_directory"].as<std::string>();
            }

            std::cout << "[CONFIG] Loaded configuration from: " << path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[CONFIG] Warning: Could not read config file '" << path 
                      << "' (" << e.what() << "). Using built-in defaults.\n";
        }
        return cfg;
    }

    void printSummary() const {
        std::cout << "\n============================================================\n";
        std::cout << "                 ACTIVE PIPELINE CONFIGURATION              \n";
        std::cout << "============================================================\n";
        std::cout << " Engine Path     : " << engine_file << "\n";
        std::cout << " Camera Serials  : [ ";
        for (size_t i = 0; i < camera_serials.size(); ++i) {
            std::cout << camera_serials[i] << (i + 1 < camera_serials.size() ? ", " : " ");
        }
        std::cout << "] (Count: " << camera_serials.size() << ")\n";
        std::cout << " Inverted Serials: [ ";
        size_t c = 0;
        for (const auto& s : rotate_180_serials) {
            std::cout << s << (++c < rotate_180_serials.size() ? ", " : " ");
        }
        std::cout << "]\n";
        std::cout << " Tracking        : " << (tracking.enabled ? "ENABLED (ByteTrack MultiCam)" : "DISABLED (Inference Only, Raw Detections in Mask)") << "\n";
        std::cout << " Frame Saver     : " << (save_frames.enabled ? "ENABLED" : "DISABLED") << "\n";
        if (save_frames.enabled) {
            std::cout << "   Save Directory: " << save_frames.directory << "\n";
            std::cout << "   Save Features : "
                      << (save_frames.save_pre_inference ? "Pre " : "")
                      << (save_frames.save_detections ? "Det " : "")
                      << (save_frames.save_masks ? "Masks " : "")
                      << (save_frames.save_combined_mask ? "CombinedMask " : "")
                      << "| Format: " << save_frames.format_images
                      << " (Q:" << save_frames.jpeg_quality << ")"
                      << " | MaxQueue: " << save_frames.max_queue_size << "\n";
        }
        std::cout << " Camera Sensor Settings:\n";
        std::cout << "   [Defaults]    : ExpAuto=" << camera_defaults.exposure_auto
                  << " | ExpTime=" << camera_defaults.exposure_time << " us"
                  << " | GainAuto=" << camera_defaults.gain_auto
                  << " | Gain=" << camera_defaults.gain << " dB\n";
        for (const auto& s : camera_serials) {
            CameraSetting cs = getCameraSetting(s);
            std::cout << "   [" << s << "] : ExpAuto=" << cs.exposure_auto
                      << " | ExpTime=" << cs.exposure_time << " us"
                      << " | GainAuto=" << cs.gain_auto
                      << " | Gain=" << cs.gain << " dB\n";
        }
        std::cout << " ROS 1 Node      : " << (ros.enabled ? "ENABLED" : "DISABLED") << "\n";
        if (ros.enabled) {
            std::cout << "   Node Name     : " << ros.node_name << "\n";
            std::cout << "   Mask Topic    : " << ros.mask_topic << "\n";
            std::cout << "   Frame ID      : " << ros.frame_id << "\n";
            std::cout << "   Queue Size    : " << ros.queue_size << "\n";
        }
        std::cout << " Mask Settings   :\n";
        std::cout << "   Single Channel: " << (mask_options.single_channel ? "true (mono8, mask_codec)" : "false (CV_8UC3)") << "\n";
        std::string modeStr = "BBOX";
        if (mask_options.mode == MaskMode::SHAPE_HULL) modeStr = "SHAPE_HULL";
        else if (mask_options.mode == MaskMode::SHAPE_RROT) modeStr = "SHAPE_RROT";
        else if (mask_options.mode == MaskMode::SHAPE_EDGE) modeStr = "SHAPE_EDGE";
        std::cout << "   Mode          : " << modeStr << "\n";
        std::cout << "   Conf Threshold: " << confidence_threshold << "\n";
        std::cout << "============================================================\n\n";
    }
};
