#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
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
    RosConfig ros;
    MaskOptions mask_options;
    float confidence_threshold = 0.45f;
    WebServerConfig web_server;

    PipelineConfig() {
        // Default to single_channel = true for maskcodec mono8 encoding
        mask_options.single_channel = true;
        mask_options.mode = MaskMode::BBOX;
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

